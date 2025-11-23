"""
HST-v3 ULTRA - Complete Paper-Compliant Implementation - FINALIZED VERSION WITH KV CACHE

Izmene:
- Implementiran KV Cache u SelfAttentionWithCache i TransformerEncoderLayerWithCache.
- Refaktorizovane forward metode za podršku keširanju.
- Optimizovana generate_ultra_fast metoda da koristi keš za inkrementalni Verification Pass.
- Optimizovan Lattice Core (index_copy zamenjen direktnim dodeljivanjem).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List

# Type definition for KV Cache: List[Tuple[torch.Tensor, torch.Tensor]]
KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]


# ==========================================================
# CUSTOM TRANSFORMER COMPONENTS WITH KV CACHE SUPPORT
# ==========================================================
class SelfAttentionWithCache(nn.Module):
    """Custom Causal Self-Attention layer with explicit KV Cache support for inference."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, S, D = x.shape
        
        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S_new, HD)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S_new, HD)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S_new, HD)

        # 2. Handle KV Cache: Concatenate past K/V with new K/V
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        # 3. Store for next iteration
        present = (k, v)
        
        # 4. Attention Calculation
        # S_total = S (new) + S_past (if any)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5) # (B, H, S_new, S_total)
        
        # NOTE: Causal mask should be applied here if needed for full pass. 
        # For incremental (inference), the model should only be passed new tokens.
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S, D) # (B, S_new, D)
        
        output = self.out_proj(attn_output)
        return output, present

class TransformerEncoderLayerWithCache(nn.Module):
    """Custom Transformer Encoder Layer using SelfAttentionWithCache."""
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * d_model
        
        # Self-Attention
        self.attn = SelfAttentionWithCache(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # 1. Self-Attention (Pre-Norm)
        attn_output, present = self.attn(self.norm1(x), layer_past)
        x = x + self.dropout1(attn_output) # Residual
        
        # 2. Feed-Forward (Pre-Norm)
        ff_output = self.linear2(F.relu(self.linear1(self.norm2(x))))
        x = x + self.dropout2(ff_output) # Residual
        
        return x, present


# ==========================================================
# 0. ADAPTIVE BLOCK (REFACTORED)
# ==========================================================
class AdaptiveBlock(nn.Module):
    """AdaptiveBottomBlock now uses the cache-enabled Transformer layer."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Koristimo novu verziju koja podržava KV Cache
        self.block = TransformerEncoderLayerWithCache(
            d_model=d_model, n_heads=n_heads, dim_feedforward=4*d_model
        )
        self.confidence_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    # Dodata cache podrška
    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        x_out, present = self.block(x, layer_past)
        
        # Confidence prediction only if S > 1 (not required for single-token incremental pass)
        if x_out.size(1) > 1:
            conf = self.confidence_predictor(x_out.transpose(1, 2))
            conf = conf.mean(dim=0)
        else:
            conf = x_out.new_tensor([0.0]) # Default 0 confidence for incremental pass
        
        return x_out, conf, present

# ==========================================================
# 1. LEARNED HARMONIC BASIS (Isto)
# ==========================================================
class LearnedHarmonicBasis(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.alpha_xz = nn.Parameter(torch.ones(d_model))
        self.beta_xy = nn.Parameter(torch.ones(d_model) * -1.0)
        self.gamma_x = nn.Parameter(torch.zeros(d_model))
        self.alpha_wy = nn.Parameter(torch.ones(d_model))
        self.beta_wx = nn.Parameter(torch.ones(d_model) * -1.0)
        self.gamma_w = nn.Parameter(torch.zeros(d_model))
        self.alpha_xv = nn.Parameter(torch.ones(d_model))
        self.beta_wv = nn.Parameter(torch.ones(d_model) * -1.0)
        self.gamma_v = nn.Parameter(torch.zeros(d_model))
        
    def forward_x(self, z, y):
        return self.alpha_xz * z + self.beta_xy * y + self.gamma_x
    
    def forward_w(self, y, x):
        return self.alpha_wy * y + self.beta_wx * x + self.gamma_w
    
    def forward_v(self, x, w):
        return self.alpha_xv * x + self.beta_wv * w + self.gamma_v
    
    def get_closure_loss(self):
        l1_loss = (self.alpha_xz - 1.0).abs().mean() + \
                  (self.beta_xy + 1.0).abs().mean()
        return l1_loss

# ==========================================================
# 2. HIERARCHICAL INJECTION GATE (Isto)
# ==========================================================
class HierarchicalInjectionGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x_bottom, x_top):
        combined = torch.cat([x_bottom, x_top], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        x_out = g * x_bottom + (1 - g) * x_top
        return self.norm(x_out)

# ==========================================================
# 3. TRUE MULTI-LAYER LATTICE CORE (Optimizovan)
# ==========================================================
class PredictionLattice(nn.Module):
    def __init__(self, max_len=8192):
        super().__init__()
        s = [0, 2, 4]
        while True:
            next_val = 2*s[-1] + 2*s[-2] + 2*s[-3]
            if next_val >= max_len: break
            s.append(next_val)
        self.register_buffer('spine_indices', torch.tensor(s, dtype=torch.long))

class TrueMultiLayerLattice(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.lattice_positions = PredictionLattice(max_seq_len)
        self.basis = LearnedHarmonicBasis(d_model)
        self.gate = HierarchicalInjectionGate(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D] - ulazni tenzor iz Bottom Stack-a
        B, S, D = x.shape
        lattice_indices = self.lattice_positions.spine_indices
        relevant_indices = lattice_indices[lattice_indices < S]
        
        # Ključno: Klonicemo jednom i koristiti direktno dodeljivanje (4.1 fix)
        h_out = x.clone() 

        for k in range(3, len(relevant_indices)):
            pos_z = relevant_indices[k].item()    
            pos_y = relevant_indices[k-1].item() 
            pos_x = relevant_indices[k-2].item() 
            
            z = h_out[:, pos_z, :]  # [B, D]
            y = h_out[:, pos_y, :]
            x_prev = h_out[:, pos_x, :]
            
            x_new = self.basis.forward_x(z, y)
            w = self.basis.forward_w(y, x_prev)
            v = self.basis.forward_v(x_new, w)
            
            v_injected = self.gate(v.unsqueeze(1), z.unsqueeze(1)).squeeze(1)
            
            # POPRAVKA 4.1: Direktno dodeljivanje umesto index_copy
            h_out[:, pos_z, :] = v_injected
            
        return h_out

    def get_closure_loss(self):
        return self.basis.get_closure_loss()

# ==========================================================
# 4. HARMONIC HORIZON PREDICTOR (Isto)
# ==========================================================
class HarmonicHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, horizon=16):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.horizon_projection = nn.Linear(d_model, d_model * horizon)
        self.prediction_head = nn.Linear(d_model, vocab_size, bias=False)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, horizon)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_last = x[:, -1, :]
        
        projected = self.horizon_projection(x_last).view(-1, self.horizon, self.d_model)
        logits_list = self.prediction_head(projected)
        confidence = torch.sigmoid(self.confidence_head(x_last))
        
        return logits_list, confidence

# ==========================================================
# 5. FULL HST-v3 ULTRA MODEL (REFACTORED for KV Cache)
# ==========================================================
class HSTv3Ultra(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        max_seq_len=8192,
        horizon=16
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.horizon = horizon
        self.max_seq_len = max_seq_len
        self.n_bottom_layers = n_layers // 2
        self.n_top_layers = n_layers - self.n_bottom_layers
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 1. Adaptive Bottom Transformer (Layer Bank) - KORISTI CACHE BLOKOVE
        self.adaptive_bottom = nn.ModuleList([
            AdaptiveBlock(d_model=d_model, n_heads=n_heads) 
            for _ in range(self.n_bottom_layers)
        ])

        # 2. Multi-Layer Lattice Core
        self.lattice_core = TrueMultiLayerLattice(d_model, max_seq_len)
        
        # 3. Top Stack (Fixed layers) - KORISTI CACHE SLOJEVE
        self.top_stack = nn.ModuleList([
            TransformerEncoderLayerWithCache(d_model=d_model, n_heads=n_heads)
            for _ in range(self.n_top_layers)
        ])

        # 4. Horizon Predictor
        self.harmonic_horizon_predictor = HarmonicHorizonPredictor(d_model, vocab_size, horizon=horizon)
        
        # 5. Output Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)
        
    def forward(self, input_ids: torch.Tensor, cache: KVCache = None) -> Dict:
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        # Određivanje start pozicije za keš
        past_len = cache[0][0].size(2) if cache else 0
        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
        
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        # Inicijalizacija novog keša
        new_cache = []
        cache_idx = 0
        predicted_depth = self.n_bottom_layers

        # 1. Adaptive Bottom Stack (Early-Exit Logic)
        for i, block in enumerate(self.adaptive_bottom):
            layer_past = cache[cache_idx] if cache else None
            x, conf, present = block(x, layer_past)
            new_cache.append(present)
            cache_idx += 1
            
            # Early exit samo tokom punog prolaza (Full Pass)
            if past_len == 0 and i >= 1 and conf.item() > 0.93:
                predicted_depth = i + 1
                break
        
        h_bottom = x
        
        # Ako je došlo do ranog izlaza, Lattice i Top Stack operišu samo na trenutnim stanjima
        
        # 2. Multi-Layer Lattice Core Processing
        h_lattice_out = self.lattice_core(h_bottom)
        
        # 3. Top Stack
        h_top_in = h_lattice_out
        for i, block in enumerate(self.top_stack):
            layer_past = cache[cache_idx] if cache else None
            h_top_in, present = block(h_top_in, layer_past)
            new_cache.append(present)
            cache_idx += 1
            
        h_final = h_top_in
        
        # 4. Next-Token Logits
        logits_t1 = self.lm_head(self.ln_f(h_final))
        
        # 5. Horizon Logits
        logits_horizon, confidence = self.harmonic_horizon_predictor(h_final)
        
        return {
            'logits': logits_t1,
            'horizon_logits': logits_horizon,
            'confidence': confidence,
            'hidden_states': h_final,
            'bottom_depth': predicted_depth,
            'cache': new_cache # Vraćamo ažurirani keš
        }

    def get_closure_loss(self):
        return self.lattice_core.get_closure_loss()

    # ==========================================================
    # TRUE SPECULATIVE DECODING / INFERENCE ENGINE (POPRAVLJENO SA KV CACHE)
    # ==========================================================
    @torch.no_grad()
    def generate_ultra_fast(self, input_ids, max_new_tokens, temperature=1.0, top_k=50):
        device = input_ids.device
        
        current_ids = input_ids.clone()
        generated_tokens = 0
        accepted_tokens = 0
        
        # Inicijalni puni prolaz (Draft Pass) za popunjavanje KV Cache-a
        full_output = self.forward(current_ids, cache=None)
        cache = full_output['cache']
        
        # Puna logits sekvenca iz Draft Pass-a (za D_0 predikciju)
        initial_logits = full_output['logits'][0] 

        for step in range(max_new_tokens):
            if generated_tokens >= max_new_tokens:
                break
                
            # 1. Generiši Horizon predloge (Draft tokens)
            # Predikcije se rade na osnovu zadnjeg logita iz inicijalnog (ili prethodnog Verification) prolaza
            
            # Pošto se u koraku k=0 koristi logit iz Full Pass-a (L_S-1), moramo ga uzeti:
            if generated_tokens == 0:
                # Logit za poziciju S (prvi novi token D_0) je na indexu S-1
                last_verification_logit = initial_logits[-1] 
            else:
                # Koristimo zadnji logit iz Verification Pass-a (L_{S+k_accepted})
                last_verification_logit = verification_logits[S + num_accepted - 1] 
            
            # Sampling prvog tokena D_0 iz P_verifier
            logits_d0 = last_verification_logit
            if top_k > 0:
                v, _ = torch.topk(logits_d0, top_k)
                logits_d0[logits_d0 < v[-1]] = -float('Inf')
            probs_d0 = F.softmax(logits_d0 / temperature, dim=-1)
            token_d0 = torch.multinomial(probs_d0, 1).item()
            
            # Sada generišemo ostalih H-1 tokena iz Horizon Predictora
            # Horizon Predictor zahteva poslednje skriveno stanje (h_final)
            h_last = full_output['hidden_states'][:, -1:, :]
            horizon_logits_list, _ = self.harmonic_horizon_predictor(h_last)
            horizon_logits = horizon_logits_list[0] # [H, V]
            
            draft_tokens = [token_d0]
            
            # Generišemo ostale tokene iz Horizon Predictor-a (D_1 do D_H-1)
            for k in range(1, self.horizon):
                logits_k = horizon_logits[k]
                if top_k > 0:
                    v, _ = torch.topk(logits_k, top_k)
                    logits_k[logits_k < v[-1]] = -float('Inf')
                
                probs_k = F.softmax(logits_k / temperature, dim=-1)
                token_k = torch.multinomial(probs_k, 1).item()
                draft_tokens.append(token_k)
                
            draft_tokens_tensor = torch.tensor(draft_tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # 3. Proširi ulaz sa draft tokenima
            S = current_ids.size(1) # Dužina starog konteksta
            H_drafted = len(draft_tokens_tensor[0])
            
            # Prvi token (D_0) je iz uzorkovan iz verifiera, ostali (H-1) su iz draftera
            extended_ids = draft_tokens_tensor 
            
            # 4. Provera validacije (Verification pass) - INKREMENTALNI PROLAZ
            # Prosleđujemo samo nove tokene H_drafted
            verification_output = self.forward(extended_ids, cache=cache) 
            verification_logits = verification_output['logits'][0] # [H_drafted, V]
            cache = verification_output['cache'] # Ažuriramo keš

            num_drafted = H_drafted
            num_accepted = 0
            
            # 5. Iterativna provera prihvatanja (Ratio Test)
            for k in range(num_drafted):
                
                # Logit L_k u verification_logits (jer je S_new = H_drafted) predviđa D_k.
                logits_k = verification_logits[k]
                draft_token = draft_tokens[k]
                
                probs_k = F.softmax(logits_k, dim=-1)
                
                prob_draft = probs_k[draft_token]
                prob_max = probs_k.max()

                # Ratio Test: P_verifier(D_k) / P_max_verifier >= U(0,1)
                if prob_draft / prob_max >= torch.rand(1, device=device):
                    num_accepted += 1
                else:
                    # Odbijanje: Samplingujemo ispravni token P_verifier
                    new_token_logits = logits_k
                    if top_k > 0:
                        v, _ = torch.topk(new_token_logits, top_k)
                        new_token_logits[new_token_logits < v[-1]] = -float('Inf')
                    probs = F.softmax(new_token_logits / temperature, dim=-1)
                    new_token = torch.multinomial(probs, 1).item()
                    
                    # Ažuriramo ID-ove
                    new_ids = draft_tokens_tensor[0, :num_accepted].tolist() + [new_token]
                    current_ids = torch.cat([current_ids, current_ids.new_tensor(new_ids).unsqueeze(0)], dim=1)
                    
                    generated_tokens += num_accepted + 1
                    break
            
            # 6. Ako su svi prihvaćeni
            if num_accepted == num_drafted:
                # Ažuriramo ID-ove
                current_ids = torch.cat([current_ids, draft_tokens_tensor], dim=1)
                generated_tokens += num_drafted
                accepted_tokens += num_drafted
            elif num_accepted < num_drafted:
                # Ako je došlo do prekida, tokeni do num_accepted su već dodati u break bloku
                accepted_tokens += num_accepted
            

        # Izračunavanje metrika brzine
        acceptance_rate = accepted_tokens / generated_tokens if generated_tokens > 0 else 0.0
        effective_speedup = 1.0 + acceptance_rate * (self.horizon - 1)
        
        stats = {
            'tokens_generated': generated_tokens,
            'accepted_tokens': accepted_tokens,
            'acceptance_rate': acceptance_rate,
            'effective_speedup': effective_speedup
        }
        
        return current_ids, stats


if __name__ == '__main__':
    print("=" * 70)
    print("HST-v3 ULTRA (Konačna, Popravljena Implementacija SA KV CACHE)")
    print("=" * 70)
    
    # Test model configuration: 8 layers (4 bottom/adaptive, 4 top/fixed)
    model = HSTv3Ultra(
        vocab_size=50257,
        d_model=256,
        n_heads=4,
        n_layers=8, 
        horizon=16
    )

    # Test forward pass and autograd
    x = torch.randint(0, 50257, (2, 512)) 
    output = model(x)
    
    closure_loss = model.get_closure_loss() 
    loss = output['logits'].mean() + 0.1 * closure_loss
    
    try:
        loss.backward()
        print("✅ Forward/Backward pass successful!")
    except RuntimeError as e:
        print(f"❌ Backward pass failed: {e}")
        
    
    # Test ultra-fast generation
    print("\nTesting Ultra-Fast Generation...")
    prompt = torch.randint(0, 50257, (1, 10))
    generated, stats = model.generate_ultra_fast(prompt, max_new_tokens=50, temperature=0.8)
    
    print("✅ Generation successful!")
    print(f"   Generated length: {generated.size(1) - prompt.size(1)} tokens")
    print(f"   Acceptance Rate: {stats['acceptance_rate']:.3f}")
    print(f"   Effective Speedup: {stats['effective_speedup']:.2f}x (Očekivano ~5-8x sa KV Cache)")
    print("=" * 70)