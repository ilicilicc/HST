import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from torch.distributions import Categorical, Gumbel

# ==============================================================================
# The following code is copied from hst.v7.0_agile.py to provide the necessary
# building blocks for the ChaosLogicAI model.
# ==============================================================================

# KV Cache with Compression
class CompressedCache(nn.Module):
    def __init__(self, d_model=2048, sparsity=0.1):
        super().__init__()
        self.compress = nn.Linear(d_model * 2, int(d_model * sparsity))
        self.decompress = nn.Linear(int(d_model * sparsity), d_model)
        self.sparse_attn = nn.MultiheadAttention(d_model, 16, batch_first=True)

    def update_cache(self, kv, prev_cache=None):
        if prev_cache is None:
            cache = self.compress(torch.cat([kv[0], kv[1]], -1))
        else:
            cache = self.compress(torch.cat([kv[0], kv[1], prev_cache], -1))
        return self.decompress(cache)

    def forward(self, x, cache):
        q = x
        k, v = cache.chunk(2, -1)
        attn_out, new_kv = self.sparse_attn(q, k.unsqueeze(0), v.unsqueeze(0), need_weights=False)
        updated_cache = self.update_cache((new_kv[0].squeeze(0), new_kv[1].squeeze(0)), cache)
        return attn_out.squeeze(0), updated_cache

# Speculative Decoding with Verification
class SpeculativeVerifier(nn.Module):
    def __init__(self, d_model=4096, n_layers=32, horizon=64, vocab_size=50257, n_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, vocab_size * horizon)
        self.horizon = horizon
        self.conf_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, draft, cache_kv):
        x = self.embed(draft)
        for layer in self.layers:
            x = layer(x, memory=cache_kv)
        logits = self.proj(x.mean(1)).view(-1, self.horizon, self.vocab_size)
        conf = self.conf_gate(x.mean(1))
        return logits * conf.unsqueeze(-1), conf.mean()

# ===== Adaptive Transformer Core Helper Modules =====
class TaskAnalyzer(nn.Module):
    def __init__(self, d_model=512, num_tasks=4):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_tasks)

    def forward(self, x):
        h = torch.mean(self.embed(x), dim=1)
        logits = self.classifier(h)
        probs = F.softmax(logits, dim=-1)
        return probs

class DepthPredictor(nn.Module):
    def __init__(self, num_tasks=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_tasks, num_tasks * 2),
            nn.ReLU(),
            nn.Linear(num_tasks * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, task_probs):
        depth = 4 + 12 * self.net(task_probs)
        return depth.clamp(4, 16)

class PatternSelector(nn.Module):
    def __init__(self, num_patterns=4):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_patterns))

    def forward(self, task_probs, num_layers):
        gumbel = Gumbel(0, 1).rsample((num_layers, self.logits.numel()))
        logits = self.logits.unsqueeze(0) + gumbel.to(self.logits.device)
        patterns = F.softmax(logits, dim=-1).argmax(-1)
        return patterns

# This is a new module that replaces the `adaptive_bottom` loop.
class AdaptiveBottomTransformer(nn.Module):
    def __init__(self, d_model=512, num_layers_max=16, n_heads=8):
        super().__init__()
        self.num_layers_max = num_layers_max
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithCache(d_model, n_heads) for _ in range(num_layers_max)
        ])
        self.task_analyzer = TaskAnalyzer(d_model)
        self.depth_pred = DepthPredictor(num_tasks=4)

    def forward(self, x, cache=None):
        past_len = cache[0][0].size(2) if cache and cache[0] and cache[0][0] is not None else 0
        
        if past_len == 0:
            with torch.no_grad():
                task_probs = self.task_analyzer(x)
                depth_tensor = self.depth_pred(task_probs)
                predicted_depth = int(depth_tensor.mean().round().item())
                predicted_depth = max(4, min(predicted_depth, self.num_layers_max))
        else:
            predicted_depth = self.num_layers_max

        new_cache = []
        out = x
        for i in range(predicted_depth):
            layer_past = cache[i] if cache and i < len(cache) else None
            out, present = self.layers[i](out, layer_past=layer_past)
            new_cache.append(present)
        
        for i in range(predicted_depth, self.num_layers_max):
            new_cache.append((None,None))

        return out, predicted_depth, new_cache

KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

def prune_cache(cache: KVCache, max_size: int = 2048) -> KVCache:
    if not cache or not cache[0] or cache[0][0] is None or cache[0][0].size(2) <= max_size:
        return cache
    
    pruned_cache = []
    for k, v in cache:
        if k is not None and v is not None:
            pruned_k = k[:, :, -max_size:, :]
            pruned_v = v[:, :, -max_size:, :]
            pruned_cache.append((pruned_k, pruned_v))
        else:
            pruned_cache.append((None, None))
    
    return pruned_cache

class ChunkEncoder(nn.Module):
    def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True
        )
        self.local_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pooling_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, token_embeddings):
        B, total_tokens, D = token_embeddings.shape
        num_chunks = total_tokens // self.chunk_size
        
        chunks = token_embeddings[:, :num_chunks * self.chunk_size, :].view(
            B * num_chunks, self.chunk_size, D
        )
        
        encoded_tokens = self.local_encoder(chunks)
        
        query = self.pooling_query.expand(B * num_chunks, -1, -1)
        pooled, _ = self.pooling_attn(query, encoded_tokens, encoded_tokens)
        
        chunk_embeddings = pooled.view(B, num_chunks, D)
        
        return chunk_embeddings


class ChunkDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.pos_embedding = nn.Embedding(chunk_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True
        )
        self.local_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, chunk_embeddings, target_token_embeddings):
        B, num_chunks, D = chunk_embeddings.shape
        seq_len = num_chunks * self.chunk_size

        pos = torch.arange(0, self.chunk_size, device=target_token_embeddings.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos).repeat(B * num_chunks, 1, 1)
        
        tgt = target_token_embeddings.view(B * num_chunks, self.chunk_size, D) + pos_emb
        
        memory = chunk_embeddings.view(B * num_chunks, 1, D).repeat(1, self.chunk_size, 1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.chunk_size).to(tgt.device)

        refined = self.local_decoder(tgt, memory, tgt_mask=causal_mask)

        refined = refined.view(B, seq_len, D)

        logits = self.lm_head(refined)
        return logits


class SelfAttentionWithCache(nn.Module):
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
        
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        present = (k, v)
        
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        full_S = k.size(2)
        if S > 1:
            attn_mask = torch.triu(torch.ones(S, full_S, dtype=torch.bool, device=x.device), diagonal=full_S - S + 1)
            attn_weights.masked_fill_(attn_mask[None, None, :, :], -torch.inf)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S, D)
        
        output = self.out_proj(attn_output)
        return output, present

class TransformerDecoderLayerWithCache(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.self_attn = SelfAttentionWithCache(d_model, n_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory, self_attn_past=None, cross_attn_past=None):
        tgt_norm = self.norm1(tgt)
        sa_output, sa_present = self.self_attn(tgt_norm, layer_past=self_attn_past)
        tgt = tgt + self.dropout1(sa_output)

        tgt_norm = self.norm2(tgt)
        
        if cross_attn_past is not None:
            ca_output, _ = self.cross_attn(tgt_norm, cross_attn_past[0], cross_attn_past[1])
            ca_present = cross_attn_past
        else:
            ca_output, _ = self.cross_attn(tgt_norm, memory, memory)
            ca_present = (memory, memory) 

        tgt = tgt + self.dropout2(ca_output)

        tgt_norm = self.norm3(tgt)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout(ff_output)
        
        return tgt, sa_present, ca_present

class TransformerEncoderLayerWithCache(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * d_model
        
        self.attn = SelfAttentionWithCache(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        attn_output, present = self.attn(self.norm1(x), layer_past)
        x = x + self.dropout1(attn_output)
        
        ff_output = self.linear2(F.relu(self.linear1(self.norm2(x))))
        x = x + self.dropout2(ff_output)
        
        return x, present

class AdaptiveBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.block = TransformerEncoderLayerWithCache(
            d_model=d_model, n_heads=n_heads, dim_feedforward=4*d_model
        )
        self.confidence_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        x_out, present = self.block(x, layer_past)
        
        if x_out.size(1) > 1:
            conf = self.confidence_predictor(x_out.transpose(1, 2))
            conf = conf.mean(dim=0)
        else:
            conf = x_out.new_tensor([0.0])
        
        return x_out, conf, present

class ChunkDecoderWithCache(nn.Module):
    def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.pos_embedding = nn.Embedding(chunk_size, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayerWithCache(d_model, n_heads) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, chunk_embeddings, target_token_embeddings, cache=None):
        B, S, D = target_token_embeddings.shape
        device = target_token_embeddings.device
        
        past_len = cache[0][0][0].size(2) if cache else 0
        positions = torch.arange(past_len, past_len + S, dtype=torch.long, device=device) % self.chunk_size
        
        pos_emb = self.pos_embedding(positions)
        tgt = target_token_embeddings + pos_emb
        
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else (None, None)
            self_attn_past, cross_attn_past = layer_cache

            memory = chunk_embeddings.repeat(1, S, 1)

            tgt, sa_present, ca_present = layer(tgt, memory, self_attn_past, cross_attn_past)
            new_cache.append((sa_present, ca_present))
            
        logits = self.lm_head(tgt)
        return logits, new_cache

class RecursiveDescentLatticeAnalyzer(nn.Module):
    def __init__(self, max_seq_len=8192):
        super().__init__()
        spine_list = self._generate_spine_list(max_seq_len)
        self.register_buffer('spine', torch.tensor(spine_list, dtype=torch.long))
        self.descent_paths = self._compute_descent_paths()
        self.layer_weights = nn.Parameter(torch.ones(10))

    def _generate_spine_list(self, max_len):
        spine = [0, 2, 4]
        while True:
            next_val = 2 * spine[-1] + 2 * spine[-2] + 2 * spine[-3]
            if next_val >= max_len:
                break
            spine.append(next_val)
        return spine

    def _find_parent(self, pos):
        if pos in self.spine:
            idx = (self.spine == pos).nonzero(as_tuple=True)[0].item()
            if idx > 0:
                return self.spine[idx-1].item()
        left_spine = self.spine[self.spine < pos]
        if len(left_spine) > 0:
            return left_spine[-1].item()
        return 0


    def _compute_descent_paths(self):
        paths = {}
        for pos_tensor in self.spine:
            pos = pos_tensor.item()
            path = []
            current = pos
            layer = 0
            while current > 0 and layer < 10:
                parent = self._find_parent(current)
                path.append((layer, parent))
                if current == parent:
                    break
                current = parent
                layer += 1
            paths[pos] = path
        return paths

    def compute_predictive_field(self, pos, target_offset):
        try:
            source_spine_idx = (self.spine == pos).nonzero(as_tuple=True)[0]
            target_spine_idx = (self.spine == (pos + target_offset)).nonzero(as_tuple=True)[0]
            spine_distance = abs(target_spine_idx - source_spine_idx)
        except (IndexError, RuntimeError):
            spine_distance = int(np.log2(target_offset + 1))


        layer_importance = torch.zeros(10, device=self.layer_weights.device)
        if spine_distance > 5:
            layer_importance[0:3] = torch.tensor([1.0, 0.8, 0.5])
        elif spine_distance > 2:
            layer_importance[1:5] = torch.tensor([0.5, 1.0, 0.8, 0.3])
        else:
            layer_importance[3:7] = torch.tensor([0.3, 0.8, 1.0, 0.8])
        
        layer_importance = layer_importance.to(self.layer_weights.device)
        
        layer_importance = layer_importance * torch.sigmoid(self.layer_weights)
        return layer_importance

class AdaptiveLatticeProcessor(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.analyzer = RecursiveDescentLatticeAnalyzer(max_seq_len)
        self.layer_processors = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
            for _ in range(10)
        ])
        self.task_router = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, horizon_targets=None) -> torch.Tensor:
        B, S, D = x.shape
        task_embedding = x.mean(dim=1)
        layer_gates = self.task_router(task_embedding)

        h = x
        for layer_idx, processor in enumerate(self.layer_processors):
            gate = layer_gates[:, layer_idx].unsqueeze(1).unsqueeze(2)
            if gate.mean() > 0.1:
                h_layer = processor(h)
                h = h + gate * (h_layer - h)
        return h

class RecursiveHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, horizon=16):
        super().__init__()
        self.horizon = horizon
        self.coarse_predictor = nn.Linear(d_model, vocab_size)
        self.medium_predictor = nn.Linear(d_model + d_model, vocab_size)
        self.fine_predictor = nn.Linear(d_model + d_model, vocab_size)
        self.lattice_embeddings = nn.Embedding(20, d_model)
        self.projection = nn.Linear(vocab_size, d_model)

    def forward(self, h_sequence):
        B, S, D = h_sequence.shape
        h_t = h_sequence[:, -1, :]
        
        coarse_offsets = [4, 10]
        coarse_preds = {}
        for offset in coarse_offsets:
            offset_emb = self.lattice_embeddings(torch.tensor([offset - 1], device=h_t.device))
            h_augmented = h_t + offset_emb
            pred = self.coarse_predictor(h_augmented)
            coarse_preds[offset] = pred

        medium_offsets = [2, 6]
        medium_preds = {}
        for offset in medium_offsets:
            left_coarse = coarse_preds[4]
            right_coarse = coarse_preds[10]
            alpha = (offset - 4) / (10 - 4)
            coarse_interp = self.projection(alpha * left_coarse + (1 - alpha) * right_coarse)
            h_interpolated = torch.cat([h_t, coarse_interp], dim=-1)
            pred = self.medium_predictor(h_interpolated)
            medium_preds[offset] = pred

        fine_offsets = [1, 3, 5]
        fine_preds = {}
        for offset in fine_offsets:
            left_med = medium_preds[2]
            right_med = medium_preds[6]
            alpha = (offset - 2) / (6 - 2)
            medium_interp = self.projection(alpha * left_med + (1-alpha) * right_med)
            h_interpolated = torch.cat([h_t, medium_interp], dim=-1)
            pred = self.fine_predictor(h_interpolated)
            fine_preds[offset] = pred
            
        all_preds = {**coarse_preds, **medium_preds, **fine_preds}
        
        logits_list = [all_preds.get(i, torch.zeros(B, self.coarse_predictor.out_features, device=h_t.device)) for i in range(1, self.horizon + 1)]
        logits = torch.stack(logits_list, dim=1)
        
        confidence = torch.ones(B, self.horizon, device=h_t.device)
        
        return logits, confidence

class HSTv7Agile(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        max_seq_len=8192,
        horizon=16,
        early_exit_confidence_threshold=0.93,
        mode='token',
        chunk_size=128
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.horizon = horizon
        self.max_seq_len = max_seq_len
        self.n_bottom_layers = n_layers // 2
        self.n_top_layers = n_layers - self.n_bottom_layers
        self.early_exit_confidence_threshold = early_exit_confidence_threshold
        self.mode = mode
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        if self.mode == 'chunk':
            self.pos_embedding = nn.Embedding(max_seq_len * chunk_size, d_model)
            self.chunk_encoder = ChunkEncoder(d_model, chunk_size)
            self.chunk_decoder = ChunkDecoderWithCache(d_model, vocab_size, chunk_size)
            self.lattice_core = AdaptiveLatticeProcessor(d_model, max_seq_len)
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.adaptive_bottom = AdaptiveBottomTransformer(
                d_model=d_model, n_heads=n_heads, num_layers_max=self.n_bottom_layers
            )
            self.lattice_core = AdaptiveLatticeProcessor(d_model, max_seq_len)
            self.top_stack = nn.ModuleList([
                TransformerEncoderLayerWithCache(d_model=d_model, n_heads=n_heads)
                for _ in range(self.n_top_layers)
            ])

        self.horizon_predictor = RecursiveHorizonPredictor(d_model, vocab_size, horizon=horizon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)
        self.speculative_verifier = SpeculativeVerifier(d_model=d_model, n_layers=self.n_top_layers, horizon=horizon, vocab_size=vocab_size, n_heads=n_heads)

    def forward(self, input_ids: torch.Tensor, cache: KVCache = None, horizon_targets=None, injected_context: Optional[Dict[int, torch.Tensor]] = None) -> Dict:
        if self.mode == 'token':
            return self.forward_token(input_ids, cache)
        elif self.mode == 'chunk':
            return self.forward_chunk(input_ids, horizon_targets, injected_context)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward_token(self, input_ids: torch.Tensor, cache: KVCache = None) -> Dict:
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        past_len = 0
        if cache and cache[0] and cache[0][0] is not None:
             past_len = cache[0][0].size(2)

        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
        
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        bottom_cache = cache[:self.n_bottom_layers] if cache else None
        h_bottom, predicted_depth, bottom_new_cache = self.adaptive_bottom(x, cache=bottom_cache)

        h_lattice_out = self.lattice_core(h_bottom)
        
        h_top_in = h_lattice_out
        new_cache = bottom_new_cache
        top_stack_cache = cache[self.n_bottom_layers:] if cache else None
        
        for i, block in enumerate(self.top_stack):
            layer_past = top_stack_cache[i] if top_stack_cache and i < len(top_stack_cache) else None
            h_top_in, present = block(h_top_in, layer_past)
            new_cache.append(present)
            
        h_final = h_top_in
        logits_t1 = self.lm_head(self.ln_f(h_final))
        logits_horizon, confidence = self.horizon_predictor(h_final)
        
        return {
            'logits': logits_t1,
            'horizon_logits': logits_horizon,
            'confidence': confidence,
            'hidden_states': h_final,
            'bottom_depth': predicted_depth,
            'cache': new_cache
        }

    def forward_chunk(self, input_ids: torch.Tensor, horizon_targets=None, injected_context: Optional[Dict[int, torch.Tensor]] = None) -> Dict:
        B, total_tokens = input_ids.shape
        device = input_ids.device

        target_ids = torch.roll(input_ids, shifts=-1, dims=1)
        target_ids[:, -1] = 0

        positions = torch.arange(0, total_tokens, dtype=torch.long, device=device)
        input_token_emb = self.token_embedding(input_ids) + self.pos_embedding(positions)
        target_token_emb = self.token_embedding(target_ids) + self.pos_embedding(positions)
        
        chunk_emb = self.chunk_encoder(input_token_emb)

        if injected_context:
            for spine_pos, context_vector in injected_context.items():
                if spine_pos < chunk_emb.size(1):
                    if B > 1 and context_vector.size(0) == 1:
                        context_vector = context_vector.expand(B, -1)
                    chunk_emb[:, spine_pos, :] = context_vector

        h_lattice_out = self.lattice_core(chunk_emb)
        
        cache = injected_context.get('decoder_cache', None) if injected_context else None
        
        logits, new_cache = self.chunk_decoder(h_lattice_out, target_token_emb, cache=cache)

        last_chunk_rep = h_lattice_out[:, -1:, :]
        logits_horizon, confidence = self.horizon_predictor(last_chunk_rep)
        
        return {
            'logits': logits,
            'horizon_logits': logits_horizon,
            'confidence': confidence,
            'hidden_states': h_lattice_out,
            'bottom_depth': 0,
            'cache': new_cache
        }

    @torch.no_grad()
    def generate_speculative(self, input_ids, max_new_tokens, temperature=1.0, top_k=50, max_cache_size=2048):
        device = input_ids.device
        
        current_ids = input_ids.clone()
        
        full_output = self.forward_token(current_ids, cache=None)
        cache = full_output['cache']
        hidden_states = full_output['hidden_states']

        for _ in range(max_new_tokens):
            draft_tokens = []
            draft_input_ids = current_ids[:,-1:]
            draft_cache = cache
            
            for _ in range(self.horizon):
                outputs = self.forward_token(draft_input_ids, cache=draft_cache)
                next_token_logits = outputs['logits'][:, -1, :]
                
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, -1].unsqueeze(-1)] = -float('Inf')
                
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                draft_tokens.append(next_token.item())
                draft_input_ids = next_token
                draft_cache = outputs['cache']

            draft_tensor = torch.tensor(draft_tokens, dtype=torch.long, device=device).unsqueeze(0)

            verified_logits, confidence = self.speculative_verifier(draft_tensor, hidden_states)

            num_accepted = 0
            for i in range(self.horizon):
                draft_token = draft_tensor[:, i]
                verified_token_probs = F.softmax(verified_logits[:, i, :], dim=-1)
                
                _, top_indices = torch.topk(verified_token_probs, top_k)
                if draft_token in top_indices:
                    current_ids = torch.cat([current_ids, draft_token.unsqueeze(0)], dim=1)
                    num_accepted += 1
                else:
                    new_token = torch.multinomial(verified_token_probs, num_samples=1)
                    current_ids = torch.cat([current_ids, new_token], dim=1)
                    break 

            if num_accepted > 0:
                accepted_ids = current_ids[:, -num_accepted:]
                outputs = self.forward_token(accepted_ids, cache=cache)
                cache = outputs['cache']
                hidden_states = torch.cat([hidden_states, outputs['hidden_states']], dim=1)

            if num_accepted < self.horizon:
                outputs = self.forward_token(current_ids[:,-1:], cache=cache)
                cache = outputs['cache']
                hidden_states = torch.cat([hidden_states, outputs['hidden_states']], dim=1)


        return current_ids

# ==============================================================================
# New Chaos Logic AI Implementation
# ==============================================================================

class ChaosLogicAI(nn.Module):
    """
    An AI model based on the principles of Chaos Logic, simulating the dynamic
    balance between Chaos (ξ), Void (0), and Creation (ξ'). This model wraps
    the HSTv7Agile architecture and introduces a rhythmic, iterative forward
    pass.

    The core idea is to create a feedback loop where the state of the universe
    (represented by token embeddings) is continuously processed through a
    cycle of:
    1.  Existence (1): The current state.
    2.  Chaos (ξ): Introduction of randomness.
    3.  Creation (ξ'): Processing by the underlying transformer model.
    4.  Void (0): Partial reset of the state.

    This cycle, `[... 1 ← 2 → 0 ← 2 → 1 ...]`, is repeated for a fixed
    number of iterations, allowing the model to explore a dynamic and
    chaotic state space before producing an output.
    """
    def __init__(
        self,
        hst_model: HSTv7Agile,
        chaos_intensity: float = 0.1,
        void_rate: float = 0.1,
        rhythm_iterations: int = 3
    ):
        super().__init__()
        self.hst_model = hst_model
        self.chaos_intensity = chaos_intensity
        self.void_rate = void_rate
        self.rhythm_iterations = rhythm_iterations
        self.void_dropout = nn.Dropout(self.void_rate)
        self.void_to_existence = nn.Linear(self.hst_model.d_model, self.hst_model.d_model)

    def set_params(self, chaos_intensity: float, void_rate: float, rhythm_iterations: int):
        """
        Dynamically sets the core parameters of the Chaos Logic AI.
        """
        self.chaos_intensity = chaos_intensity
        self.void_rate = void_rate
        self.rhythm_iterations = rhythm_iterations
        self.void_dropout.p = self.void_rate

    def forward(self, input_ids: torch.Tensor) -> Dict:
        """
        Performs the rhythmic forward pass of the Chaos Logic AI.
        """
        B, total_tokens = input_ids.shape
        device = input_ids.device

        # Initial state of "Existence"
        existence = self.hst_model.token_embedding(input_ids)
        positions = torch.arange(0, total_tokens, dtype=torch.long, device=device)
        existence += self.hst_model.pos_embedding(positions)

        # The rhythmic cycle of Chaos, Creation, and Void
        for _ in range(self.rhythm_iterations):
            # 1. Inject Chaos (ξ)
            chaos = torch.randn_like(existence) * self.chaos_intensity
            chaotic_existence = existence + chaos

            # 2. Creation (ξ')
            # We use the core components of the HST model to represent Creation.
            # This is a simplified forward pass that focuses on the transformation
            # of the chaotic state into a new, more structured state.
            created_chunks = self.hst_model.chunk_encoder(chaotic_existence)
            creation = self.hst_model.lattice_core(created_chunks)

            # 3. Transition to Void (0)
            # We apply a dropout-like mechanism to partially reset the state,
            # pulling it towards the void.
            void = self.void_dropout(creation)

            # 4. Feedback loop: The new "Existence" is the result of the
            #    previous cycle's "Void" state.
            # We need to bring the state back to the token embedding space.
            # For simplicity, we'll use a linear projection.
            
            # The void state has shape (B, num_chunks, D), but we need to get back to (B, total_tokens, D)
            # We will upsample the chunk embeddings to token embeddings.
            num_chunks = void.shape[1]
            upsampled_void = void.repeat_interleave(self.hst_model.chunk_size, dim=1)
            
            existence = self.void_to_existence(upsampled_void)


        # Final pass through the decoder to get logits
        # After the rhythmic iterations, the final "existence" state is used to
        # generate the output.
        target_ids = torch.roll(input_ids, shifts=-1, dims=1)
        target_ids[:, -1] = 0
        target_token_emb = self.hst_model.token_embedding(target_ids) + self.hst_model.pos_embedding(positions)
        
        # The 'memory' for the decoder is the final state of 'creation'
        # from the rhythmic loop.
        final_memory = creation
        
        logits, _ = self.hst_model.chunk_decoder(final_memory, target_token_emb)

        return {
            'logits': logits
        }


if __name__ == '__main__':
    print("=" * 70)
    print("Chaos Logic AI - Self-Test")
    print("=" * 70)

    vocab_size = 50257
    d_model = 256
    n_heads = 4
    n_layers = 8
    chunk_size = 128
    seq_len = 512

    # 1. Initialize the underlying HSTv7Agile model in chunk mode
    try:
        hst_model = HSTv7Agile(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mode='chunk',
            chunk_size=chunk_size
        )
        print("✅ HSTv7Agile model initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize HSTv7Agile model: {e}")
        exit()

    # 2. Initialize the ChaosLogicAI model
    try:
        chaos_model = ChaosLogicAI(
            hst_model=hst_model,
            chaos_intensity=0.05,
            void_rate=0.05,
            rhythm_iterations=2
        )
        print("✅ ChaosLogicAI model initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize ChaosLogicAI model: {e}")
        exit()

    # 3. Create a dummy input tensor
    x = torch.randint(0, vocab_size, (1, seq_len))

    # 4. Perform a forward and backward pass
    print("\n--- Testing Forward and Backward Pass ---")
    try:
        output = chaos_model(x)
        logits = output['logits']
        print(f"✅ Forward pass successful! Output logits shape: {logits.shape}")
        
        loss = logits.mean()
        loss.backward()
        print("✅ Backward pass successful!")
    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")

    print("\n" + "=" * 70)
    print("Chaos Logic AI Self-Test Complete")
    print("=" * 70)
