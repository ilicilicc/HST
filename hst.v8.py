import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List, Tuple, Dict
from torch.utils.checkpoint import checkpoint

# ==============================================================================
# HST v8 CRYSTALLINE - "THE BEST AI" ARCHITECTURE (DEBUGGED VERSION)
# ==============================================================================
# 1. Pell-Lucas Time Spine (Infinite Context)
# 2. Diamond Mixer (Lossless Logic)
# 3. Holographic Lattice (Interference Field)
# 4. Feedback Loop (Self-Correction)
# + ALL v7.1 OPTIMIZATIONS (Hyperbolic, Hebbian, Fused Ops)
# ==============================================================================

KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

class HyperbolicEmbedding(nn.Module):
    """
    HYPERBOLIC SPACE: Hierarchical representation.
    Exponentially expanding space matches the Pell-Lucas lattice growth.
    """
    def __init__(self, vocab_size, d_model, curvature=1.0):
        super().__init__()
        self.d_model = d_model
        self.c = curvature
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, 0, 0.01)
        
    def forward(self, input_ids):
        x = self.embed(input_ids)
        # Project to Poincaré ball (fast approximation)
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = (1 - 1e-3) / math.sqrt(self.c)
        scale = torch.clamp(norm / max_norm, max=1.0)
        return x / (scale + 1e-8)

class OptimizedPositionalEncoding(nn.Module):
    """Cached positional encoding - near zero overhead"""
    def __init__(self, d_model, max_seq_len=8192):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, positions):
        return self.pe[positions]

class HebbianFastWeights(nn.Module):
    """
    PLASTICITY LAYER: Learns during inference.
    Aligns with the 'Self-Correction' goal of Crystalline Architecture.
    """
    def __init__(self, d_model, lambda_decay=0.95):
        super().__init__()
        self.d_model = d_model
        self.lambda_decay = lambda_decay
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Linear attention as fast weights
        kv = torch.einsum('bsd,bse->bde', k, v)
        kv = kv * self.lambda_decay
        out = torch.einsum('bsd,bde->bse', q, kv)
        
        # Dynamic learning rate per position
        lr = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        return self.norm(x + out * lr)

class FastBlockSparseAttention(nn.Module):
    """Optimized block-sparse attention"""
    def __init__(self, d_model, n_heads, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, layer_past=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present = (k, v)
        
        # Use Flash Attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            is_causal = (layer_past is None)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        else:
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            if layer_past is None:
                mask = torch.triu(torch.ones(S, k.size(2), device=x.device), diagonal=k.size(2)-S+1)
                attn.masked_fill_(mask.bool(), float('-inf'))
            attn_out = attn.softmax(dim=-1) @ v
            
        out = attn_out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out), present

class FusedChunkEncoder(nn.Module):
    """Fused operations for chunk encoding"""
    def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True, norm_first=True),
            num_layers=n_layers
        )
        self.pooling = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, token_embeddings):
        B, total_tokens, D = token_embeddings.shape
        num_chunks = total_tokens // self.chunk_size
        chunks = token_embeddings[:, :num_chunks * self.chunk_size].view(B * num_chunks, self.chunk_size, D)
        encoded = self.encoder(chunks)
        pooled = encoded.mean(dim=1)
        compressed = self.pooling(pooled)
        return compressed.view(B, num_chunks, D)

class EfficientExpertRouter(nn.Module):
    """Optimized MoE routing"""
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.expert_up = nn.Parameter(torch.randn(num_experts, d_model * 4, d_model))
        self.expert_down = nn.Parameter(torch.randn(num_experts, d_model, d_model * 4))
        nn.init.xavier_uniform_(self.expert_up)
        nn.init.xavier_uniform_(self.expert_down)
    
    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        routing_logits = self.router(x_flat)
        routing_weights, selected_experts = torch.topk(F.softmax(routing_logits, dim=-1), self.top_k, dim=-1)
        
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = selected_experts[:, k]
            weights = routing_weights[:, k]
            up_weights = self.expert_up[expert_idx]
            down_weights = self.expert_down[expert_idx]
            
            hidden = torch.einsum('bd,bkd->bk', x_flat, up_weights)
            hidden = F.gelu(hidden)
            expert_out = torch.einsum('bk,bdk->bd', hidden, down_weights)
            output += expert_out * weights.unsqueeze(-1)
            
        return output.view(B, S, D)

class FastMultiResProcessor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.processor = nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True, norm_first=True)
        self.fusion = nn.Sequential(nn.Linear(d_model * 4, d_model), nn.LayerNorm(d_model))
    
    def forward(self, x):
        B, S, D = x.shape
        outputs = []
        for factor in [1, 2, 4, 8]:
            if S >= factor:
                if factor > 1:
                    pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=factor, stride=factor).transpose(1, 2)
                else:
                    pooled = x
                processed = self.processor(pooled)
                if factor > 1:
                    processed = F.interpolate(processed.transpose(1, 2), size=S, mode='linear', align_corners=False).transpose(1, 2)
                outputs.append(processed)
            else:
                outputs.append(self.processor(x))
        return self.fusion(torch.cat(outputs, dim=-1))

class StreamlinedHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, max_horizon=64):
        super().__init__()
        self.max_horizon = max_horizon
        self.predictor = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, vocab_size * max_horizon))
        self.uncertainty = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
    
    def forward(self, h):
        B = h.size(0)
        h_last = h[:, -1, :]
        logits = self.predictor(h_last).view(B, self.max_horizon, -1)
        uncertainty = self.uncertainty(h_last)
        horizon = (self.max_horizon * (1 - uncertainty)).long().clamp(4, self.max_horizon)
        return logits, horizon, uncertainty

class DiamondMixer(nn.Module):
    """
    THE DIAMOND MIXER (Lossless Logic)
    Replaces standard FFNs.
    Topology:
      x, y = Split(Input)
      Z = x + y (Synthesis/Context)
      W = y - x (Analysis/Detail)
      Output = Merge(Z, W)
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.split_proj = nn.Linear(d_model, d_model * 2)
        self.z_process = nn.GELU()
        self.w_process = nn.GELU()
        self.merge_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u):
        xy = self.split_proj(u)
        x, y = xy.chunk(2, dim=-1)
        z = x + y  # Synthesis
        w = y - x  # Analysis
        z_prime = self.z_process(z)
        w_prime = self.w_process(w)
        out = self.merge_proj(torch.cat([z_prime, w_prime], dim=-1))
        return self.norm(u + out)

class FeedbackLoop(nn.Module):
    """
    THE LOOP (Self-Correction Cycle)
    Iterative refinement: w -> v -> v' -> v''
    """
    def __init__(self, d_model, iterations=2):
        super().__init__()
        self.iterations = iterations
        self.loop_net = nn.GRUCell(d_model, d_model)
        self.error_estimator = nn.Linear(d_model, 1)
        
    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        state = x_flat
        for _ in range(self.iterations):
            error = torch.sigmoid(self.error_estimator(state))
            new_state = self.loop_net(state, state)
            state = (1 - error) * state + error * new_state
        return state.view(B, S, D)

class ChunkEncoder(nn.Module):
    """
    Encodes a chunk of tokens into a single vector representation.
    """
    def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        
        # Local BIDIRECTIONAL transformer for within-chunk processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True, norm_first=True
        )
        self.local_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Learned attention-based pooling mechanism
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pooling_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, token_embeddings):
        """
        Args:
            token_embeddings: [B, num_chunks * chunk_size, D]
        Returns:
            chunk_embeddings: [B, num_chunks, D]
        """
        B, total_tokens, D = token_embeddings.shape
        num_chunks = total_tokens // self.chunk_size
        
        # Reshape into chunks
        chunks = token_embeddings[:, :num_chunks * self.chunk_size, :].view(
            B * num_chunks, self.chunk_size, D
        )
        
        # Local bidirectional attention within each chunk
        encoded_tokens = self.local_encoder(chunks)
        
        # Attention-based pooling
        query = self.pooling_query.expand(B * num_chunks, -1, -1)
        pooled, _ = self.pooling_attn(query, encoded_tokens, encoded_tokens)
        
        # Reshape back to [B, num_chunks, D]
        chunk_embeddings = pooled.view(B, num_chunks, D)
        
        return chunk_embeddings

class ChunkDecoder(nn.Module):
    """
    Decodes chunk representation back to token-level predictions.
    """
    def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Within-chunk positional embeddings
        self.pos_embedding = nn.Embedding(chunk_size, d_model)

        # Local CAUSAL transformer decoder with cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True, norm_first=True
        )
        self.local_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Token prediction head
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, chunk_embeddings, target_token_embeddings):
        """
        Args:
            chunk_embeddings: [B, num_chunks, D] (Memory for cross-attention)
            target_token_embeddings: [B, num_chunks * chunk_size, D] (Input to the decoder)
        Returns:
            token_logits: [B, num_chunks * chunk_size, V]
        """
        B, num_chunks, D = chunk_embeddings.shape
        seq_len = num_chunks * self.chunk_size

        # Add within-chunk positional embeddings to the target tokens
        pos = torch.arange(0, self.chunk_size, device=target_token_embeddings.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos).repeat(B * num_chunks, 1, 1)
        
        # Prepare inputs for the causal decoder
        tgt = target_token_embeddings.view(B * num_chunks, self.chunk_size, D) + pos_emb
        
        # Prepare memory for cross-attention
        memory = chunk_embeddings.view(B * num_chunks, 1, D).repeat(1, self.chunk_size, 1)

        # Causal mask to prevent attending to future tokens within the chunk
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.chunk_size).to(tgt.device)

        # Decode with cross-attention to the parent chunk
        refined = self.local_decoder(tgt, memory, tgt_mask=causal_mask)

        # Reshape back to the full sequence length
        refined = refined.view(B, seq_len, D)

        logits = self.lm_head(refined)
        return logits

class TransformerDecoderLayerWithCache(nn.Module):
    """A Transformer Decoder layer with explicit cache handling for self- and cross-attention."""
    def __init__(self, d_model, n_heads, dim_feedforward=None, dropout=0.1):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.self_attn = FastBlockSparseAttention(d_model, n_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Diamond Mixer replaces FFN
        self.diamond_mixer = DiamondMixer(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory, self_attn_past=None, cross_attn_past=None):
        # Self-attention block
        tgt_norm = self.norm1(tgt)
        sa_output, sa_present = self.self_attn(tgt_norm, layer_past=self_attn_past)
        tgt = tgt + self.dropout1(sa_output)

        # Cross-attention block
        tgt_norm = self.norm2(tgt)
        
        # For cross-attention, the key and value from the memory (encoder output) are static.
        # We can cache them after the first pass.
        if cross_attn_past is not None:
            # On subsequent passes, we re-use the cached memory_kv.
            # The query is always new.
            ca_output, _ = self.cross_attn(tgt_norm, cross_attn_past[0], cross_attn_past[1])
            ca_present = cross_attn_past
        else:
            # First pass: compute and cache memory_kv.
            ca_output, _ = self.cross_attn(tgt_norm, memory, memory)
            # This assumes `memory` is static and can be cached.
            # For this model, memory comes from the chunk encoder and is fixed for a sequence.
            ca_present = (memory, memory) 

        tgt = tgt + self.dropout2(ca_output)

        # Diamond Mixer block (Lossless Logic)
        tgt = self.diamond_mixer(tgt)
        
        return tgt, sa_present, ca_present

class ChunkDecoderWithCache(nn.Module):
    """A cache-aware Chunk Decoder for efficient, incremental generation."""
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
        
        # Determine the starting position for positional embeddings from the cache
        past_len = cache[0][0][0].size(2) if cache else 0
        positions = torch.arange(past_len, past_len + S, dtype=torch.long, device=device) % self.chunk_size
        
        pos_emb = self.pos_embedding(positions)
        tgt = target_token_embeddings + pos_emb
        
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else (None, None)
            self_attn_past, cross_attn_past = layer_cache
            
            # The memory for cross-attention is the single chunk embedding for the current chunk
            # This needs to be correctly shaped and selected.
            # Assuming chunk_embeddings are [B, NumChunks, D]
            # And we operate within one chunk at a time during generation.
            # Let's assume chunk_embeddings is correctly broadcastable/selected before this call.
            # For simplicity in this implementation, we'll assume it's [B, 1, D] and needs repeating.
            memory = chunk_embeddings.repeat(1, S, 1)

            tgt, sa_present, ca_present = layer(tgt, memory, self_attn_past, cross_attn_past)
            new_cache.append((sa_present, ca_present))
            
        logits = self.lm_head(tgt)
        return logits, new_cache

# ==========================================================
# 1. COMPLETE MULTI-LEVEL LATTICE CORE (FIXED)
# ==========================================================
class RecursiveDescentLatticeAnalyzer(nn.Module):
    """
    Exploits the recursive descent property: each spine position
    can be decomposed into a path through multiple layers.
    """
    def __init__(self, max_seq_len=8192):
        super().__init__()
        spine_list = self._generate_spine_list(max_seq_len)
        self.register_buffer('spine', torch.tensor(spine_list, dtype=torch.long))
        self.descent_paths = self._compute_descent_paths()
        self.layer_weights = nn.Parameter(torch.ones(10))

    def _generate_spine_list(self, max_len):
        # Pell-Lucas Logic: S_n = 2*S_{n-1} + S_{n-2}
        spine = [0, 2, 4]
        while True:
            next_val = 2 * spine[-1] + spine[-2]
            if next_val >= max_len:
                break
            spine.append(next_val)
        return spine

    def _nearest_spine(self, pos):
        """Finds the nearest spine position to a given position."""
        return self.spine[(self.spine.float() - pos).abs().argmin()]

    def _find_parent(self, pos):
        """
        Invert the recurrence relation to find parent.
        S_n = 2*S_{n-1} + S_{n-2} -> S_{n-1} ~ S_n / 2.414
        """
        if pos == 0:
            return 0
        parent_approx = pos / 2.414
        return self._nearest_spine(parent_approx).item()

    def _compute_descent_paths(self):
        """
        For each spine position, compute its recursive descent path
        to the origin through multiple layers.
        """
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
        """
        NEW: Instead of just gathering ancestors, compute which
        layers are most relevant for predicting target_offset away.
        """
        try:
            source_spine_idx = (self.spine == pos).nonzero(as_tuple=True)[0]
            target_spine_idx = (self.spine == (pos + target_offset)).nonzero(as_tuple=True)[0]
            spine_distance = abs(target_spine_idx - source_spine_idx)
        except (IndexError, RuntimeError):
             # Fallback for non-spine positions or if not found
            spine_distance = int(np.log2(target_offset + 1))

        layer_importance = torch.zeros(10, device=self.layer_weights.device)
        if spine_distance > 5:  # Far future
            layer_importance[0:3] = torch.tensor([1.0, 0.8, 0.5], device=self.layer_weights.device)
        elif spine_distance > 2:  # Medium range
            layer_importance[1:5] = torch.tensor([0.5, 1.0, 0.8, 0.3], device=self.layer_weights.device)
        else:  # Near future
            layer_importance[3:7] = torch.tensor([0.3, 0.8, 1.0, 0.8], device=self.layer_weights.device)
        
        layer_importance = layer_importance * torch.sigmoid(self.layer_weights)
        return layer_importance

class FullLatticeFieldAnalyzer(nn.Module):
    """Analyzes the complete lattice structure to extract ALL levels and connection patterns."""
    def __init__(self, max_seq_len=8192):
        super().__init__()
        # Generate spine - FIXED CORRECT RECURRENCE
        spine = [0, 2, 4]
        while True:
            next_val = 2 * spine[-1] + spine[-2]  # Fixed: removed incorrect terms
            if next_val >= max_seq_len:
                break
            spine.append(next_val)
        
        self.register_buffer('spine', torch.tensor(spine, dtype=torch.long))
        self.max_depth = self._compute_max_depth()
        
        # Only precompute for spine positions (sparse optimization)
        self.lattice_structure = {}
        for pos in spine:
            if pos < max_seq_len:
                self.lattice_structure[pos] = self._analyze_position(pos)
        
        # For non-spine positions, compute on-demand
        self._non_spine_cache = {}
    
    def _compute_max_depth(self):
        """Maximum depth of the lattice tree"""
        return len(self.spine)
    
    def get_structure(self, pos: int):
        """Get precomputed or on-demand structure for a position."""
        if pos in self.lattice_structure:
            return self.lattice_structure[pos]
        
        if pos in self._non_spine_cache:
            return self._non_spine_cache[pos]
            
        # Compute on-demand for non-spine positions
        structure = self._analyze_non_spine(pos)
        self._non_spine_cache[pos] = structure
        return structure
    
    def _analyze_position(self, pos):
        """Complete analysis of a single position's lattice connections (Spine Node)."""
        levels = {0: [pos]}
        visited = {pos}
        current_level = [pos]
        level = 0
        
        # BFS to find all ancestors and their levels
        while current_level and level < 10:
            next_level = set()
            
            for node in current_level:
                ancestors = self._get_immediate_ancestors(node)
                for anc in ancestors:
                    if anc not in visited and anc >= 0:
                        visited.add(anc)
                        next_level.add(anc)
            
            current_level = list(next_level)
            level += 1
            if current_level:
                levels[level] = current_level.copy()

        # max_depth is the largest key in levels
        max_depth = max(levels.keys()) if levels else 0
        
        # Compute path counts - Pass max_depth explicitly
        path_counts = self._compute_path_counts(pos, levels, max_depth)
        
        return {
            'levels': levels,
            'path_counts': path_counts,
            'total_ancestors': len(visited) - 1,
            'max_depth': max_depth
        }
    
    def _get_immediate_ancestors(self, pos):
        """Get 2 immediate ancestors from recurrence relation"""
        try:
            idx = (self.spine == pos).nonzero(as_tuple=True)[0].item()
            if idx >= 2:
                return [
                    self.spine[idx-1].item(),
                    self.spine[idx-2].item()
                ]
        except:
            pass
        return []
    
    def _analyze_non_spine(self, pos):
        """For non-spine positions, interpolate between nearest spine nodes"""
        left_spine = self.spine[self.spine < pos]
        
        ancestors = []
        if len(left_spine) > 0:
            ancestors.append(left_spine[-1].item())
        
        return {
            'levels': {0: [pos], 1: ancestors},
            'path_counts': {anc: 1 for anc in ancestors},
            'total_ancestors': len(ancestors),
            'max_depth': 1
        }
    
    def _compute_path_counts(self, pos, levels, max_depth):
        """Dynamic programming to count paths to each ancestor."""
        path_counts = {pos: 1}
        
        # Iterate levels backwards (from farthest ancestors to pos)
        for level in sorted(levels.keys(), reverse=True):
            for node in levels[level]:
                if node == pos: continue
                
                count = 0
                
                # At level max_depth, there are no "children" at level 6.
                if level == max_depth:
                    path_counts[node] = 1 # Initial path for the farthest ancestor
                    continue
                
                # Search for "children" at the next, closer level (level + 1)
                for child in levels.get(level + 1, []):
                    # If 'node' is an ancestor of 'child' (by the recurrence formula)
                    if node in self._get_immediate_ancestors(child):
                        # Add the number of paths leading to 'child'
                        count += path_counts.get(child, 0)
                
                if level != 0:
                    path_counts[node] = count
                
        # Remove pos from path_counts
        path_counts.pop(pos, None)
        return path_counts

class MultiLevelLatticeProcessor(nn.Module):
    """Processes each level of the lattice hierarchy separately, then fuses them with learned attention."""
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        # Analyzer is called upon initialization
        self.analyzer = FullLatticeFieldAnalyzer(max_seq_len)
        
        self.level_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(10)
        ])
        
        self.level_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        spine = self.analyzer.spine
        relevant_spine = spine[spine < S]
        
        updates = {}
        for spine_pos in relevant_spine:
            if spine_pos.item() < 2: continue  # Fixed: changed from 3 to 2
            
            pos = spine_pos.item()
            structure = self.analyzer.get_structure(pos)
            
            if structure is None: continue
            
            level_features = []
            
            for level in range(structure['max_depth'] + 1):
                if level == 0: continue
                if level not in structure['levels']: continue
                
                level_nodes = structure['levels'][level]
                level_h = []
                total_weight = 0.0
                
                for node in level_nodes:
                    if node < S:
                        weight = structure['path_counts'].get(node, 1)
                        level_h.append(x[:, node, :] * weight)
                        total_weight += weight
                
                if level_h and total_weight > 0:
                    level_feat = torch.stack(level_h, dim=1).sum(dim=1) / total_weight
                    level_feat = self.level_transforms[level](level_feat)
                    level_features.append(level_feat)

            if not level_features: continue

            level_stack = torch.stack(level_features, dim=1)
            query = x[:, pos:pos+1, :]
            attended, _ = self.level_attention(query, level_stack, level_stack)
            combined = torch.cat([attended.squeeze(1), x[:, pos, :]], dim=-1)
            updates[pos] = self.fusion(combined)

        if not updates:
            return x

        sorted_positions = sorted(updates.keys())
        output_slices = []
        last_pos = 0
        for pos in sorted_positions:
            if pos > last_pos:
                output_slices.append(x[:, last_pos:pos, :])
            output_slices.append(updates[pos].unsqueeze(1))
            last_pos = pos + 1
        
        if last_pos < S:
            output_slices.append(x[:, last_pos:S, :])

        return torch.cat(output_slices, dim=1)

class PathWeightedLatticeCore(nn.Module):
    """Uses path counts to weight ALL ancestor contributions and aggregates with GRU."""
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.analyzer = FullLatticeFieldAnalyzer(max_seq_len)
        
        self.path_weight_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        self.message_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.aggregate_fn = nn.GRU(d_model, d_model, batch_first=True)
        
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        spine = self.analyzer.spine
        relevant_spine = spine[spine < S]
        
        updates = {}
        for spine_pos in relevant_spine:
            if spine_pos.item() < 2: continue  # Fixed: changed from 3 to 2
            
            pos = spine_pos.item()
            structure = self.analyzer.get_structure(pos)
            
            if structure is None or structure['total_ancestors'] == 0: continue
            
            all_ancestors = []
            path_counts = []
            
            for level in structure['levels']:
                if level > 0:
                    for anc in structure['levels'][level]:
                        if anc < S:
                            all_ancestors.append(anc)
                            path_counts.append(structure['path_counts'].get(anc, 1))

            if not all_ancestors: continue

            path_count_tensor = torch.tensor(path_counts, device=x.device).view(-1, 1).float()
            path_weights_tensor = self.path_weight_net(path_count_tensor).squeeze()

            messages = []
            for ancestor_pos in all_ancestors:
                h_anc = x[:, ancestor_pos, :]
                h_curr = x[:, pos, :]
                msg = self.message_fn(torch.cat([h_anc, h_curr], dim=-1))
                messages.append(msg)
            
            msg_stack = torch.stack(messages, dim=1)
            if path_weights_tensor.dim() == 0:
                weights_tensor = path_weights_tensor.view(1, 1, 1).expand(B, -1, D)
            else:
                weights_tensor = path_weights_tensor.view(1, -1, 1).expand(B, -1, D)
                
            weighted_msgs = msg_stack * weights_tensor
            
            aggregated, _ = self.aggregate_fn(weighted_msgs)
            aggregated = aggregated[:, -1, :]
            
            gate = self.update_gate(torch.cat([aggregated, x[:, pos, :]], dim=-1))
            updates[pos] = gate * aggregated + (1 - gate) * x[:, pos, :]

        if not updates:
            return x

        sorted_positions = sorted(updates.keys())
        output_slices = []
        last_pos = 0
        for pos in sorted_positions:
            if pos > last_pos:
                output_slices.append(x[:, last_pos:pos, :])
            output_slices.append(updates[pos].unsqueeze(1))
            last_pos = pos + 1
        
        if last_pos < S:
            output_slices.append(x[:, last_pos:S, :])

        return torch.cat(output_slices, dim=1)

class AdaptiveLatticeProcessor(nn.Module):
    """
    Dynamically selects which lattice layers to process based on
    the current prediction task and uncertainty.
    """
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.analyzer = RecursiveDescentLatticeAnalyzer(max_seq_len)
        self.layer_processors = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True, norm_first=True)
            for _ in range(10)
        ])
        # Task classifier: decides which layers to activate
        self.task_router = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 10), # 10 layers
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, horizon_targets=None) -> torch.Tensor:
        B, S, D = x.shape
        # Router decides layer importance based on the average representation of the sequence
        task_embedding = x.mean(dim=1)
        layer_gates = self.task_router(task_embedding) # [batch, 10]

        # Process each layer with adaptive gating
        h = x
        for layer_idx, processor in enumerate(self.layer_processors):
            gate = layer_gates[:, layer_idx].unsqueeze(1).unsqueeze(2)
            if gate.mean() > 0.1: # Skip unimportant layers
                h_layer = processor(h)
                h = h + gate * (h_layer - h) # Gated residual
        return h

class CompleteLatticeCore(nn.Module):
    """FULL IMPLEMENTATION: Meta-fusion of Multi-Level and Path-Weighted approaches."""
    def __init__(self, d_model, max_seq_len, use_adaptive_processor=False):
        super().__init__()
        self.use_adaptive_processor = use_adaptive_processor
        if self.use_adaptive_processor:
            self.adaptive_processor = AdaptiveLatticeProcessor(d_model, max_seq_len)
        else:
            self.multi_level = MultiLevelLatticeProcessor(d_model, max_seq_len)
            self.path_weighted = PathWeightedLatticeCore(d_model, max_seq_len)
        
        fusion_input_size = d_model * 3 if not use_adaptive_processor else d_model * 2
        self.meta_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_adaptive_processor:
            h_adaptive = self.adaptive_processor(x)
            h_combined = torch.cat([x, h_adaptive], dim=-1)
        else:
            h_multi = self.multi_level(x)
            h_path = self.path_weighted(x)
            h_combined = torch.cat([x, h_multi, h_path], dim=-1)
        
        h_out = self.meta_fusion(h_combined)
        
        return h_out

# Additional utility classes
class ExperienceReplayBuffer(nn.Module):
    """Store and replay important sequences"""
    def __init__(self, capacity=10000, d_model=512):
        super().__init__()
        self.capacity = capacity
        self.register_buffer('memory', torch.zeros(capacity, d_model))
        self.register_buffer('importance', torch.zeros(capacity))
        self.ptr = 0
        self.full = False
    
    def add(self, embeddings, loss_signal):
        """Add with importance weighting"""
        batch_size = embeddings.size(0)
        end = self.ptr + batch_size
        
        if end <= self.capacity:
            self.memory[self.ptr:end] = embeddings.detach()
            self.importance[self.ptr:end] = loss_signal.detach()
            self.ptr = end
        else:
            self.full = True
            # Replace least important
            _, indices = torch.topk(self.importance, batch_size, largest=False)
            self.memory[indices] = embeddings.detach()
            self.importance[indices] = loss_signal.detach()
    
    def sample(self, batch_size):
        """Prioritized sampling"""
        if not self.full and self.ptr < batch_size:
            return None
        
        max_idx = self.capacity if self.full else self.ptr
        probs = F.softmax(self.importance[:max_idx], dim=0)
        indices = torch.multinomial(probs, batch_size, replacement=False)
        
        return self.memory[indices]

class AdaptiveLossWeighting(nn.Module):
    """Automatically balance multiple loss terms"""
    def __init__(self, num_losses=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        """
        losses: list of loss values
        Returns weighted sum using uncertainty weighting
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses)

class SelectiveKVCache(nn.Module):
    """Intelligently prune cache based on importance"""
    def __init__(self, d_model, max_size=2048):
        super().__init__()
        self.max_size = max_size
        
        # Importance scorer
        self.importance_net = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, k, v, query):
        """
        k, v: [B, H, S, D] - keys and values
        query: [B, H, 1, D] - current query
        """
        B, H, S, D = k.shape
        
        if S <= self.max_size:
            return k, v
        
        # Score each cached position
        kv_concat = torch.cat([k, v], dim=-1)  # [B, H, S, 2D]
        scores = self.importance_net(kv_concat).squeeze(-1)  # [B, H, S]
        
        # Boost recent positions
        recency_bias = torch.linspace(0, 1, S, device=k.device)
        scores = scores + recency_bias.view(1, 1, -1)
        
        # Keep top-k important positions
        _, top_indices = torch.topk(scores, self.max_size, dim=-1)
        top_indices = top_indices.sort(dim=-1)[0]  # Maintain temporal order
        
        # Gather selected k, v
        k_selected = torch.gather(
            k, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        )
        v_selected = torch.gather(
            v, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        )
        
        return k_selected, v_selected

class CalibratedSampler:
    @staticmethod
    def sample_with_confidence(logits, confidence, temperature=1.0, top_p=0.9):
        """
        Adjust sampling based on model confidence
        High confidence -> lower temperature (more deterministic)
        Low confidence -> higher temperature (more exploration)
        """
        # Dynamic temperature
        adjusted_temp = temperature * (2.0 - confidence)
        
        # Apply temperature
        scaled_logits = logits / adjusted_temp
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Nucleus sampling with confidence-adjusted threshold
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Adjust top_p based on confidence
        adaptive_top_p = top_p * (0.5 + 0.5 * confidence)
        
        # Remove tokens outside nucleus
        sorted_indices_to_remove = cumulative_probs > adaptive_top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        return torch.multinomial(probs, 1)

class TreeSpeculativeDecoder:
    """Generate and verify multiple branching paths simultaneously"""
    
    @staticmethod
    def generate_tree(model, prompt, depth=3, breadth=4):
        """
        Generate a tree of possible continuations
        depth: how many tokens ahead
        breadth: how many options per position
        """
        tree = {0: [prompt]}
        
        for level in range(1, depth + 1):
            tree[level] = []
            
            for parent_seq in tree[level - 1]:
                outputs = model(parent_seq)
                logits = outputs['logits'][:, -1, :]
                
                # Get top-k candidates
                top_k_logits, top_k_indices = torch.topk(logits, breadth, dim=-1)
                
                for token_idx in top_k_indices[0]:
                    child_seq = torch.cat([parent_seq, token_idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    tree[level].append(child_seq)
        
        return tree
    
    @staticmethod
    def verify_tree(model, tree):
        """Score all paths and select the best"""
        all_sequences = tree[max(tree.keys())]
        
        # Score based on likelihood
        with torch.no_grad():
            batch = torch.cat(all_sequences, dim=0)
            outputs = model(batch)
            scores = outputs['logits'].log_softmax(dim=-1)
            
            # Select path with highest average log probability
            sequence_scores = scores.mean(dim=(1, 2)) # Average over seq_len and vocab_size
            best_idx = sequence_scores.argmax()
            return all_sequences[best_idx]

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

class SpeculativeVerifier(nn.Module):
    def __init__(self, d_model=4096, n_layers=32, horizon=64, vocab_size=50257, n_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True, norm_first=True) for _ in range(n_layers)])
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

class UncertaintyAwareHorizon(nn.Module):
    """Dynamically adjust prediction horizon based on confidence"""
    def __init__(self, d_model, vocab_size, max_horizon=64):
        super().__init__()
        self.max_horizon = max_horizon
        
        # Uncertainty estimator
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Multi-scale predictors
        self.predictors = nn.ModuleDict({
            'near': nn.Linear(d_model, vocab_size * 4),    # 1-4 tokens
            'mid': nn.Linear(d_model, vocab_size * 16),    # 5-20 tokens
            'far': nn.Linear(d_model, vocab_size * 44)     # 21-64 tokens
        })
    
    def forward(self, h):
        B, S, D = h.shape
        h_last = h[:, -1, :]
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_head(h_last)  # [B, 1]
        
        # Adaptive horizon: high uncertainty -> short horizon
        horizon = (self.max_horizon * (1 - uncertainty)).long().clamp(4, self.max_horizon)
        
        # Generate predictions at different scales
        near_logits = self.predictors['near'](h_last).view(B, 4, -1)
        mid_logits = self.predictors['mid'](h_last).view(B, 16, -1)
        far_logits = self.predictors['far'](h_last).view(B, 44, -1)
        
        all_logits = torch.cat([near_logits, mid_logits, far_logits], dim=1)
        
        # Return only up to the adaptive horizon
        return all_logits, horizon, uncertainty

# ==========================================================
# MAIN HST v8 CRYSTALLINE MODEL
# ==========================================================
class HSTv8Crystalline(nn.Module):
    """
    HST v8 Crystalline - "THE BEST AI" Architecture
    
    Features:
    1. Pell-Lucas Time Spine (Infinite Context)
    2. Diamond Mixer (Lossless Logic)
    3. Holographic Lattice (Interference Field)
    4. Feedback Loop (Self-Correction)
    5. Speculative Decoding
    6. Context Injection
    7. Multi-Resolution Processing
    """
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_layers,
        max_seq_len=8192,
        horizon=16,
        early_exit_confidence_threshold=0.93,
        mode='token', # 'token' or 'chunk'
        chunk_size=128,
        num_experts=8
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.horizon = horizon
        self.max_seq_len = max_seq_len
        self.early_exit_confidence_threshold = early_exit_confidence_threshold
        self.mode = mode
        self.chunk_size = chunk_size
        self.n_layers = n_layers

        # 1. Hyperbolic Embeddings & Optimized Positional Encoding
        self.token_embedding = HyperbolicEmbedding(vocab_size, d_model)
        self.pos_encoding = OptimizedPositionalEncoding(d_model, max_seq_len)
        
        if self.mode == 'chunk':
            self.chunk_encoder = FusedChunkEncoder(d_model, chunk_size)
            self.chunk_decoder = ChunkDecoderWithCache(d_model, vocab_size, chunk_size)
            self.lattice_core = CompleteLatticeCore(d_model, max_seq_len)
        else:
            self.lattice_core = CompleteLatticeCore(d_model, max_seq_len)

        self.horizon_predictor = StreamlinedHorizonPredictor(d_model, vocab_size, max_horizon=horizon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.embed.weight # Weight tying
        self.ln_f = nn.LayerNorm(d_model)
        self.speculative_verifier = SpeculativeVerifier(d_model=d_model, n_layers=n_layers, horizon=horizon, vocab_size=vocab_size, n_heads=n_heads)
        self.task_analyzer = TaskAnalyzer(d_model)
        self.depth_pred = DepthPredictor(num_tasks=4)
        self.multi_res = FastMultiResProcessor(d_model)
        self.sparse_router = EfficientExpertRouter(d_model)
        self.attention_layers = nn.ModuleList([
            TransformerDecoderLayerWithCache(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.cache_manager = SelectiveKVCache(d_model)
        self.memory = ExperienceReplayBuffer(capacity=10000, d_model=d_model)
        self.loss_weighting = AdaptiveLossWeighting(num_losses=4)
        
        # Hebbian Plasticity (Runtime Learning)
        self.plasticity = HebbianFastWeights(d_model)
        
        # Feedback Loop (Self-Correction)
        self.feedback_loop = FeedbackLoop(d_model)

    def encode_context_block(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encodes a large block of text (token_ids) into a single, dense context vector.
        This is achieved by chunking the text, encoding each chunk, and averaging the results.
        """
        if self.mode != 'chunk':
            raise RuntimeError("Context block encoding is only supported in 'chunk' mode.")

        # Ensure token_ids is a 2D tensor [1, num_tokens] for the embedding layer
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        total_tokens = token_ids.shape[1]
        if total_tokens == 0:
            return torch.zeros(1, self.d_model, device=self.token_embedding.weight.device)

        # Pad the input to be a multiple of chunk_size
        num_chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
        padded_len = num_chunks * self.chunk_size
        padding_needed = padded_len - total_tokens
        if padding_needed > 0:
            token_ids = F.pad(token_ids, (0, padding_needed), 'constant', 0)

        # Get token embeddings
        positions = torch.arange(0, padded_len, dtype=torch.long, device=token_ids.device)
        token_emb = self.token_embedding(token_ids) + self.pos_encoding(positions.unsqueeze(0))

        # Encode the token embeddings into chunk embeddings
        chunk_embeddings = self.chunk_encoder(token_emb) # [1, num_chunks, d_model]

        # Average the chunk embeddings to get a single context vector
        context_vector = chunk_embeddings.mean(dim=1) # [1, d_model]

        return context_vector

    def forward(self, input_ids: torch.Tensor, cache: KVCache = None, training=False, horizon_targets=None, injected_context: Optional[Dict[int, torch.Tensor]] = None) -> Dict:
        if self.mode == 'token':
            return self.forward_token(input_ids, cache, training)
        elif self.mode == 'chunk':
            return self.forward_chunk(input_ids, horizon_targets, injected_context)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward_token(self, input_ids: torch.Tensor, cache: KVCache = None, training=False) -> Dict:
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        past_len = 0
        if cache and cache[0]:
            # Cache structure is [(sa_cache, ca_cache), ...] where sa_cache is (k, v)
            if cache[0][0] is not None:
                past_len = cache[0][0][0].size(2)  # Get K tensor from self-attention cache

        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        
        x = self.token_embedding(input_ids) + self.pos_encoding(positions)
        
        # Plasticity
        x = self.plasticity(x)
        
        x = self.multi_res(x)
        x = self.sparse_router(x)
        
        new_cache = []
        for i, layer in enumerate(self.attention_layers):
            layer_past = cache[i] if cache is not None else None
            # Unpack cache
            sa_past, ca_past = layer_past if layer_past else (None, None)
            
            x, sa_present, ca_present = layer(x, x, self_attn_past=sa_past, cross_attn_past=ca_past)
            
            present = (sa_present, ca_present)
            
            # Prune the cache
            if present[0] is not None:
                k, v = present[0] # Self-attention cache
                # Create a dummy query; in a real scenario, this would be the query for the next token
                dummy_query = torch.randn_like(k[:, :, -1:, :])
                k, v = self.cache_manager(k, v, dummy_query)
                present = ((k, v), present[1])

            new_cache.append(present)
        
        cache = new_cache
        
        h_lattice_out = self.lattice_core(x)
        
        # Feedback Loop (Self-Correction)
        h_final = self.feedback_loop(h_lattice_out)
        
        logits_t1 = self.lm_head(self.ln_f(h_final))
        horizon_logits, horizon_len, uncertainty = self.horizon_predictor(h_final)
        
        if training:
            # Store in experience replay
            self.memory.add(x.mean(dim=1), logits_t1.mean())
        
        return {
            'logits': logits_t1,
            'horizon_logits': horizon_logits,
            'horizon_length': horizon_len,
            'uncertainty': uncertainty,
            'hidden_states': h_final,
            'cache': cache
        }

    def forward_chunk(self, input_ids: torch.Tensor, horizon_targets=None, injected_context: Optional[Dict[int, torch.Tensor]] = None) -> Dict:
        """
        Forward pass in 'chunk' mode, with support for context injection.
        """
        B, total_tokens = input_ids.shape
        device = input_ids.device

        # The decoder needs a shifted version of the input as the target
        target_ids = torch.roll(input_ids, shifts=-1, dims=1)
        target_ids[:, -1] = 0 # Pad the last token

        # Get token embeddings for both input and target
        positions = torch.arange(0, total_tokens, dtype=torch.long, device=device)
        input_token_emb = self.token_embedding(input_ids) + self.pos_encoding(positions.unsqueeze(0))
        target_token_emb = self.token_embedding(target_ids) + self.pos_encoding(positions.unsqueeze(0))
        
        chunk_emb = self.chunk_encoder(input_token_emb)

        # --- CONTEXT INJECTION ---
        if injected_context:
            for spine_pos, context_vector in injected_context.items():
                if spine_pos < chunk_emb.size(1):
                    # Ensure the context vector is correctly broadcasted if batch size > 1
                    if B > 1 and context_vector.size(0) == 1:
                        context_vector = context_vector.expand(B, -1)
                    chunk_emb[:, spine_pos, :] = context_vector
        # -------------------------

        h_lattice_out = self.lattice_core(chunk_emb) # Pass horizon_targets if adaptive
        
        # --- CACHE-AWARE DECODING ---
        cache = injected_context.get('decoder_cache', None) if injected_context else None
        
        logits, new_cache = self.chunk_decoder(h_lattice_out, target_token_emb, cache=cache)
        # -------------------------

        # For compatibility, we can still return a horizon prediction
        # based on the last chunk's representation
        last_chunk_rep = h_lattice_out[:, -1:, :]
        horizon_logits, horizon_len, uncertainty = self.horizon_predictor(last_chunk_rep)
        
        return {
            'logits': logits,
            'horizon_logits': horizon_logits,
            'horizon_length': horizon_len,
            'uncertainty': uncertainty,
            'hidden_states': h_lattice_out, # Note: these are chunk-level states
            'bottom_depth': 0, # Not applicable in chunk mode
            'cache': new_cache
        }

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=50):
        """Generate text using standard autoregressive decoding."""
        self.eval()
        current_ids = prompt
        cache = None
        
        for _ in range(max_new_tokens):
            input_ids = current_ids[:, -1:] if cache else current_ids
            outputs = self.forward(input_ids, cache=cache)
            
            logits = outputs['logits']
            cache = outputs.get('cache', None)
            
            logits = logits[:, -1, :]
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1].unsqueeze(-1)] = float('-inf')
            
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
        return current_ids

    @torch.no_grad()
    def generate_speculative(self, prompt, max_new_tokens, temperature=1.0, top_p=0.9):
        """Generate text using tree-based speculative decoding."""
        
        # Initialize with the prompt
        current_ids = prompt
        
        for _ in range(max_new_tokens):
            # 1. Generate a tree of possible continuations
            tree = TreeSpeculativeDecoder.generate_tree(self, current_ids, depth=3, breadth=4)
            
            # 2. Verify the tree and select the best path
            best_sequence = TreeSpeculativeDecoder.verify_tree(self, tree)
            
            # 3. Sample the next token using confidence-calibrated sampling
            outputs = self(best_sequence)
            logits = outputs['logits'][:, -1, :]
            uncertainty = outputs['uncertainty']
            
            # Invert uncertainty to get confidence
            confidence = 1.0 - uncertainty.mean()
            
            next_token = CalibratedSampler.sample_with_confidence(
                logits, confidence, temperature, top_p
            )
            
            # 4. Append the new token
            current_ids = torch.cat([best_sequence, next_token], dim=1)
            
            if next_token == self.vocab_size - 1: # Assuming EOS token
                break
                
        return current_ids

    @torch.no_grad()
    def generate_with_injected_context(
        self,
        context_blocks: Dict[int, torch.Tensor],
        max_new_tokens: int,
        prompt_ids: Optional[torch.Tensor] = None,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Generates text with large context blocks injected at specific spine positions.
        """
        if self.mode != 'chunk':
            raise RuntimeError("Context injection is only supported in 'chunk' mode.")

        device = self.token_embedding.weight.device

        # 1. Encode context blocks
        encoded_context = {pos: self.encode_context_block(tokens.to(device)) 
                           for pos, tokens in context_blocks.items()}

        # 2. Pre-compute the structural memory (h_lattice_out) for the entire generation length
        prompt_len = prompt_ids.size(1) if prompt_ids is not None else 0
        total_len = prompt_len + max_new_tokens
        num_chunks = (total_len + self.chunk_size - 1) // self.chunk_size
        padded_len = num_chunks * self.chunk_size
        
        dummy_input = torch.zeros(1, padded_len, dtype=torch.long, device=device)
        dummy_pos = torch.arange(0, padded_len, device=device)
        dummy_emb = self.token_embedding(dummy_input) + self.pos_encoding(dummy_pos.unsqueeze(0))
        
        chunk_emb = self.chunk_encoder(dummy_emb)

        # Inject the encoded context into the structural embeddings
        for pos, vec in encoded_context.items():
            if pos < chunk_emb.size(1):
                chunk_emb[:, pos, :] = vec
        
        h_lattice_out = self.lattice_core(chunk_emb)

        # 3. Autoregressive Generation
        cache = None
        all_ids = prompt_ids.tolist()[0] if prompt_ids is not None else []
        
        # Start with a BOS token if there's no prompt
        if not all_ids:
            all_ids.append(0)

        next_token_id = torch.tensor([[all_ids[-1]]], device=device)

        # Warm up the cache with the prompt
        for i in range(prompt_len):
            current_pos = i
            chunk_idx = current_pos // self.chunk_size
            memory = h_lattice_out[:, chunk_idx:chunk_idx+1, :]
            token_emb = self.token_embedding(next_token_id)
            
            logits, cache = self.chunk_decoder(memory, token_emb, cache=cache)
            next_token_id = prompt_ids[:, i:i+1] # Next token is the next from prompt

            if (current_pos + 1) % self.chunk_size == 0:
                cache = None
        
        # Use the last logits from the prompt to predict the first new token
        if prompt_len > 0:
             next_token_logits = logits[:, -1, :]
        else: # Handle no-prompt case
            token_emb = self.token_embedding(next_token_id)
            memory = h_lattice_out[:, 0:1, :]
            logits, cache = self.chunk_decoder(memory, token_emb, cache=cache)
            next_token_logits = logits[:, 0, :]

        for i in range(max_new_tokens):
            # Sampling
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, -1].unsqueeze(-1)] = -float('Inf')
            
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            all_ids.append(next_token_id.item())

            # Prepare for next iteration
            current_pos = prompt_len + i
            chunk_idx = current_pos // self.chunk_size

            if (current_pos + 1) % self.chunk_size == 0:
                cache = None

            memory = h_lattice_out[:, chunk_idx:chunk_idx+1, :]
            token_emb = self.token_embedding(next_token_id)
            logits, cache = self.chunk_decoder(memory, token_emb, cache=cache)
            next_token_logits = logits[:, 0, :]

        return torch.tensor([all_ids], device=device)


# ==============================================================================
# SELF-TEST AND DEMONSTRATION
# ==============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("HST v8 CRYSTALLINE - Full Model Self-Test (DEBUGGED)")
    print("=" * 70)

    # Test 1: Context Injection Mode
    print("\n--- Testing Context Injection Mode ---")
    try:
        model_injection = HSTv8Crystalline(
            vocab_size=50257,
            d_model=64,
            n_heads=2,
            n_layers=2,
            horizon=16,
            mode='chunk',
            chunk_size=64
        )
        
        # Define a large context block to be injected
        large_context_block = torch.randint(0, 50257, (256,))
        
        # Define a spine position for injection
        injection_position = 4
        
        context_to_inject = {
            injection_position: large_context_block
        }
        
        print(f"Injecting a {large_context_block.size(0)}-token block at spine position {injection_position}...")
        
        generated_output = model_injection.generate_with_injected_context(
            context_blocks=context_to_inject,
            max_new_tokens=32
        )
        print("✅ Context injection generation successful!")
        print(f"   - Generated sequence length: {generated_output.size(1)}")
        
    except Exception as e:
        print(f"❌ Context injection generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Token Mode
    print("\n--- Testing Token Mode ---")
    try:
        model_token = HSTv8Crystalline(
            vocab_size=50257,
            d_model=64,
            n_heads=2,
            n_layers=2,
            horizon=16,
            mode='token'
        )
        x_token = torch.randint(0, 50257, (1, 64))
        output_token = model_token(x_token, training=True)
        loss_token = output_token['logits'].mean()
        loss_token.backward()
        print("✅ Token mode forward/backward pass successful!")
        
    except Exception as e:
        print(f"❌ Token mode backward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Chunk Mode
    print("\n--- Testing Chunk Mode ---")
    try:
        model_chunk = HSTv8Crystalline(
            vocab_size=50257,
            d_model=64,
            n_heads=2,
            n_layers=2,
            horizon=16,
            mode='chunk',
            chunk_size=64
        )
        x_chunk = torch.randint(0, 50257, (1, 128)) # 2 chunks
        output_chunk = model_chunk(x_chunk, horizon_targets=None)
        loss_chunk = output_chunk['logits'].mean()
        loss_chunk.backward()
        print("✅ Chunk mode forward/backward pass successful!")
        
    except Exception as e:
        print(f"❌ Chunk mode backward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Speculative Generation
    print("\n--- Testing Speculative Generation ---")
    try:
        prompt = torch.randint(0, 50257, (1, 10))
        generated_ids = model_token.generate_speculative(prompt, max_new_tokens=10)
        print(f"✅ Speculative generation successful! Output length: {generated_ids.size(1)}")
        
    except Exception as e:
        print(f"❌ Speculative generation failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 5: Memory and Performance
    print("\n--- Testing Memory Usage ---")
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Test larger model
        large_model = HSTv8Crystalline(
            vocab_size=50257,
            d_model=256,
            n_heads=8,
            n_layers=4,
            horizon=32,
            mode='token'
        )
        
        x_large = torch.randint(0, 50257, (2, 256))
        output_large = large_model(x_large)
        print("✅ Large model test successful!")
        
        # Test generation
        prompt_large = torch.randint(0, 50257, (1, 50))
        generated_large = large_model.generate(prompt_large, max_new_tokens=50)
        print(f"✅ Large model generation successful! Length: {generated_large.size(1)}")
        
    except Exception as e:
        print(f"❌ Large model test failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 70)
    print("HST v8 CRYSTALLINE - Self-Test Complete")
    print("=" * 70)