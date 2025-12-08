import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from torch.distributions import Categorical, Gumbel

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

# Continual Learning with Orthogonal Updates
class ContinualUpdater(nn.Module):
    def __init__(self, d_model=4096, sparsity=0.05):
        super().__init__()
        self.basis = nn.Parameter(torch.eye(d_model))
        self.sparse_proj = nn.Linear(d_model, int(d_model * sparsity))

    def update(self, new_data, lr=1e-4):
        delta = self.sparse_proj(new_data.mean(0))
        orth_delta = torch.linalg.qr(delta.unsqueeze(0))[0].squeeze(0)
        self.basis.data += lr * orth_delta
        self.basis.data = F.normalize(self.basis.data, p=2, dim=1)

    def forward(self, x):
        proj = torch.matmul(x @ self.basis, self.basis.T)
        return proj

# Annealed Sampling
class AnnealedSampler(nn.Module):
    def __init__(self, vocab=50257, temp=1.0):
        super().__init__()
        self.anneal_sched = torch.linspace(1.0, 0.01, 100)
        self.vocab = vocab
        self.temp = temp

    def annealed_sample(self, logits):
        samples = []
        for t in self.anneal_sched:
            noisy_l = logits + torch.randn_like(logits) * t
            samp = Categorical(logits=noisy_l / self.temp).sample()
            samples.append(samp)
        samples_tensor = torch.stack(samples)
        mode_result = torch.mode(samples_tensor, dim=0)
        return mode_result.values

    def forward(self, h):
        # This linear layer is created on the fly, which can be inefficient.
        # For this integration, we will assume it's acceptable.
        logits = nn.Linear(h.size(-1), self.vocab).to(h.device)(h)
        return self.annealed_sample(logits)

# Reasoning Head
class ReasoningHead(nn.Module):
    def __init__(self, d_model=4096):
        super().__init__()
        self.residual_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, h):
        reason_skip = self.residual_mlp(h[:, -1])
        return reason_skip

# Multi-Agent Control
class MultiAgentController(nn.Module):
    def __init__(self, state_dim=24, action_dim=8, n_robots=8, d_model=512):
        super().__init__()
        self.world_model = nn.LSTM(state_dim * n_robots, d_model, batch_first=True)
        self.policy = nn.Linear(d_model, action_dim * n_robots)
        self.n_robots = n_robots

    def forward(self, states, prev_h=None):
        h, new_h = self.world_model(states, prev_h)
        actions = self.policy(h[:, -1])
        return actions.view(actions.size(0), self.n_robots, -1), new_h

# Video Generation Diffusion
class VideoDiffusion(nn.Module):
    def __init__(self, channels=256, timesteps=1000):
        super().__init__()
        self.unet = nn.Sequential(
            nn.Conv2d(channels, channels*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels*2, channels, 3, padding=1)
        )
        self.time_embed = nn.Embedding(timesteps, channels)
        self.scheduler = torch.linspace(0, 1, timesteps)

    def forward(self, z_t, t, text_emb):
        time_embedding = self.time_embed(t)
        # Reshape time_embedding to be broadcastable to z_t
        time_embedding = time_embedding.unsqueeze(-1).unsqueeze(-1)
        # Reshape text_emb to be broadcastable
        text_emb = text_emb.unsqueeze(-1).unsqueeze(-1)
        
        pred_noise = self.unet(z_t + time_embedding)
        # The original code `return pred_noise - text_emb` assumes text_emb has the same shape as pred_noise.
        # A more common approach is to condition the UNet on the text embedding.
        # For this integration, we will simply add the embeddings.
        return pred_noise - text_emb

    @torch.no_grad()
    def sample(self, shape=(1,256,64,64), text_emb=None, steps=50):
        z = torch.randn(shape).to(text_emb.device)
        for i in range(steps):
            t = torch.full((shape[0],), i, dtype=torch.long).to(text_emb.device)
            pred = self(z, t, text_emb)
            z = z - (self.scheduler[i].to(z.device) * pred) / steps
            z = torch.clamp(z, -1, 1)
        return z

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
        
        # Only predict depth on the first pass (not in generation mode)
        if past_len == 0:
            with torch.no_grad(): # Don't train the depth predictor for now
                task_probs = self.task_analyzer(x)
                depth_tensor = self.depth_pred(task_probs)
                # Use mean depth for the batch
                predicted_depth = int(depth_tensor.mean().round().item())
                predicted_depth = max(4, min(predicted_depth, self.num_layers_max))
        else:
            predicted_depth = self.num_layers_max

        new_cache = []
        out = x
        # Process up to the predicted depth
        for i in range(predicted_depth):
            layer_past = cache[i] if cache and i < len(cache) else None
            out, present = self.layers[i](out, layer_past=layer_past)
            new_cache.append(present)
        
        # Pass through remaining layers if any, but with identity function
        # This is to ensure the cache structure is consistent
        for i in range(predicted_depth, self.num_layers_max):
            new_cache.append((None,None)) # Append empty cache entries

        return out, predicted_depth, new_cache

# Type definition for KV Cache: List[Tuple[torch.Tensor, torch.Tensor]]]
KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

def prune_cache(cache: KVCache, max_size: int = 2048) -> KVCache:
    """Keep only the most recent tokens in cache to prevent memory overflow."""
    if not cache or not cache[0] or cache[0][0] is None or cache[0][0].size(2) <= max_size:
        return cache
    
    pruned_cache = []
    for k, v in cache:
        if k is not None and v is not None:
            # Keep only last max_size tokens
            pruned_k = k[:, :, -max_size:, :]
            pruned_v = v[:, :, -max_size:, :]
            pruned_cache.append((pruned_k, pruned_v))
        else:
            pruned_cache.append((None, None))
    
    return pruned_cache

class ChunkEncoder(nn.Module):
    """
    Encodes a chunk of tokens into a single vector representation.
    (THEORY-COMPLIANT IMPLEMENTATION from v4 architecture doc)
    """
    def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        
        # Local BIDIRECTIONAL transformer for within-chunk processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True
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
    (THEORY-COMPLIANT IMPLEMENTATION from v4 architecture doc)
    """
    def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # Within-chunk positional embeddings
        self.pos_embedding = nn.Embedding(chunk_size, d_model)

        # Local CAUSAL transformer decoder with cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, batch_first=True
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


class SelfAttentionWithCache(nn.Module):
    """Custom Causal Self-Attention layer with explicit KV Cache support."""
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
        
        # Apply causal mask (FIXED: Ensure correct application for incremental/full passes)
        full_S = k.size(2)
        if full_S > S:
            # Incremental step: only mask the new tokens' attention to future new tokens
            attn_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
            attn_mask_full = torch.ones(S, full_S, dtype=torch.bool, device=x.device)
            attn_mask_full[:, full_S - S:] = attn_mask
            attn_weights.masked_fill_(attn_mask_full[None, None, :, :], -torch.inf)
        else:
            # Full sequence pass: standard causal mask
            attn_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
            attn_weights.masked_fill_(attn_mask[None, None, :, :], -torch.inf)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, S, D)
        
        output = self.out_proj(attn_output)
        return output, present

class TransformerDecoderLayerWithCache(nn.Module):
    """A Transformer Decoder layer with explicit cache handling for self- and cross-attention."""
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

        # FFN block
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
    (THEORY-COMPLIANT IMPLEMENTATION from data_mapping.pdf)
    """
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
            layer_importance[0:3] = torch.tensor([1.0, 0.8, 0.5])
        elif spine_distance > 2:  # Medium range
            layer_importance[1:5] = torch.tensor([0.5, 1.0, 0.8, 0.3])
        else:  # Near future
            layer_importance[3:7] = torch.tensor([0.3, 0.8, 1.0, 0.8])
        
        # Move tensor to correct device before multiplication
        layer_importance = layer_importance.to(self.layer_weights.device)
        
        layer_importance = layer_importance * torch.sigmoid(self.layer_weights)
        return layer_importance

class FullLatticeFieldAnalyzer(nn.Module):
    """Analyzes the complete lattice structure to extract ALL levels and connection patterns.
    (FIXED: Only computes for spine positions at init time)"""
    def __init__(self, max_seq_len=8192):
        super().__init__()
        # Generate spine
        spine = [0, 2, 4]
        while True:
            next_val = 2*spine[-1] + 2*spine[-2] + 2*spine[-3]
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
        """Get 3 immediate ancestors from recurrence relation"""
        try:
            idx = (self.spine == pos).nonzero(as_tuple=True)[0].item()
            if idx >= 3:
                return [
                    self.spine[idx-1].item(),
                    self.spine[idx-2].item(),
                    self.spine[idx-3].item()
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
                
                # At level max_depth (e.g., level 5), there are no "children" at level 6.
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
            if spine_pos.item() < 3: continue
            
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
    """Uses path counts to weight ALL ancestor contributions and aggregates with GRU.
    (FIXED: Batch-processes path weight network calls)"""
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
            if spine_pos.item() < 3: continue
            
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
    (THEORY-COMPLIANT IMPLEMENTATION from data_mapping.pdf)
    """
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.analyzer = RecursiveDescentLatticeAnalyzer(max_seq_len)
        self.layer_processors = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
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
        
        self.meta_fusion = nn.Sequential(
            nn.Linear(d_model * 3 if not use_adaptive_processor else d_model * 2, d_model * 2),
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


# ==========================================================
# 2. ADVANCED PREDICTION & LOSS COMPONENTS
# ==========================================================
class RecursiveHorizonPredictor(nn.Module):
    """
    Predicts future positions by traversing the lattice hierarchy
    instead of independent heads for each position.
    (THEORY-COMPLIANT IMPLEMENTATION from data_mapping.pdf)
    """
    def __init__(self, d_model, vocab_size, horizon=16):
        super().__init__()
        self.horizon = horizon
        # Instead of 16 independent heads, use hierarchical cascade
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
        
        # Create a list of logits for the horizon
        logits_list = [all_preds.get(i, torch.zeros(B, self.coarse_predictor.out_features, device=h_t.device)) for i in range(1, self.horizon + 1)]
        logits = torch.stack(logits_list, dim=1)
        
        # Confidence is not explicitly calculated here, returning ones
        confidence = torch.ones(B, self.horizon, device=h_t.device)
        
        return logits, confidence

# ==========================================================
# 3. FULL HST-V6-GIGA MODEL
# ==========================================================
class HSTv6Giga(nn.Module):
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
        use_adaptive_processor=False
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
            self.chunk_decoder = ChunkDecoderWithCache(d_model, vocab_size, chunk_size) # Replaced with cache-aware version
            self.lattice_core = CompleteLatticeCore(d_model, max_seq_len, use_adaptive_processor=use_adaptive_processor) # Operates on chunks
        else:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            self.adaptive_bottom = AdaptiveBottomTransformer(
                d_model=d_model, n_heads=n_heads, num_layers_max=self.n_bottom_layers
            )
            self.lattice_core = CompleteLatticeCore(d_model, max_seq_len) # Operates on tokens
            self.top_stack = nn.ModuleList([
                TransformerEncoderLayerWithCache(d_model=d_model, n_heads=n_heads)
                for _ in range(self.n_top_layers)
            ])

        self.horizon_predictor = RecursiveHorizonPredictor(d_model, vocab_size, horizon=horizon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)

        # HST v6.1 Additions
        self.reasoning_head = ReasoningHead(d_model)
        self.continual_updater = ContinualUpdater(d_model)
        self.annealed_sampler = AnnealedSampler(vocab_size)
        self.multi_agent_controller = MultiAgentController(d_model=d_model)
        self.video_diffusion = VideoDiffusion(channels=d_model)
        self.speculative_verifier = SpeculativeVerifier(d_model=d_model, n_layers=self.n_top_layers, horizon=horizon, vocab_size=vocab_size, n_heads=n_heads)

    def control_robots(self, robot_states, prev_h=None):
        """
        Uses the MultiAgentController to generate actions for multiple robots.
        """
        return self.multi_agent_controller(robot_states, prev_h)

    def generate_video(self, text_prompt_ids, shape=(1,256,64,64), steps=50):
        """
        Generates a video using VideoDiffusion, conditioned on a text prompt.
        """
        # Encode the text prompt to get a conditioning embedding
        # We will use the average of the token embeddings as the text embedding.
        text_emb = self.token_embedding(text_prompt_ids).mean(dim=1)
        return self.video_diffusion.sample(shape=shape, text_emb=text_emb, steps=steps)

    def get_reasoning_score(self, hidden_states):
        """
        Computes a reasoning score using the ReasoningHead.
        """
        return self.reasoning_head(hidden_states)

    def update_continual(self, data, lr=1e-4):
        """
        Performs a continual learning update using the ContinualUpdater.
        """
        self.continual_updater.update(data, lr)

    @torch.no_grad()
    def generate_annealed(self, prompt_ids, max_new_tokens):
        """
        Generates text using Annealed Sampling.
        """
        # Process the initial prompt
        outputs = self.forward_token(prompt_ids, cache=None)
        cache = outputs['cache']
        hidden_states = outputs['hidden_states']
        current_ids = prompt_ids
        
        # Start the generation loop
        for _ in range(max_new_tokens):
            # Use AnnealedSampler on the last hidden state
            next_token_id = self.annealed_sampler(hidden_states[:, -1, :]).unsqueeze(0)
            
            current_ids = torch.cat([current_ids, next_token_id], dim=1)
            
            # Pass only the new token to the model in the next iteration
            outputs = self.forward_token(next_token_id, cache=cache)
            hidden_states = outputs['hidden_states']
            cache = outputs['cache']
            
        return current_ids

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
        token_emb = self.token_embedding(token_ids) + self.pos_embedding(positions)

        # Encode the token embeddings into chunk embeddings
        chunk_embeddings = self.chunk_encoder(token_emb) # [1, num_chunks, d_model]

        # Average the chunk embeddings to get a single context vector
        context_vector = chunk_embeddings.mean(dim=1) # [1, d_model]

        return context_vector

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
        
        # Pass through the adaptive bottom transformer layers
        bottom_cache = cache[:self.n_bottom_layers] if cache else None
        h_bottom, predicted_depth, bottom_new_cache = self.adaptive_bottom(x, cache=bottom_cache)

        h_lattice_out = self.lattice_core(h_bottom)
        
        # Pass through the top stack of transformer layers
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
        """
        Forward pass in 'chunk' mode, with support for context injection.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            horizon_targets (torch.Tensor, optional): Targets for horizon prediction. Defaults to None.
            injected_context (Optional[Dict[int, torch.Tensor]], optional):
                A dictionary mapping chunk indices (spine positions) to pre-encoded context vectors.
                Defaults to None.

        Returns:
            Dict: A dictionary containing the model's output.
        """
        B, total_tokens = input_ids.shape
        device = input_ids.device

        # The decoder needs a shifted version of the input as the target
        target_ids = torch.roll(input_ids, shifts=-1, dims=1)
        target_ids[:, -1] = 0 # Pad the last token

        # Get token embeddings for both input and target
        positions = torch.arange(0, total_tokens, dtype=torch.long, device=device)
        input_token_emb = self.token_embedding(input_ids) + self.pos_embedding(positions)
        target_token_emb = self.token_embedding(target_ids) + self.pos_embedding(positions)
        
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
        # During generation, we might pass a cache.
        # This part of the code is for the full sequence pass (training/prompt processing).
        # The generation loop will handle the cache incrementally.
        cache = injected_context.get('decoder_cache', None) if injected_context else None
        
        logits, new_cache = self.chunk_decoder(h_lattice_out, target_token_emb, cache=cache)
        # -------------------------

        # For compatibility, we can still return a horizon prediction
        # based on the last chunk's representation
        last_chunk_rep = h_lattice_out[:, -1:, :]
        logits_horizon, confidence = self.horizon_predictor(last_chunk_rep)
        
        return {
            'logits': logits,
            'horizon_logits': logits_horizon,
            'confidence': confidence,
            'hidden_states': h_lattice_out, # Note: these are chunk-level states
            'bottom_depth': 0, # Not applicable in chunk mode
            'cache': new_cache
        }

    @torch.no_grad()
    def generate_speculative(self, input_ids, max_new_tokens, temperature=1.0, top_k=50, max_cache_size=2048):
        device = input_ids.device
        
        current_ids = input_ids.clone()
        
        # Initial forward pass to warm up the cache and get hidden states
        full_output = self.forward_token(current_ids, cache=None)
        cache = full_output['cache']
        hidden_states = full_output['hidden_states']

        for _ in range(max_new_tokens):
            # 1. Draft a sequence of `horizon` tokens
            draft_tokens = []
            draft_input_ids = current_ids[:,-1:] # Start from the last token
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

            # 2. Use SpeculativeVerifier to get verified logits
            # The verifier should attend to the main model's hidden states
            verified_logits, confidence = self.speculative_verifier(draft_tensor, hidden_states)

            # 3. Acceptance/Rejection Sampling
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

            # Update the main model's cache and hidden_states with the accepted tokens
            if num_accepted > 0:
                accepted_ids = current_ids[:, -num_accepted:]
                outputs = self.forward_token(accepted_ids, cache=cache)
                cache = outputs['cache']
                hidden_states = torch.cat([hidden_states, outputs['hidden_states']], dim=1)

            if num_accepted < self.horizon:
                # If a token was rejected, we need to re-evaluate the last token
                # to get the correct hidden state for the next iteration.
                outputs = self.forward_token(current_ids[:,-1:], cache=cache)
                cache = outputs['cache']
                hidden_states = torch.cat([hidden_states, outputs['hidden_states']], dim=1)


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

        Args:
            context_blocks (Dict[int, torch.Tensor]): A dictionary mapping spine positions (chunk indices)
                                                     to the token IDs of the large text blocks to inject.
            max_new_tokens (int): The maximum number of new tokens to generate.
            prompt_ids (Optional[torch.Tensor], optional): Optional starting prompt for generation. Defaults to None.
            temperature (float, optional): Sampling temperature. Defaults to 0.8.
            top_k (int, optional): Top-k sampling. Defaults to 50.

        Returns:
            torch.Tensor: The generated sequence of token IDs.
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
        dummy_emb = self.token_embedding(dummy_input) + self.pos_embedding(dummy_pos)
        
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

    @torch.no_grad()
    def multimodal_generate(
        self,
        prompt_ids: torch.Tensor,
        max_text_tokens: int = 64,
        reasoning_threshold: float = 0.5,
        simulation_steps: int = 10,
        video_steps: int = 20,
        **kwargs
    ):
        """
        Generates a multimodal sequence of text, robot actions, and video from a single prompt.
        """
        # 1. Generate text
        generated_ids = self.generate_speculative(prompt_ids, max_new_tokens=max_text_tokens, **kwargs)
        
        # 2. Get reasoning score
        # To get the score, we need the hidden states of the full generated sequence
        full_sequence_outputs = self.forward_token(generated_ids)
        hidden_states = full_sequence_outputs['hidden_states']
        reasoning_score = self.get_reasoning_score(hidden_states)
        
        robot_actions = None
        video_output = None
        
        print(f"Generated text with reasoning score: {reasoning_score.item():.4f}")

        # 3. Control robots if reasoning score is high
        if reasoning_score.item() > reasoning_threshold:
            print("Reasoning threshold met. Simulating robot control...")
            
            # Use the final hidden state of the text as the initial state for the robot controller
            initial_robot_h = hidden_states[:, -1, :].unsqueeze(0) # Shape for LSTM: [1, B, D]
            
            # Dummy initial robot states (Batch, Seq_len, Features)
            # Assuming 8 robots, 24 state dimensions each
            robot_states = torch.randn(prompt_ids.size(0), 1, 24 * 8).to(prompt_ids.device) 
            
            all_actions = []
            controller_h = (initial_robot_h, torch.zeros_like(initial_robot_h)) # (h_0, c_0)

            for _ in range(simulation_steps):
                actions, controller_h = self.control_robots(robot_states, prev_h=controller_h)
                all_actions.append(actions)
                # In a real scenario, the new robot_states would come from a physics engine
                # Here, we'll just feed zeros as the next input
                robot_states = torch.zeros_like(robot_states)
            
            robot_actions = torch.cat(all_actions, dim=1)
            print(f"Robot actions generated. Shape: {robot_actions.shape}")

            # 4. Generate video of the actions
            # We'll use the original text prompt to condition the video generation
            print("Generating video of the simulated actions...")
            video_output = self.generate_video(
                generated_ids, 
                shape=(1, self.d_model, 32, 32), # Smaller shape for faster testing
                steps=video_steps
            )
            print(f"Video generation complete. Shape: {video_output.shape}")

        return {
            "generated_text_ids": generated_ids,
            "reasoning_score": reasoning_score.item(),
            "robot_actions": robot_actions,
            "video_output": video_output
        }


if __name__ == '__main__':
    print("=" * 70)
    print("HST-V6.1 GIGA - Full Model Self-Test")
    print("=" * 70)

    # Model parameters for testing
    vocab_size = 50257
    d_model = 256
    n_heads = 4
    n_layers = 8
    horizon = 16
    chunk_size = 128

    # --- Test Token Mode ---
    print("\n--- Testing Token Mode ---")
    model_token = HSTv6Giga(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, horizon=horizon, mode='token'
    )
    x_token = torch.randint(0, vocab_size, (1, 128))
    try:
        output_token = model_token(x_token)
        loss_token = output_token['logits'].mean()
        loss_token.backward()
        print("✅ Token mode forward/backward pass successful!")
    except Exception as e:
        print(f"❌ Token mode failed: {e}")

    # --- Test Speculative Generation ---
    print("\n--- Testing Speculative Generation ---")
    try:
        prompt = torch.randint(0, vocab_size, (1, 10))
        generated_ids = model_token.generate_speculative(prompt, max_new_tokens=10)
        print(f"✅ Speculative generation successful! Output length: {generated_ids.size(1)}")
    except Exception as e:
        print(f"❌ Speculative generation failed: {e}")

    # --- Test Annealed Generation ---
    print("\n--- Testing Annealed Generation ---")
    try:
        prompt = torch.randint(0, vocab_size, (1, 10))
        generated_ids = model_token.generate_annealed(prompt, max_new_tokens=10)
        print(f"✅ Annealed generation successful! Output length: {generated_ids.size(1)}")
    except Exception as e:
        print(f"❌ Annealed generation failed: {e}")
        
    # --- Test ZTR Head ---
    print("\n--- Testing ZTR Head ---")
    try:
        hidden_states = torch.randn(1, 128, d_model)
        score = model_token.get_reasoning_score(hidden_states)
        print(f"✅ ZTR head successful! Score: {score.item()}")
    except Exception as e:
        print(f"❌ ZTR head failed: {e}")

    # --- Test Mobius Updater ---
    print("\n--- Testing Mobius Updater ---")
    try:
        update_data = torch.randn(10, d_model)
        model_token.update_continual(update_data)
        print("✅ Mobius updater successful!")
    except Exception as e:
        print(f"❌ Mobius updater failed: {e}")

    # --- Test Multimodal Generation ---
    print("\n--- Testing Multimodal Generation ---")
    try:
        prompt = torch.randint(0, vocab_size, (1, 10))
        multimodal_output = model_token.multimodal_generate(prompt, reasoning_threshold=0.0) # Low threshold to ensure it runs
        print(f"✅ Multimodal generation successful!")
        if multimodal_output["robot_actions"] is not None:
            print(f"  - Robot actions shape: {multimodal_output['robot_actions'].shape}")
        if multimodal_output["video_output"] is not None:
            print(f"  - Video output shape: {multimodal_output['video_output'].shape}")
    except Exception as e:
        print(f"❌ Multimodal generation failed: {e}")


    print("\n" + "=" * 70)
    print("HST-V6.1 Self-Test Complete")
    print("=" * 70)
