import torch
import torch.nn as nn
import torch.nn.functional as F

class ChunkEncoder(nn.Module):
    """Encodes a chunk of tokens into a single vector representation."""
    def __init__(self, d_model, chunk_size=128):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model
        
        # Local transformer for within-chunk processing
        self.local_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=d_model*4),
            num_layers=2
        )
        
        # Compress chunk to single representation
        self.chunk_pooling = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, token_embeddings):
        """
        Args:
            token_embeddings: [batch, num_chunks * chunk_size, d_model]
        Returns:
            chunk_embeddings: [batch, num_chunks, d_model]
        """
        B, total_tokens, D = token_embeddings.shape
        num_chunks = total_tokens // self.chunk_size
        
        # Reshape into chunks
        chunks = token_embeddings[:, :num_chunks * self.chunk_size, :]
        chunks = chunks.view(B, num_chunks, self.chunk_size, D)
        
        # Process each chunk independently (can be parallelized)
        chunk_reprs = []
        for i in range(num_chunks):
            chunk_tokens = chunks[:, i, :, :]  # [B, chunk_size, D]
            
            # Local attention within chunk
            encoded = self.local_encoder(chunk_tokens)
            
            # Pool to single vector (mean + learned compression)
            pooled = encoded.mean(dim=1)  # [B, D]
            compressed = self.chunk_pooling(pooled)
            
            chunk_reprs.append(compressed)
        
        # Stack all chunks
        chunk_embeddings = torch.stack(chunk_reprs, dim=1)  # [B, num_chunks, D]
        
        return chunk_embeddings


class ChunkDecoder(nn.Module):
    """Decodes chunk representation back to token-level predictions."""
    def __init__(self, d_model, vocab_size, chunk_size=128):
        super().__init__()
        self.chunk_size = chunk_size
        
        # Expand chunk vector to chunk_size tokens
        self.chunk_expander = nn.Linear(d_model, d_model * chunk_size)
        
        # Local refinement transformer
        self.local_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8, dim_feedforward=d_model*4),
            num_layers=2
        )
        
        # Token prediction head
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, chunk_embeddings, chunk_idx=None):
        """
        Args:
            chunk_embeddings: [batch, num_chunks, d_model]
            chunk_idx: Optional specific chunk to decode (for autoregressive)
        Returns:
            token_logits: [batch, chunk_size, vocab_size] if chunk_idx given
                         [batch, num_chunks * chunk_size, vocab_size] otherwise
        """
        B, num_chunks, D = chunk_embeddings.shape
        
        if chunk_idx is not None:
            # Decode single chunk (autoregressive generation)
            chunk = chunk_embeddings[:, chunk_idx:chunk_idx+1, :]  # [B, 1, D]
            expanded = self.chunk_expander(chunk)  # [B, 1, D*chunk_size]
            expanded = expanded.view(B, self.chunk_size, D)
            
            # Refine with local attention
            refined = self.local_decoder(expanded, expanded)
            logits = self.lm_head(refined)
            return logits
        else:
            # Decode all chunks (training)
            expanded = self.chunk_expander(chunk_embeddings)  # [B, num_chunks, D*chunk_size]
            expanded = expanded.view(B, num_chunks * self.chunk_size, D)
            
            refined = self.local_decoder(expanded, expanded)
            logits = self.lm_head(refined)
            return logits


class ChunkBasedHST(nn.Module):
    """
    Harmonic Spine Transformer operating on chunks instead of tokens.
    
    Architecture:
    1. Chunk Encoding: Tokens → Chunks (local compression)
    2. Lattice Processing: Chunks processed via HST lattice
    3. Chunk Decoding: Chunks → Tokens (local expansion)
    """
    def __init__(self, vocab_size, d_model=256, chunk_size=128, max_chunks=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token-level embeddings (still needed for encoding)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(chunk_size * max_chunks, d_model)
        
        # Chunk encoder/decoder
        self.chunk_encoder = ChunkEncoder(d_model, chunk_size)
        self.chunk_decoder = ChunkDecoder(d_model, vocab_size, chunk_size)
        
        # Core HST lattice (UNCHANGED - just operates on chunks now!)
        self.lattice_spine = self._generate_spine(max_chunks)
        
        # Simplified lattice for demo (use your full CompleteLatticeCore here)
        self.lattice_processor = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*4)
            for _ in range(4)
        ])
        
        self.lattice_injection = nn.Linear(d_model * 2, d_model)
        
    def _generate_spine(self, max_len):
        """Generate lattice spine: 0, 2, 4, 10, 24, 58, ..."""
        spine = [0, 2, 4]
        while spine[-1] < max_len:
            next_val = 2*spine[-1] + 2*spine[-2] + 2*spine[-3]
            if next_val >= max_len:
                break
            spine.append(next_val)
        return spine
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch, num_tokens] - raw token IDs
        Returns:
            logits: [batch, num_tokens, vocab_size]
        """
        B, total_tokens = input_ids.shape
        
        # 1. Token embeddings
        positions = torch.arange(total_tokens, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        
        # 2. Encode tokens → chunks
        chunk_emb = self.chunk_encoder(x)  # [B, num_chunks, D]
        num_chunks = chunk_emb.size(1)
        
        # 3. Process chunks through lattice (same algorithm, different granularity!)
        h_lattice = chunk_emb.clone()
        
        # Apply lattice processing at spine positions
        for pos in self.lattice_spine:
            if pos >= num_chunks:
                break
            
            # Gather neighborhood (simplified - use your full topology)
            neighborhood_indices = []
            if pos > 0:
                neighborhood_indices.append(pos - 1)
            neighborhood_indices.append(pos)
            if pos < num_chunks - 1:
                neighborhood_indices.append(pos + 1)
            
            # Aggregate neighborhood
            neighborhood = torch.stack([
                chunk_emb[:, idx, :] for idx in neighborhood_indices
            ], dim=1)
            
            aggregated = neighborhood.mean(dim=1)
            
            # Process through lattice layers
            for layer in self.lattice_processor:
                aggregated = layer(aggregated.unsqueeze(1)).squeeze(1)
            
            # Inject back
            combined = torch.cat([h_lattice[:, pos, :], aggregated], dim=-1)
            h_lattice[:, pos, :] = self.lattice_injection(combined)
        
        # 4. Decode chunks → tokens
        logits = self.chunk_decoder(h_lattice)  # [B, num_tokens, vocab_size]
        
        return logits


# ============================================================================
# TRAINING EXAMPLE
# ============================================================================
def train_chunk_hst():
    """Demonstrates chunk-based training."""
    model = ChunkBasedHST(
        vocab_size=50257,
        d_model=256,
        chunk_size=128,  # Each "position" = 128 tokens
        max_chunks=512   # 512 positions = 65,536 token context!
    )
    
    # Example: 8192 token sequence
    batch_size = 2
    seq_len = 8192
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    # Forward pass
    logits = model(input_ids)  # [2, 8192, 50257]
    
    # Loss (same as token-level!)
    labels = input_ids
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 50257),
        labels[:, 1:].reshape(-1)
    )
    
    print(f"✅ Chunk-based HST training:")
    print(f"   Input: {seq_len} tokens = {seq_len // 128} chunks")
    print(f"   Lattice operates on {seq_len // 128} positions (not {seq_len}!)")
    print(f"   Loss: {loss.item():.4f}")
    
    return model


if __name__ == '__main__':
    train_chunk_hst()
    print("\n🎯 Key Insight:")
    print("The SAME lattice structure (0,2,4,10,24,58...)")
    print("now processes CHUNKS instead of tokens.")
    print("Position 24 = 24th chunk = tokens 3072-3200")
    print("Position 58 = 58th chunk = tokens 7424-7552")
    print("\nThis gives you MASSIVE context with O(log N) lattice efficiency!")
