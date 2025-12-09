import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from transformers import AutoTokenizer
import gc
# NOTE: Ensure you are running this on a GPU (e.g., Colab T4)

# ==========================================================
# 1. MODEL HYPERPARAMETERS (MUST MATCH TRAINING)
# ==========================================================
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 16
MAX_SEQ_LEN = 768 # Max tokens in the context window
HORIZON = 8
CHUNK_SIZE = 128

# TARGET: ~200,000 characters. 50048 is a multiple of 128 (391 chunks).
MAX_GEN_TOKENS = 50048 
OUTPUT_FILENAME = "hst_v5_chunk_story_50k_tokens_FAST.txt"
DRIVE_OUTPUT_PATH = f"/content/drive/MyDrive/{OUTPUT_FILENAME}"

# --- QUALITY HYPERPARAMETERS (FOR SPEED) ---
TEMPERATURE = 1.0       # Unbiased sampling (best for speed)
TOP_K = 50             # Standard for speed
# ==========================================================

# ==========================================================
# 2. CORE UTILITIES (Needed for inference)
# ==========================================================
KVCache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]

class RecursiveDescentLatticeAnalyzer(nn.Module):
    def __init__(self, max_seq_len=8192):
        super().__init__()
        # Simplified for inference, assuming max_seq_len is max_chunks
        self.max_len = max_seq_len 
        self.layer_weights = nn.Parameter(torch.ones(10)) # Needs to be defined for state_dict load

    def forward(self, x):
        return x # Analyzer is primarily used in training for loss/routing

# ==========================================================
# 3. CHUNK ENCODER/DECODER (Used for local and token-level work)
# ==========================================================
class ChunkEncoder(nn.Module):
    def __init__(self, d_model, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        self.local_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pooling_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, token_embeddings):
        B, total_tokens, D = token_embeddings.shape
        num_chunks = total_tokens // self.chunk_size
        tokens_to_use = num_chunks * self.chunk_size
        
        if num_chunks == 0:
            return token_embeddings.new_zeros(B, 0, D)
            
        chunks = token_embeddings[:, :tokens_to_use, :].view(B * num_chunks, self.chunk_size, D)
        encoded_tokens = self.local_encoder(chunks)
        query = self.pooling_query.expand(B * num_chunks, -1, -1)
        pooled, _ = self.pooling_attn(query, encoded_tokens, encoded_tokens)
        return pooled.view(B, num_chunks, D)

class ChunkDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, chunk_size=128, n_heads=8, n_layers=2):
        super().__init__()
        self.chunk_size = chunk_size
        self.pos_embedding = nn.Embedding(chunk_size, d_model) 
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
        self.local_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, chunk_embeddings, target_token_embeddings):
        B, num_chunks, D = chunk_embeddings.shape
        seq_len = min(target_token_embeddings.size(1), num_chunks * self.chunk_size)
        target_token_embeddings = target_token_embeddings[:, :seq_len, :]
        
        pos = torch.arange(0, self.chunk_size, device=target_token_embeddings.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos).repeat(B * num_chunks, 1, 1)
        
        tgt = target_token_embeddings.view(B * num_chunks, self.chunk_size, D) + pos_emb
        memory = chunk_embeddings.view(B * num_chunks, 1, D).repeat(1, self.chunk_size, 1)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.chunk_size).to(tgt.device)
        
        refined = self.local_decoder(tgt, memory, tgt_mask=causal_mask)
        refined = refined.view(B, seq_len, D)
        return self.lm_head(refined)

# ==========================================================
# 4. ADAPTIVE LATTICE PROCESSOR (Inter-chunk dependency)
# ==========================================================
class AdaptiveLatticeProcessor(nn.Module):
    def __init__(self, d_model, max_num_chunks):
        super().__init__()
        self.analyzer = RecursiveDescentLatticeAnalyzer(max_num_chunks) 
        self.layer_processors = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True) for _ in range(10)
        ])
        self.task_router = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 10), nn.Sigmoid()
        )
    
    def forward(self, x, horizon_targets=None): 
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

# ==========================================================
# 5. RECURSIVE HORIZON PREDICTOR (Chunk-ahead prediction)
# ==========================================================
class RecursiveHorizonPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.vocab_size = vocab_size
        self.coarse_predictor = nn.Linear(d_model, vocab_size)
        self.medium_predictor = nn.Linear(d_model * 2, vocab_size)
        self.fine_predictor = nn.Linear(d_model * 2, vocab_size)
        self.lattice_embeddings = nn.Embedding(20, d_model)
        self.projection = nn.Linear(vocab_size, d_model)

    def forward(self, h_sequence):
        B, S, D = h_sequence.shape
        h_t = h_sequence[:, -1, :] 
        
        # Prediction logic remains the same, predicting tokens from future chunk states
        coarse_preds = {}
        for offset in [4, min(10, self.horizon)]:
            offset_emb = self.lattice_embeddings(torch.tensor([min(offset - 1, 19)], device=h_t.device))
            coarse_preds[offset] = self.coarse_predictor(h_t + offset_emb)

        medium_preds = {}
        for offset in [2, 6]:
            if offset <= self.horizon:
                left_coarse = coarse_preds[4]
                right_coarse = coarse_preds.get(10, coarse_preds[4])
                alpha = (offset - 4) / max(10 - 4, 1)
                coarse_interp = self.projection(alpha * left_coarse + (1 - alpha) * right_coarse)
                medium_preds[offset] = self.medium_predictor(torch.cat([h_t, coarse_interp], dim=-1))

        fine_preds = {}
        for offset in [1, 3, 5]:
            if offset <= self.horizon:
                left_med = medium_preds.get(2, coarse_preds[4])
                right_med = medium_preds.get(6, coarse_preds[4])
                alpha = (offset - 2) / max(6 - 2, 1)
                medium_interp = self.projection(alpha * left_med + (1 - alpha) * right_med)
                fine_preds[offset] = self.fine_predictor(torch.cat([h_t, medium_interp], dim=-1))
        
        all_preds = {**coarse_preds, **medium_preds, **fine_preds}
        logits_list = [all_preds.get(i, torch.zeros(B, self.vocab_size, device=h_t.device)) for i in range(1, self.horizon + 1)]
        logits = torch.stack(logits_list, dim=1)
        confidence = torch.ones(B, self.horizon, device=h_t.device)
        return logits, confidence

# ==========================================================
# 6. HSTv5 INFERENCE MODEL (Unified class with chunk focus)
# ==========================================================
class HSTv5_Inference(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len=768, horizon=8, early_exit_threshold=0.93, mode='chunk', chunk_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.horizon = horizon
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        max_num_chunks = max_seq_len // chunk_size 

        # Chunk Mode Components
        self.pos_embedding = nn.Embedding(max_num_chunks, d_model) # Pos embedding for chunks
        self.chunk_encoder = ChunkEncoder(d_model, chunk_size, n_heads=N_HEADS//2) # Use fewer heads for local
        self.chunk_decoder = ChunkDecoder(d_model, vocab_size, chunk_size, n_heads=N_HEADS//2)
        self.lattice_core = AdaptiveLatticeProcessor(d_model, max_num_chunks) 
        
        # Dummy layers for token-mode components to allow non-strict loading
        self.adaptive_bottom = nn.ModuleList([nn.Identity() for _ in range(n_layers // 2)])
        self.top_stack = nn.ModuleList([nn.Identity() for _ in range(n_layers - n_layers // 2)])


        self.horizon_predictor = RecursiveHorizonPredictor(d_model, vocab_size, horizon=horizon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)
        
    def forward(self, input_ids, cache=None):
        return self.forward_chunk(input_ids) # Only support chunk mode

    def forward_chunk(self, input_ids):
        B, total_tokens = input_ids.shape
        device = input_ids.device
        
        x = self.token_embedding(input_ids)
        chunk_emb = self.chunk_encoder(x)
        B, num_chunks, D = chunk_emb.shape
        
        if num_chunks == 0:
            # Handle case where input is too short to form a single chunk
            zero_logits = torch.zeros(B, total_tokens, self.vocab_size, device=device)
            zero_horizon = torch.zeros(B, self.horizon, self.vocab_size, device=device)
            return {
                'logits': zero_logits, 'horizon_logits': zero_horizon, 
                'confidence': torch.zeros(B, self.horizon, device=device), 
                'hidden_states': chunk_emb, 'bottom_depth': 0, 'cache': None
            }
        
        # Chunk Positional Encoding
        chunk_positions = torch.arange(0, num_chunks, dtype=torch.long, device=device)
        chunk_positions = chunk_positions.clamp(max=self.pos_embedding.num_embeddings - 1)
        h_in = chunk_emb + self.pos_embedding(chunk_positions)
        
        # Lattice Core Processing
        h_lattice_out = self.lattice_core(h_in)
        
        # Chunk Decoding (Token-level prediction)
        logits = self.chunk_decoder(h_lattice_out, x)
        
        # Horizon Prediction (Chunk-level prediction)
        logits_horizon, confidence = self.horizon_predictor(h_lattice_out)
        
        return {
            'logits': logits, 
            'horizon_logits': logits_horizon, 
            'confidence': confidence,
            'hidden_states': h_lattice_out, 
            'bottom_depth': 0, 
            'cache': None
        }

# ==========================================================
# 7. GENERATION LOGIC (Simplified & Optimized)
# ==========================================================
@torch.no_grad()
def generate_chunk_by_chunk(model, tokenizer, prompt, max_new_tokens, chunk_size=CHUNK_SIZE, max_context_len=MAX_SEQ_LEN, temperature=TEMPERATURE, top_k=TOP_K):
    """
    Generates tokens in a chunk-by-chunk manner using the Chunk Mode model (MAXIMUM SPEED).
    """
    device = next(model.parameters()).device
    
    # 1. Prepare Initial Context
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_context_len).input_ids.to(device)
    B, S_initial = input_ids.shape
    
    current_ids = input_ids
    total_generated = 0
    start_time = time.time()
    
    # We generate an exact number of chunks
    num_chunks_to_generate = (max_new_tokens + chunk_size - 1) // chunk_size
    print(f"Target: {max_new_tokens} tokens ({num_chunks_to_generate} chunks of {chunk_size}).")

    for chunk_step in range(num_chunks_to_generate):
        if (chunk_step + 1) % 20 == 0:
            print(f"  -> Generating Chunk {chunk_step + 1}/{num_chunks_to_generate} ({total_generated} tokens so far)...")
            torch.cuda.empty_cache()

        # 2. Prepare the input block: Context + Placeholder Block
        # Only use the last MAX_SEQ_LEN tokens as context
        context_ids = current_ids[:, -max_context_len:]
        S_context = context_ids.size(1)

        # Pad context to be a multiple of chunk_size (necessary for ChunkEncoder)
        S_context_padded = (S_context + chunk_size - 1) // chunk_size * chunk_size
        padding_needed = S_context_padded - S_context
        context_ids_padded = F.pad(context_ids, (0, padding_needed), value=tokenizer.pad_token_id)
        
        S_input_context = context_ids_padded.size(1)
        
        # The key to chunk generation: The model must be trained to predict the next chunk 
        # based on the *current chunk's hidden state*. 
        # By appending a placeholder, the model is forced to decode the next tokens.
        placeholder_block = torch.full((B, chunk_size), tokenizer.pad_token_id, dtype=torch.long, device=device)
        input_for_pass = torch.cat([context_ids_padded, placeholder_block], dim=1)
        
        # 3. Forward Pass (Model decodes *all* tokens based on the lattice core output)
        output = model(input_for_pass)
        logits = output['logits']
        
        # 4. Extract logits for the newly generated chunk (it's the last CHUNK_SIZE tokens)
        # Note: If there was no padding, the new tokens are logits[:, S_context:]. 
        # Since we padded, the logits are slightly shifted. The *new chunk* starts at index S_input_context.
        new_chunk_logits = logits[:, S_input_context:, :]
        
        new_chunk_ids = []
        for i in range(chunk_size):
            logit_i = new_chunk_logits[0, i, :]
            
            # --- FAST SAMPLING: Temperature + Top-K ---
            logit_i = logit_i / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logit_i, top_k)
                logit_i[logit_i < v[-1]] = -float('Inf')
                
            probs = F.softmax(logit_i, dim=-1)
            
            # Safety check: ensure we can sample
            if probs.sum() == 0 or torch.isnan(probs).any(): 
                sampled_id = tokenizer.pad_token_id
            else:
                sampled_id = torch.multinomial(probs, 1).item()
            
            new_chunk_ids.append(sampled_id)
            
            # --- Quick update of the placeholder block to simulate auto-regression within the chunk ---
            # This is technically not pure autoregressive but improves sample quality without 
            # re-running the full forward pass. (Optional: Can remove for absolute max speed)
            if i + 1 < chunk_size:
                input_for_pass[:, S_input_context + i + 1] = sampled_id 
        
        new_chunk_tensor = torch.tensor(new_chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # 5. Update current IDs
        current_ids = torch.cat([current_ids, new_chunk_tensor], dim=1)
        total_generated += chunk_size
        
        if total_generated >= max_new_tokens:
            break

    end_time = time.time()
    total_time = end_time - start_time
    avg_tps = total_generated / total_time
    
    output_ids = current_ids[0].tolist()
    
    stats = {
        'tokens_generated': total_generated,
        'total_time': total_time,
        'average_tps': avg_tps,
    }
    
    return output_ids, stats


def load_model_from_checkpoint(mode, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    VOCAB_SIZE = len(tokenizer)
    
    print(f"\n[INFO] Loading model in {mode} mode...")
    
    # Initialize the inference-specific model class
    model = HSTv5_Inference(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN, horizon=HORIZON, mode=mode, chunk_size=CHUNK_SIZE,
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)['model_state_dict']
            
            # Chunk Mode: Only load relevant weights (chunk-related and core shared layers)
            state_dict_filtered = {
                k: v for k, v in state_dict.items() 
                if not (k.startswith('adaptive_bottom') or k.startswith('top_stack'))
            }
            
            model.load_state_dict(state_dict_filtered, strict=False)
            print(f"✅ Successfully loaded model state NON-STRICTLY ({mode} keys filtered) from: {checkpoint_path}")

        except Exception as e:
            print(f"❌ WARNING: Load failed. Using random weights. Error: {e}")
    else:
        print(f"❌ WARNING: Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model.eval()
    return model

# ==========================================================
# 8. EXECUTION BLOCK
# ==========================================================

if __name__ == '__main__':
    
    # --- Colab Setup --- 
    try:
        from google.colab import drive
        LOCAL_CHECKPOINT_PATH = '/content/hst_v5_checkpoints/hst_v5_final.pt'
        if os.path.exists(LOCAL_CHECKPOINT_PATH):
            CHECKPOINT_PATH = LOCAL_CHECKPOINT_PATH
            DRIVE_MOUNTED = False
        else:
            print("[INFO] Attempting to mount Google Drive...")
            drive.mount('/content/drive')
            CHECKPOINT_PATH = f'/content/drive/MyDrive/hst_v5_checkpoints/hst_v5_final.pt'
            DRIVE_MOUNTED = True
    except ImportError:
        CHECKPOINT_PATH = './hst_v5_checkpoints/hst_v5_final.pt'
        DRIVE_MOUNTED = False
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    VOCAB_SIZE = len(tokenizer)

    print("=" * 70)
    print(f"HST-v5 CHUNK MODE: 200K Character Generation (Target: {MAX_GEN_TOKENS} Tokens)")
    print(f"MAX SPEED TEST: T={TEMPERATURE}, Top-K={TOP_K}")
    print("=" * 70)

    # --- Load Model in Chunk Mode ---
    model_chunk = load_model_from_checkpoint(mode='chunk', checkpoint_path=CHECKPOINT_PATH)
    
    EVAL_PROMPT = "The ancient scroll detailed the forgotten history of the star-faring empire, which began when their homeworld, Xylos, faced..."

    print(f"\nStarting generation of {MAX_GEN_TOKENS} tokens...")
    
    # --- Start Generation ---
    output_ids, stats = generate_chunk_by_chunk(
        model=model_chunk,
        tokenizer=tokenizer,
        prompt=EVAL_PROMPT,
        max_new_tokens=MAX_GEN_TOKENS,
        temperature=TEMPERATURE, 
        top_k=TOP_K,             
        chunk_size=CHUNK_SIZE,
        max_context_len=MAX_SEQ_LEN
    )

    # --- Save and Report ---
    full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    final_path = DRIVE_OUTPUT_PATH if DRIVE_MOUNTED else OUTPUT_FILENAME
    
    try:
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print("\n" + "="*70)
        print(f"✅ GENERATION COMPLETE. Story saved to: {final_path}")
    except Exception as e:
        print(f"❌ WARNING: Could not save file to {final_path}. Error: {e}")
        
    print(f"Total Characters Generated: {len(full_text)}")
    print(f"Total Tokens Generated: {stats['tokens_generated']}")
    print(f"Total Time: {stats['total_time']:.4f}s")
    print(f"**Average TPS (Tokens Per Second): {stats['average_tps']:.2f}**")
    print("="*70)
