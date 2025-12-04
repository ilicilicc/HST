import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from transformers import AutoTokenizer

# ==========================================================
# 1. MODEL HYPERPARAMETERS (MUST MATCH TRAINING)
# ==========================================================
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 16
MAX_SEQ_LEN = 768
HORIZON = 8
CHUNK_SIZE = 128

# TARGET: ~200,000 characters. 50048 is a multiple of 128 (391 chunks).
MAX_GEN_TOKENS = 50048 
OUTPUT_FILENAME = "hst_v5_chunk_story_50k_tokens_FAST.txt"
DRIVE_OUTPUT_PATH = f"/content/drive/MyDrive/{OUTPUT_FILENAME}"

# --- QUALITY HYPERPARAMETERS (FOR SPEED) ---
TEMPERATURE = 1.0        # Unbiased sampling (best for speed)
TOP_K = 50               # Standard for speed
# ==========================================================

# ... (All class definitions: ChunkEncoder, ChunkDecoder, etc. must be here) ...

# WARNING: Assuming all necessary classes (HSTv5_Inference, ChunkEncoder, etc.) 
# are correctly defined above this point in the final script.

# Dummy class to avoid a NameError in this block's syntax check, REMOVE ME
class HSTv5_Inference(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len=768, horizon=8, early_exit_threshold=0.93, mode='token', chunk_size=128):
        super().__init__()
        self.mode = mode; self.chunk_size = chunk_size; self.horizon = horizon; self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model); self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.chunk_encoder = ChunkEncoder(d_model, chunk_size, n_layers=2); self.chunk_decoder = ChunkDecoder(d_model, vocab_size, chunk_size, n_layers=2)
        self.lattice_core = AdaptiveLatticeProcessor(d_model, max_seq_len); self.adaptive_bottom = nn.ModuleList([AdaptiveBlock(d_model, n_heads) for _ in range(n_layers // 2)])
        self.top_stack = nn.ModuleList([TransformerEncoderLayerWithCache(d_model, n_heads) for _ in range(n_layers - n_layers // 2)]); self.horizon_predictor = RecursiveHorizonPredictor(d_model, vocab_size, horizon=horizon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False); self.ln_f = nn.LayerNorm(d_model)
    def forward(self, input_ids, cache=None):
        if self.mode == 'token': return self.forward_token(input_ids, cache)
        else: return self.forward_chunk(input_ids)
    def forward_token(self, input_ids, cache=None):
        B, seq_len = input_ids.shape; device = input_ids.device; past_len = 0
        if cache and cache[0] and cache[0][0] is not None: past_len = cache[0][0].size(2)
        positions = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device).clamp(max=self.pos_embedding.num_embeddings - 1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions); new_cache = []; cache_idx = 0; predicted_depth = self.n_bottom_layers
        for i, block in enumerate(self.adaptive_bottom):
            layer_past = cache[cache_idx] if cache and cache_idx < len(cache) else None; x, conf, present = block(x, layer_past); new_cache.append(present); cache_idx += 1
            conf_val = conf.mean().item() if x.size(1) > 1 else conf.item()
            if past_len == 0 and i >= 1 and conf_val > self.early_exit_threshold: predicted_depth = i + 1; break
        h_lattice_out = self.lattice_core(x); h_top_in = h_lattice_out
        for i, block in enumerate(self.top_stack):
            layer_past = cache[cache_idx] if cache and cache_idx < len(cache) else None; h_top_in, present = block(h_top_in, layer_past); new_cache.append(present); cache_idx += 1
        h_final = h_top_in; logits_t1 = self.lm_head(self.ln_f(h_final)); logits_horizon, confidence = self.horizon_predictor(h_final[:, -1:, :])
        if logits_horizon.dim() == 3: logits_horizon = logits_horizon.squeeze(1);
        if confidence.dim() == 2: confidence = confidence.squeeze(1)
        return {'logits': logits_t1, 'horizon_logits': logits_horizon, 'confidence': confidence, 'hidden_states': h_final, 'bottom_depth': predicted_depth, 'cache': new_cache}
    def forward_chunk(self, input_ids):
        B, total_tokens = input_ids.shape; device = input_ids.device
        positions = torch.arange(0, total_tokens, dtype=torch.long, device=device).clamp(max=self.pos_embedding.num_embeddings - 1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        chunk_emb = self.chunk_encoder(x); h_lattice_out = self.lattice_core(chunk_emb)
        logits = self.chunk_decoder(h_lattice_out, x); logits_horizon, confidence = self.horizon_predictor(h_lattice_out)
        return {'logits': logits, 'horizon_logits': logits_horizon, 'confidence': confidence, 'hidden_states': h_lattice_out, 'bottom_depth': 0, 'cache': None}
# --- (End Placeholder) ---


@torch.no_grad()
def generate_chunk_by_chunk(model, tokenizer, prompt, max_new_tokens, chunk_size=CHUNK_SIZE, max_context_len=MAX_SEQ_LEN, temperature=TEMPERATURE, top_k=TOP_K):
    """
    Generates tokens in a chunk-by-chunk manner using the Chunk Mode model (MAXIMUM SPEED).
    """
    device = next(model.parameters()).device
    
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_context_len).input_ids.to(device)
    B, S_initial = input_ids.shape
    
    current_ids = input_ids
    total_generated = 0
    start_time = time.time()
    
    num_chunks_to_generate = (max_new_tokens + chunk_size - 1) // chunk_size
    print(f"Target: {max_new_tokens} tokens ({num_chunks_to_generate} chunks of {chunk_size}).")

    for chunk_step in range(num_chunks_to_generate):
        if (chunk_step + 1) % 20 == 0:
             print(f"  -> Generating Chunk {chunk_step + 1}/{num_chunks_to_generate} ({total_generated} tokens so far)...")

        # 2. Prepare the input block
        context_ids = current_ids[:, -max_context_len:]
        S_context = context_ids.size(1)

        S_context_padded = (S_context + chunk_size - 1) // chunk_size * chunk_size
        padding_needed = S_context_padded - S_context
        context_ids_padded = F.pad(context_ids, (0, padding_needed), value=tokenizer.pad_token_id)
        
        S_input = context_ids_padded.size(1)
        
        placeholder_block = torch.full((B, chunk_size), tokenizer.pad_token_id, dtype=torch.long, device=device)
        input_for_pass = torch.cat([context_ids_padded, placeholder_block], dim=1)
        
        # 3. Forward Pass (Predicts logits for the *entire* sequence)
        output = model(input_for_pass)
        logits = output['logits']
        
        # 4. Extract logits for the new chunk and sample
        new_chunk_logits = logits[:, S_input:, :]
        
        new_chunk_ids = []
        for i in range(chunk_size):
            logit_i = new_chunk_logits[0, i, :]
            
            # --- FASTEST SAMPLING: Temperature + Top-K (NO REPETITION PENALTY) ---
            logit_i = logit_i / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logit_i, top_k); logit_i[logit_i < v[-1]] = -float('Inf')
                
            probs = F.softmax(logit_i, dim=-1)
            
            # Safety check: ensure we can sample
            if probs.sum() == 0: 
                sampled_id = tokenizer.pad_token_id
            else:
                sampled_id = torch.multinomial(probs, 1).item()
            new_chunk_ids.append(sampled_id)

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
    # ... (omitted for brevity)
    print(f"\n[INFO] Loading model in {mode} mode...")
    
    model = HSTv5_Inference(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN, horizon=HORIZON, mode=mode, chunk_size=CHUNK_SIZE,
    ).to(device)

    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            if mode == 'token':
                state_dict_filtered = {k: v for k, v in state_dict.items() if not (k.startswith('chunk_encoder') or k.startswith('chunk_decoder'))}
            elif mode == 'chunk':
                state_dict_filtered = {k: v for k, v in state_dict.items() if not (k.startswith('adaptive_bottom') or k.startswith('top_stack') or (k == 'pos_embedding.weight' and v.shape[0] != MAX_SEQ_LEN))}
                
            model.load_state_dict(state_dict_filtered, strict=False)
            print(f"✅ Successfully loaded model state NON-STRICTLY ({mode} keys filtered) from: {checkpoint_path}")

        except Exception as e:
            print(f"❌ WARNING: Load failed even with filtering. Using random weights. Error: {e}")
    else:
         print(f"❌ WARNING: Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model.eval()
    return model

# ==========================================================
# 6. EXECUTION BLOCK
# ==========================================================

if __name__ == '__main__':
    
    # --- Colab Setup --- (re-defined from the working script)
    D_MODEL = 768; N_HEADS = 12; N_LAYERS = 16; MAX_SEQ_LEN = 768; HORIZON = 8; CHUNK_SIZE = 128
    
    try:
        from google.colab import drive
        LOCAL_CHECKPOINT_PATH = '/content/hst_v5_checkpoints/hst_v5_final.pt'
        if os.path.exists(LOCAL_CHECKPOINT_PATH):
            CHECKPOINT_PATH = LOCAL_CHECKPOINT_PATH
            DRIVE_MOUNTED = False
        else:
            drive.mount('/content/drive'); CHECKPOINT_PATH = f'/content/drive/MyDrive/hst_v5_checkpoints/hst_v5_final.pt'
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
    
    if DRIVE_MOUNTED:
        final_path = DRIVE_OUTPUT_PATH
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print("\n" + "="*70)
        print(f"✅ GENERATION COMPLETE. Story saved to: {final_path}")
    else:
        # Local save for non-Colab environments
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"✅ GENERATION COMPLETE. Story saved locally to: {OUTPUT_FILENAME}")
        
    print(f"Total Characters Generated: {len(full_text)}")
    print(f"Total Tokens Generated: {stats['tokens_generated']}")
    print(f"Total Time: {stats['total_time']:.4f}s")
    print(f"**Average TPS (Tokens Per Second): {stats['average_tps']:.2f}**")
    print("="*70)