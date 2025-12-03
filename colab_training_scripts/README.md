# HST (Harmonic Sequence Transformer) Colab Training Scripts

## Overview
Training scripts for all HST model architecture variants, optimized for Google Colab T4 GPU (16GB VRAM).

## T4 GPU Optimization Settings
All scripts are pre-configured with these memory-safe settings:
- **d_model**: 768 (reduced from original)
- **n_layers**: 16 (reduced from original)
- **n_heads**: 12
- **max_seq_len**: 512
- **batch_size**: 1 (with gradient accumulation)
- **gradient_accumulation_steps**: 16 (effective batch size: 16)
- **Mixed Precision**: FP16 with automatic scaling
- **8-bit Optimizer**: AdamW 8-bit when bitsandbytes available
- **Memory Config**: `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:32`

## Available Scripts

### 1. HST_v3_Ultra_Colab_Training.py
**Features:**
- Complete Lattice Core (Multi-Level + Path-Weighted Fusion)
- Adaptive Block with early exit confidence prediction
- Harmonic Horizon Predictor (multi-step prediction)
- Full lattice structure analyzer

### 2. HST_v4_Unified_Colab_Training.py
**Features:**
- Token and Chunk mode support
- KV Cache for efficient generation
- Unified forward pass architecture
- ChunkEncoder/ChunkDecoder for chunk-level processing

### 3. HST_v4_1_HyperLattice_Colab_Training.py
**Features:**
- Dynamic Hyper-Lattice Block with learnable lattice weights
- Paged KV Cache for memory-efficient inference
- Self-Attention with Paged Cache support
- Dynamic lattice path routing

### 4. HST_v5_Colab_Training.py
**Features:**
- Recursive Descent Lattice Analyzer with layer-aware predictive fields
- Multi-scale Recursive Horizon Predictor (coarse → medium → fine)
- Adaptive Lattice Processor with task routing
- Enhanced early exit mechanism

### 5. HST_v6_Colab_Training.py
**Features:**
- ChunkDecoderWithCache for efficient incremental generation
- TransformerDecoderLayerWithCache with cross-attention caching
- Full cache support for both self and cross attention
- Speculative verification support

### 6. HST_v7_1_2_Agile_Colab_Training.py
**Features:**
- Adaptive Bottom Transformer with dynamic depth prediction
- Speculative Verifier for draft verification
- Recursive Horizon Predictor with lattice embeddings
- Enhanced chunk mode with context injection support

### 7. HST_v8_2_Crystalline_Colab_Training.py
**Features:**
- **Diamond Mixer**: Lossless logic FFN replacement (split → synthesize/analyze → merge)
- **Feedback Loop**: Self-correction with GRU-based iterative refinement
- **Hyperbolic Embeddings**: Hierarchical representation in Poincaré ball
- **Hebbian Fast Weights**: Plasticity layer that learns during inference
- **Hyper-Lattice Block**: Dynamic lattice path routing
- Streamlined Horizon Predictor with uncertainty estimation

### 8. HST_ChaosLogic_Colab_Training.py
**Features:**
- Compressed KV Cache with sparse attention
- Task Analyzer with adaptive depth prediction
- Pattern Selector with Gumbel-softmax routing
- Adaptive Bottom Transformer
- Multi-scale Recursive Horizon Predictor

## Usage

### 1. Upload to Colab
Upload the desired training script to Google Colab.

### 2. Install Dependencies
Run the installation cell at the top of each script:
```python
!pip install torch transformers datasets bitsandbytes accelerate -q
```

### 3. Mount Google Drive (Optional)
For saving checkpoints to Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Run Training
Execute all cells to start training.

## Checkpoint Management
- Checkpoints are saved every 5000 steps to the configured `save_dir`
- Graceful interrupt: Press Ctrl+C to stop training and save current state
- Resume training by loading the checkpoint and adjusting the starting step

## Memory Tips for T4 GPU
1. **If OOM occurs**: Reduce `D_MODEL` to 512 or `N_LAYERS` to 12
2. **For longer sequences**: Reduce `MAX_SEQ_LEN` to 256
3. **For larger batches**: Increase `GRADIENT_ACCUMULATION_STEPS` instead of `BATCH_SIZE`
4. **Clear cache regularly**: Scripts automatically call `torch.cuda.empty_cache()` every 50 steps

## Dataset
All scripts use `HuggingFaceFW/fineweb-edu` (sample-10BT) by default. Modify the `load_dataset` call to use a different dataset.

## Customization
Key hyperparameters at the top of each script:
```python
D_MODEL = 768              # Model dimension
N_HEADS = 12               # Attention heads
N_LAYERS = 16              # Transformer layers
MAX_SEQ_LEN = 512          # Maximum sequence length
HORIZON = 8                # Horizon prediction steps
VOCAB_SIZE = 32000         # Vocabulary size
BATCH_SIZE = 1             # Batch size per step
GRADIENT_ACCUMULATION_STEPS = 16  # Gradient accumulation
MAX_TRAINING_STEPS = 100000       # Total training steps
INITIAL_LR = 2e-4          # Learning rate
WARMUP_STEPS = 2000        # Warmup steps
```

## Architecture Differences Summary

| Model | Key Innovation |
|-------|---------------|
| v3 Ultra | Complete Lattice Core with meta-fusion |
| v4 Unified | Token/Chunk dual-mode |
| v4.1 | Paged KV Cache + Hyper-Lattice |
| v5 | Layer-aware predictive fields |
| v6 | Cross-attention caching |
| v7.1.2 | Speculative verification |
| v8.2 | Diamond Mixer + Feedback Loop |
| ChaosLogic | Compressed cache + task routing |
