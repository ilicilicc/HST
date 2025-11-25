# HST-v4 Unified: Token and Chunk-Based Harmonic Spine Transformer

This repository contains the implementation of the HST-v4 Unified model, an advanced and upgraded version of the Harmonic Spine Transformer. This model introduces a novel chunk-based processing mode, allowing it to operate on sequences of token chunks rather than individual tokens, which enables the processing of much larger contexts with greater efficiency.

## Key Features

- **Unified Architecture**: The `HSTv4Unified` model seamlessly integrates both token-based and chunk-based processing, which can be selected with a simple configuration setting.
- **Chunk-Based Processing**: The new chunk mode allows the model to handle massive contexts by treating sequences of tokens as single units, reducing the effective sequence length and enabling O(log N) efficiency over chunks.
- **Advanced Lattice Core**: The model retains the powerful `CompleteLatticeCore`, which uses a multi-level, path-weighted approach to process information hierarchically.
- **Unified Training Script**: The `train_unified.py` script is a single, unified training solution that can train both token and chunk-based models.

## How to Use

### Model Configuration

The `HSTv4Unified` model can be configured to operate in one of two modes:

- **`token`**: The default mode, which processes individual tokens.
- **`chunk`**: The new mode, which processes sequences of token chunks.

To select a mode, set the `mode` parameter in the model's configuration. When using `chunk` mode, you can also specify the `chunk_size`.

```python
# Token mode
model_token = HSTv4Unified(
    # ... other parameters
    mode='token'
)

# Chunk mode
model_chunk = HSTv4Unified(
    # ... other parameters
    mode='chunk',
    chunk_size=128
)
```

### Training

The unified training script, `train_unified.py`, can be used to train both token and chunk-based models. To switch between modes, modify the `mode` parameter in the `config` dictionary within the script's `main` function.

```python
# In train_unified.py
config['mode'] = 'token'  # or 'chunk'
```

Then, run the script to begin training:

```bash
python train_unified.py
```

The script will automatically handle the appropriate loss calculation and validation for the selected mode.
