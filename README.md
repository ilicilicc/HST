# Harmonic Spine Transformer (HST) v6 - Giga

This repository contains the official PyTorch implementation for the Harmonic Spine Transformer (HST) v6 - Giga, a state-of-the-art language model architecture designed for generating long-form, structured content.

## Key Features

*   **Structurally-Aware Context Injection**: A groundbreaking feature that allows you to inject large blocks of text (e.g., chapters, technical documents) at specific, influential positions within the model's hierarchical spine. This enables unprecedented control over the generated output, making it ideal for tasks like writing books, generating code, or compiling reports.
*   **Unified Architecture**: Supports both standard token-by-token processing (`token` mode) and a highly efficient chunk-based processing (`chunk` mode) in a single, unified model.
*   **Complete Lattice Core**: Implements a sophisticated sparse attention mechanism based on a multi-level lattice structure, allowing for efficient long-context processing.
*   **Ultra-Fast Generation**: Includes a speculative decoding algorithm (`generate_ultra_fast`) for significant speedups during inference.
*   **Advanced Training**: Features a `HierarchicalPredictiveLoss` that improves model convergence and prediction accuracy by incorporating future token predictions into the loss calculation.

## How to Run

### 1. Installation

```bash
pip install torch numpy
```

### 2. Training

The main training script is `train_v6_giga.py`. You can configure the model and training parameters directly within the `main()` function of the script.

To train the model in **chunk mode** (recommended for efficiency):

```bash
python train_v6_giga.py
```
*(The default mode in the script is 'chunk')*

To train the model in **token mode**:

1.  Open `train_v6_giga.py`.
2.  Change the `mode` variable to `'token'`.
3.  Run the script: `python train_v6_giga.py`

### 3. Model Self-Test

To quickly verify that the model implementation is correct and that a forward/backward pass runs without errors, you can execute the model file directly:

```bash
python hst_v6_giga.py
```

This will run a self-test for both `token` and `chunk` modes and report success or failure.
