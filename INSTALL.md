# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- virtualenv (recommended)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ilicilicc/HST.git
cd HST
```

### 2. Create a Virtual Environment (Recommended)

Using venv:
```bash
python -m venv hst_env
```

Activate the virtual environment:

**On Linux/macOS:**
```bash
source hst_env/bin/activate
```

**On Windows:**
```bash
hst_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import numpy; import torch; print('Installation successful!')"
```

## Development Setup

For development and testing, ensure you have the development dependencies:

```bash
pip install -r requirements.txt
```

### Optional: GPU Support (CUDA)

For GPU acceleration with PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version (cu117, cu121, etc.) or use `cpu` for CPU-only.

## Environment Configuration

No special environment variables are required for basic usage. The HST model runs with default configurations out of the box.

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution:** Ensure PyTorch is installed:
```bash
pip install torch
```

### Issue: Version Compatibility

If you encounter version conflicts, try installing with specific versions:
```bash
pip install -r requirements-pinned.txt
```

### Issue: CUDA/GPU Not Detected

Verify your PyTorch installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

- Read [USAGE.md](USAGE.md) for usage examples
- Check [API.md](API.md) for detailed API documentation
- Review example scripts in the `examples/` directory

## Support

For installation issues, please open an issue on [GitHub Issues](https://github.com/ilicilicc/HST/issues).
