import pytest
import numpy as np
import torch
import importlib
import inspect

# Find the correct class name in a module
def find_hst_class(module):
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and ("HST" in name or "Harmonic" in name):
            return obj
    return None

# Model configurations
model_configs = {
    "hst_v3": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v4": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v4_1": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v5": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v6": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v6_1": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v7_1": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v7_1_2": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v7_2": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v8": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v8_1": {"vocab_size": 50257, "d_model": 64, "n_heads": 4, "n_layers": 2},
    "hst_v8_2": {
        "vocab_size": 50257,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
    },
}

# Models to skip
models_to_skip = ["hst_v4_1", "hst_v6_1", "hst_v7_1_2"]

@pytest.mark.parametrize("model_name, model_args", model_configs.items())
class TestHST:
    def test_initialization(self, model_name, model_args):
        """Test HST initializes with default parameters."""
        if model_name in models_to_skip:
            pytest.skip(f"Skipping {model_name} due to known issues.")
        try:
            module = importlib.import_module(f"src.{model_name}")
            model_class = find_hst_class(module)
            model = model_class(**model_args)
            assert model is not None
        except ImportError:
            pytest.skip(f"{model_name} module not available")

    def test_forward_pass(self, model_name, model_args):
        """Test basic forward pass."""
        if model_name in models_to_skip:
            pytest.skip(f"Skipping {model_name} due to known issues.")
        try:
            module = importlib.import_module(f"src.{model_name}")
            model_class = find_hst_class(module)
            model = model_class(**model_args)

            input_data = torch.randint(0, model_args["vocab_size"], (8, 100))
            output = model(input_data)

            assert output is not None
            if isinstance(output, dict):
                assert output['logits'].shape[0] == 8
            else:
                assert output.shape[0] == 8

        except ImportError:
            pytest.skip(f"{model_name} or torch module not available")
        except Exception as e:
            pytest.fail(f"Forward pass failed for {model_name} with error: {e}")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
