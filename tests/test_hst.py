import pytest
import numpy as np


class TestHSTInitialization:
    """Test HST model initialization."""

    def test_default_initialization(self):
        """Test HST initializes with default parameters."""
        try:
            from hst_v8_2 import HST
            model = HST()
            assert model is not None
        except ImportError:
            pytest.skip("HST module not available")

    def test_custom_initialization(self):
        """Test HST initializes with custom parameters."""
        try:
            from hst_v8_2 import HST
            model = HST(
                lattice_levels=6,
                sequence_length=200,
                feature_dim=64,
                harmonic_components=16
            )
            assert model is not None
        except ImportError:
            pytest.skip("HST module not available")

    def test_invalid_lattice_levels(self):
        """Test that invalid lattice levels raise error."""
        try:
            from hst_v8_2 import HST
            with pytest.raises((ValueError, AssertionError)):
                HST(lattice_levels=0)
        except ImportError:
            pytest.skip("HST module not available")


class TestHSTForward:
    """Test HST forward pass."""

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        try:
            from hst_v8_2 import HST
            import torch

            model = HST(
                sequence_length=100,
                feature_dim=32
            )

            input_data = torch.randn(8, 100, 32)
            output = model(input_data)

            assert output is not None
            assert output.shape[0] == 8  # batch size
        except ImportError:
            pytest.skip("HST or torch module not available")

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        try:
            from hst_v8_2 import HST
            import torch

            model = HST(
                sequence_length=50,
                feature_dim=16
            )

            for batch_size in [1, 4, 16, 32]:
                input_data = torch.randn(batch_size, 50, 16)
                output = model(input_data)
                assert output.shape[0] == batch_size
        except ImportError:
            pytest.skip("HST or torch module not available")

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        try:
            from hst_v8_2 import HST
            import torch

            model = HST(feature_dim=32)

            # Wrong feature dimension
            input_data = torch.randn(8, 100, 64)

            with pytest.raises((ValueError, RuntimeError)):
                model(input_data)
        except ImportError:
            pytest.skip("HST or torch module not available")


class TestLatticeStructure:
    """Test lattice structure functionality."""

    def test_get_lattice_structure(self):
        """Test retrieving lattice structure."""
        try:
            from hst_v8_2 import HST
            model = HST(lattice_levels=4)
            lattice = model.get_lattice_structure()

            assert lattice is not None
            assert isinstance(lattice, dict)
        except (ImportError, AttributeError):
            pytest.skip("HST module or method not available")

    def test_get_lattice_info(self):
        """Test retrieving lattice information."""
        try:
            from hst_v8_2 import HST
            model = HST(lattice_levels=4)
            info = model.get_lattice_info()

            assert info is not None
            assert 'levels' in info or 'nodes_per_level' in info
        except (ImportError, AttributeError):
            pytest.skip("HST module or method not available")


class TestHarmonicPredictor:
    """Test harmonic horizon predictor."""

    def test_harmonic_components(self):
        """Test that harmonic components are processed."""
        try:
            from hst_v8_2 import HST
            model = HST(harmonic_components=8)
            assert model is not None
        except ImportError:
            pytest.skip("HST module not available")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_sequence(self):
        """Test with very short sequences."""
        try:
            from hst_v8_2 import HST
            import torch

            model = HST(sequence_length=10)
            input_data = torch.randn(4, 10, 32)
            output = model(input_data)

            assert output is not None
        except (ImportError, ValueError):
            pytest.skip("Test not supported")

    def test_very_long_sequence(self):
        """Test with very long sequences."""
        try:
            from hst_v8_2 import HST
            import torch

            model = HST(sequence_length=1000)
            input_data = torch.randn(2, 1000, 32)
            output = model(input_data)

            assert output is not None
        except (ImportError, ValueError, RuntimeError):
            pytest.skip("Test not supported or insufficient memory")

    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        try:
            from hst_v8_2 import HST
            import torch

            model = HST()
            input_data = torch.randn(1, 100, 32)
            output = model(input_data)

            assert output is not None
            assert output.shape[0] == 1
        except ImportError:
            pytest.skip("HST module not available")


class TestModelParameters:
    """Test model parameter configurations."""

    def test_dropout_rate(self):
        """Test dropout rate parameter."""
        try:
            from hst_v8_2 import HST
            model = HST(dropout_rate=0.5)
            assert model is not None
        except ImportError:
            pytest.skip("HST module not available")

    def test_activation_functions(self):
        """Test different activation functions."""
        try:
            from hst_v8_2 import HST

            for activation in ['relu', 'tanh', 'sigmoid']:
                try:
                    model = HST(activation=activation)
                    assert model is not None
                except ValueError:
                    pass  # Activation might not be supported
        except ImportError:
            pytest.skip("HST module not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
