"""
Basic HST Usage Example

This example demonstrates how to initialize and use the HST model
for a simple forward pass through temporal sequence data.
"""

import numpy as np


def main():
    """Run basic HST usage example."""
    try:
        from hst_v8_2 import HST
        import torch

        print("=" * 60)
        print("HST Basic Usage Example")
        print("=" * 60)

        # Initialize the model
        print("\n1. Initializing HST model...")
        model = HST(
            lattice_levels=4,
            sequence_length=100,
            feature_dim=32,
            harmonic_components=8,
            dropout_rate=0.1,
            activation='relu'
        )
        print(f"   Model initialized successfully!")

        # Prepare sample data
        print("\n2. Preparing sample temporal sequence data...")
        batch_size = 8
        sequence_length = 100
        feature_dim = 32

        # Random temporal sequence data
        input_sequence = torch.randn(batch_size, sequence_length, feature_dim)
        print(f"   Input shape: {input_sequence.shape}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Sequence length: {sequence_length}")
        print(f"   - Feature dimension: {feature_dim}")

        # Forward pass
        print("\n3. Running forward pass...")
        output = model(input_sequence)
        print(f"   Output shape: {output.shape}")
        print(f"   Output type: {type(output)}")

        # Access lattice information
        print("\n4. Accessing lattice structure information...")
        try:
            info = model.get_lattice_info()
            print(f"   Lattice info: {info}")
        except AttributeError:
            print("   Lattice info method not available in this version")

        # Access path weights
        print("\n5. Accessing path weights...")
        try:
            weights = model.get_path_weights()
            print(f"   Path weights shape: {weights.shape}")
            print(f"   Total connections: {weights.sum().item():.2f}")
        except AttributeError:
            print("   Path weights method not available in this version")

        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure to install HST and PyTorch:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
