"""
Representation Learning with HST

This example demonstrates how to use HST for learning representations
of temporal sequences that can be used for downstream tasks.
"""

import numpy as np


def main():
    """Run representation learning example."""
    try:
        import torch
        from hst_v8_2 import HST

        print("=" * 60)
        print("HST Representation Learning Example")
        print("=" * 60)

        # Configuration
        print("\n1. Setting up configuration...")
        seq_len = 100
        n_features = 32
        batch_size = 16
        lattice_levels = 4

        print(f"   Sequence length: {seq_len}")
        print(f"   Feature dimension: {n_features}")
        print(f"   Batch size: {batch_size}")
        print(f"   Lattice levels: {lattice_levels}")

        # Initialize model
        print("\n2. Initializing HST model...")
        model = HST(
            lattice_levels=lattice_levels,
            sequence_length=seq_len,
            feature_dim=n_features,
            harmonic_components=8
        )
        model.eval()
        print("   Model ready for inference!")

        # Generate sample data
        print("\n3. Generating sample temporal sequences...")
        sequences = torch.randn(batch_size, seq_len, n_features)
        print(f"   Input shape: {sequences.shape}")

        # Extract representations at different levels
        print("\n4. Extracting representations at different lattice levels...")

        with torch.no_grad():
            representations = {}

            # Full sequence representation
            full_repr = model(sequences)
            representations['full'] = full_repr
            print(f"   Full representation shape: {full_repr.shape}")

            # Extract representations at specific levels
            try:
                for level in range(lattice_levels):
                    level_repr = model.get_representation(sequences, level=level)
                    representations[f'level_{level}'] = level_repr
                    print(f"   Level {level} representation shape: {level_repr.shape}")
            except AttributeError:
                print("   Note: get_representation method not available, "
                      "using full output only")

        # Analyze representations
        print("\n5. Analyzing learned representations...")

        print("   Statistical properties of representations:")
        for key, repr_tensor in representations.items():
            print(f"\n   {key}:")
            print(f"     Shape: {repr_tensor.shape}")
            print(f"     Min: {repr_tensor.min().item():.4f}")
            print(f"     Max: {repr_tensor.max().item():.4f}")
            print(f"     Mean: {repr_tensor.mean().item():.4f}")
            print(f"     Std: {repr_tensor.std().item():.4f}")

        # Use representations for downstream tasks
        print("\n6. Using representations for downstream tasks...")

        # Example: Classification using first representation
        cls_repr = representations['full']
        # Average pooling over temporal dimension
        pooled = cls_repr.mean(dim=1)  # (batch_size, n_features)
        print(f"   Pooled representation shape: {pooled.shape}")
        print(f"   Ready for classification head")

        # Example: Clustering
        print("\n7. Preparing for unsupervised learning...")
        # Reshape for clustering
        clustering_input = representations['full'].reshape(batch_size, -1)
        print(f"   Flattened shape for clustering: {clustering_input.shape}")
        print(f"   Can be used with K-means, DBSCAN, etc.")

        # Memory inspection
        print("\n8. Inspecting lattice memory structure...")
        try:
            info = model.get_lattice_info()
            print("   Lattice structure information:")
            for key, value in info.items():
                print(f"     {key}: {value}")
        except AttributeError:
            print("   Lattice info not available in this version")

        print("\n" + "=" * 60)
        print("Representation learning example completed!")
        print("=" * 60)

        print("\n   Next steps:")
        print("   - Use pooled representations for classification")
        print("   - Apply clustering algorithms on flattened representations")
        print("   - Use representations as features for downstream models")
        print("   - Compare with other representation learning methods")

    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure to install dependencies:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
