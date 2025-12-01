"""
Time Series Forecasting with HST

This example demonstrates how to use HST for time series forecasting tasks.
It shows training a model to predict future values based on historical sequences.
"""

import numpy as np


def generate_synthetic_timeseries(n_samples: int = 1000, seq_len: int = 100,
                                   n_features: int = 32) -> tuple:
    """
    Generate synthetic time series data.

    Args:
        n_samples: Number of samples to generate
        seq_len: Sequence length
        n_features: Number of features

    Returns:
        Tuple of (sequences, targets)
    """
    sequences = np.random.randn(n_samples, seq_len, n_features)
    targets = np.random.randn(n_samples, 1, n_features)
    return sequences, targets


def main():
    """Run time series forecasting example."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from hst_v8_2 import HST

        print("=" * 60)
        print("HST Time Series Forecasting Example")
        print("=" * 60)

        # Hyperparameters
        print("\n1. Setting up hyperparameters...")
        batch_size = 32
        learning_rate = 0.001
        num_epochs = 10
        seq_len = 100
        n_features = 32

        # Generate synthetic data
        print("\n2. Generating synthetic time series data...")
        X, y = generate_synthetic_timeseries(
            n_samples=200,
            seq_len=seq_len,
            n_features=n_features
        )
        print(f"   Generated {len(X)} sequences")
        print(f"   Sequence shape: {X.shape}")
        print(f"   Target shape: {y.shape}")

        # Convert to PyTorch tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        print("\n3. Initializing HST model...")
        model = HST(
            lattice_levels=4,
            sequence_length=seq_len,
            feature_dim=n_features,
            harmonic_components=8,
            dropout_rate=0.1
        )
        print("   Model initialized successfully!")

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        print("\n4. Starting training...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}\n")

        training_losses = []

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch_sequences, batch_targets in dataloader:
                # Forward pass
                outputs = model(batch_sequences)

                # Adjust shapes if needed (targets might have different temporal dim)
                if outputs.shape != batch_targets.shape:
                    # Take first step of output or last step of target
                    if outputs.shape[1] > batch_targets.shape[1]:
                        outputs = outputs[:, :batch_targets.shape[1], :]
                    else:
                        batch_targets = batch_targets[:, :outputs.shape[1], :]

                loss = criterion(outputs, batch_targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            training_losses.append(avg_loss)

            if (epoch + 1) % 2 == 0:
                print(f"   Epoch {epoch + 1}/{num_epochs} - "
                      f"Loss: {avg_loss:.6f}")

        print("\n5. Training completed!")
        print(f"   Final loss: {training_losses[-1]:.6f}")
        print(f"   Loss improvement: "
              f"{(training_losses[0] - training_losses[-1]):.6f}")

        # Evaluation
        print("\n6. Running inference on sample data...")
        model.eval()
        with torch.no_grad():
            sample_input = X_tensor[:4]
            predictions = model(sample_input)
            print(f"   Input shape: {sample_input.shape}")
            print(f"   Prediction shape: {predictions.shape}")
            print(f"   Predictions (first sample):")
            print(f"     Min: {predictions[0].min().item():.4f}")
            print(f"     Max: {predictions[0].max().item():.4f}")
            print(f"     Mean: {predictions[0].mean().item():.4f}")

        print("\n" + "=" * 60)
        print("Time series forecasting example completed!")
        print("=" * 60)

    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure to install dependencies:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
