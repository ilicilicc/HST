# Usage Guide

## Quick Start

### Basic HST Model Usage

```python
from hst_v8_2 import HST
import numpy as np

# Initialize the HST model
model = HST(
    lattice_levels=4,
    sequence_length=100,
    feature_dim=32,
    harmonic_components=8
)

# Prepare your temporal sequence data
X = np.random.randn(32, 100, 32)  # (batch_size, sequence_length, features)

# Forward pass
output = model(X)
print(output.shape)  # Output shape
```

## Core Concepts

### 1. Complete Lattice Core

The HST model replaces traditional linear RNN memory with a multi-level graph structure:

```python
# The model automatically constructs a complete lattice
# with hierarchical levels for memory representation
model = HST(lattice_levels=4)
```

### 2. Path-Weighted GNN Logic

Information flows through multiple paths in the lattice:

```python
# The model calculates path weights internally
# and aggregates information from relevant memory nodes
output = model.forward(sequence)
```

### 3. Hierarchical Storage

Short-term and long-term information are separated:

```python
# High-frequency details (short-term)
# Global trends (long-term)
# Both are captured in the hierarchical lattice
```

### 4. Harmonic Horizon Predictor

Treats memory as frequency/harmonic patterns for stable predictions:

```python
# The harmonic component is integrated into the model
# Use the model for sequence-to-sequence or forecasting tasks
predictions = model(historical_data)
```

## Common Use Cases

### Time Series Forecasting

```python
from hst_v8_2 import HST
import numpy as np

# Initialize model
model = HST(lattice_levels=4, sequence_length=100, feature_dim=16)

# Prepare training data (sequence, target)
train_sequences = np.random.randn(1000, 100, 16)
train_targets = np.random.randn(1000, 1, 16)

# Forward pass
predictions = model(train_sequences)

# Use predictions for training your loss function
```

### Sequence Encoding

```python
# Use HST to encode temporal sequences into fixed representations
encoded = model(sequences)

# Use encoded representations for downstream tasks
```

### Memory Inspection

```python
# Access the internal lattice structure
lattice = model.get_lattice_structure()

# Extract specific memory levels
short_term = model.get_level(0)
long_term = model.get_level(3)

# Analyze connection patterns
paths = model.get_path_weights()
```

## Configuration Parameters

### Model Initialization

```python
model = HST(
    lattice_levels=4,           # Number of hierarchy levels
    sequence_length=100,        # Input sequence length
    feature_dim=32,             # Feature dimension
    harmonic_components=8,      # Number of harmonic components
    dropout_rate=0.1,           # Dropout for regularization
    activation='relu'           # Activation function
)
```

### Parameter Descriptions

- **lattice_levels**: Number of abstraction layers in the lattice (typically 3-6)
- **sequence_length**: Length of input sequences
- **feature_dim**: Dimensionality of features
- **harmonic_components**: Number of frequency components in harmonic analyzer
- **dropout_rate**: Regularization parameter (0.0-0.5)
- **activation**: Activation function ('relu', 'tanh', 'sigmoid')

## Training Example

```python
import torch
import torch.nn as nn
from hst_v8_2 import HST

# Initialize model
model = HST(lattice_levels=4, sequence_length=100, feature_dim=32)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    for batch_sequences, batch_targets in train_loader:
        # Forward pass
        outputs = model(batch_sequences)
        loss = criterion(outputs, batch_targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')
```

## Debugging and Monitoring

### Check Model Structure

```python
print(model)  # Print model architecture
```

### Inspect Lattice

```python
# Access lattice components
lattice_info = model.get_lattice_info()
print(f"Levels: {lattice_info['levels']}")
print(f"Nodes per level: {lattice_info['nodes_per_level']}")
print(f"Connections: {lattice_info['total_connections']}")
```

### Monitor Memory Usage

```python
# Check model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
```

## Performance Tips

1. **Batch Processing**: Use larger batch sizes for better GPU utilization
2. **Sequence Length**: Longer sequences provide better context but increase computation
3. **Lattice Levels**: More levels capture finer details but increase memory usage
4. **Harmonic Components**: Adjust based on your data's frequency characteristics

## Advanced Usage

### Custom Lattice Configuration

```python
# For specific research or applications
model = HST(
    lattice_levels=6,           # Deeper hierarchy
    harmonic_components=16,     # More frequency analysis
    dropout_rate=0.2            # Higher regularization
)
```

### Extracting Representations

```python
# Get intermediate representations at different levels
level_0_repr = model.get_representation(sequences, level=0)
level_1_repr = model.get_representation(sequences, level=1)
level_3_repr = model.get_representation(sequences, level=3)
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py`: Simple model initialization and forward pass
- `time_series_forecasting.py`: Full training pipeline for forecasting
- `representation_learning.py`: Using HST for feature extraction
- `comparative_analysis.py`: Comparing HST with baseline models

## Next Steps

- Check [API.md](API.md) for complete API reference
- Review [CHANGELOG.md](CHANGELOG.md) for version differences
- Visit the [GitHub repository](https://github.com/ilicilicc/HST) for issues and discussions
