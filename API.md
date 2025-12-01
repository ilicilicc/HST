# API Reference

## HST Core Classes

### HST

Main class for Hierarchical Spatial-Temporal model.

```python
class HST:
    def __init__(
        self,
        lattice_levels: int = 4,
        sequence_length: int = 100,
        feature_dim: int = 32,
        harmonic_components: int = 8,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu'
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lattice_levels` | int | 4 | Number of hierarchical levels in the lattice structure |
| `sequence_length` | int | 100 | Maximum length of input sequences |
| `feature_dim` | int | 32 | Dimensionality of input features |
| `harmonic_components` | int | 8 | Number of harmonic frequency components |
| `dropout_rate` | float | 0.1 | Dropout probability for regularization |
| `activation` | str | 'relu' | Activation function: 'relu', 'tanh', 'sigmoid' |
| `device` | str | 'cpu' | Device for computation: 'cpu' or 'cuda' |

#### Methods

##### forward()

Processes input sequences through the HST model.

```python
def forward(self, x: Tensor) -> Tensor
```

**Parameters:**
- `x`: Input tensor of shape `(batch_size, sequence_length, feature_dim)`

**Returns:**
- Output tensor of shape `(batch_size, sequence_length, feature_dim)`

**Example:**
```python
model = HST(feature_dim=32)
input_seq = torch.randn(32, 100, 32)
output = model.forward(input_seq)
```

##### get_lattice_structure()

Returns the complete lattice structure.

```python
def get_lattice_structure(self) -> Dict[str, Any]
```

**Returns:**
- Dictionary containing lattice topology and node information

**Example:**
```python
lattice = model.get_lattice_structure()
print(lattice['levels'])
print(lattice['nodes'])
```

##### get_level()

Retrieves the representation at a specific lattice level.

```python
def get_level(self, level: int) -> Tensor
```

**Parameters:**
- `level`: Index of the lattice level (0 to lattice_levels-1)

**Returns:**
- Tensor containing the level representation

**Example:**
```python
short_term = model.get_level(0)
long_term = model.get_level(3)
```

##### get_path_weights()

Returns the path weights between lattice nodes.

```python
def get_path_weights(self) -> Tensor
```

**Returns:**
- Tensor of shape `(total_nodes, total_nodes)` with connection weights

**Example:**
```python
weights = model.get_path_weights()
print(f"Number of connections: {weights.sum()}")
```

##### get_representation()

Extracts intermediate representations at specified level.

```python
def get_representation(self, x: Tensor, level: int) -> Tensor
```

**Parameters:**
- `x`: Input tensor of shape `(batch_size, sequence_length, feature_dim)`
- `level`: Lattice level (0 to lattice_levels-1)

**Returns:**
- Intermediate representation tensor

**Example:**
```python
level_2_repr = model.get_representation(input_seq, level=2)
```

##### get_lattice_info()

Returns information about the lattice structure.

```python
def get_lattice_info(self) -> Dict[str, Any]
```

**Returns:**
- Dictionary with keys:
  - `levels`: Number of levels
  - `nodes_per_level`: List of node counts per level
  - `total_connections`: Total number of edges
  - `harmonic_components`: Number of harmonic features

**Example:**
```python
info = model.get_lattice_info()
print(f"Total levels: {info['levels']}")
```

## Utility Functions

### HarmonicHorizonPredictor

Analyzes frequency patterns in memory.

```python
class HarmonicHorizonPredictor:
    def __init__(
        self,
        num_components: int = 8,
        window_size: int = 32
    )
```

#### Methods

##### predict()

Extracts harmonic patterns from sequences.

```python
def predict(self, x: Tensor) -> Tensor
```

**Parameters:**
- `x`: Input tensor

**Returns:**
- Harmonic analysis results

### LatticeBuilder

Constructs the complete lattice structure.

```python
class LatticeBuilder:
    def __init__(
        self,
        levels: int,
        nodes_per_level: List[int]
    )
```

#### Methods

##### build()

Constructs the lattice graph.

```python
def build(self) -> Graph
```

**Returns:**
- Complete lattice graph with all connections

##### add_connections()

Adds weighted connections between levels.

```python
def add_connections(
        self,
        weights: Tensor
    ) -> None
```

### PathWeightedGNN

Graph Neural Network with path weighting.

```python
class PathWeightedGNN:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    )
```

#### Methods

##### forward()

Processes graph data with path weighting.

```python
def forward(
        self,
        node_features: Tensor,
        edges: Tensor,
        weights: Tensor
    ) -> Tensor
```

**Parameters:**
- `node_features`: Node feature matrix
- `edges`: Edge list (source, target pairs)
- `weights`: Edge weights

**Returns:**
- Output embeddings with aggregated information

## Data Structures

### LatticeNode

Represents a single node in the lattice.

```python
class LatticeNode:
    id: int                    # Unique node identifier
    level: int                 # Hierarchical level
    features: Tensor           # Node feature vector
    connections: List[int]     # Connected node IDs
    timestamps: List[int]      # Associated timestamps
```

### MemorySnapshot

Captures the state of memory at a point in time.

```python
class MemorySnapshot:
    timestamp: int             # Time step
    lattice_state: Dict        # Current lattice configuration
    node_activations: Tensor   # Activation values
    path_usage: Tensor         # Which paths were used
```

## Configuration

### Model Configuration File

Create a `config.yaml` for reusable configurations:

```yaml
model:
  lattice_levels: 4
  sequence_length: 100
  feature_dim: 32
  harmonic_components: 8
  dropout_rate: 0.1
  activation: 'relu'

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  optimizer: 'adam'

device: 'cuda'
```

## Error Handling

### Common Exceptions

#### InvalidLatticeConfigError

Raised when lattice configuration is invalid.

```python
try:
    model = HST(lattice_levels=0)  # Invalid
except InvalidLatticeConfigError as e:
    print(f"Configuration error: {e}")
```

#### DimensionMismatchError

Raised when input dimensions don't match model expectations.

```python
try:
    model = HST(feature_dim=32)
    wrong_input = torch.randn(32, 100, 64)
    model(wrong_input)  # Raises error
except DimensionMismatchError as e:
    print(f"Dimension error: {e}")
```

#### OutOfMemoryError

Raised when lattice is too large for available memory.

```python
try:
    model = HST(lattice_levels=20, feature_dim=1024)
except OutOfMemoryError as e:
    print(f"Memory error: {e}")
```

## Performance Metrics

### Model Statistics

```python
# Get model statistics
stats = model.get_statistics()
print(f"Total parameters: {stats['total_params']}")
print(f"Trainable parameters: {stats['trainable_params']}")
print(f"Memory usage: {stats['memory_mb']} MB")
```

## Version Information

Check the installed version:

```python
import hst
print(hst.__version__)  # Returns version string
```

## See Also

- [USAGE.md](USAGE.md) - Usage examples and guides
- [INSTALL.md](INSTALL.md) - Installation instructions
- [CHANGELOG.md](CHANGELOG.md) - Version history and changes
