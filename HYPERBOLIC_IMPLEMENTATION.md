# Hyperbolic Distance Implementation for SinKD

This document describes the implementation of hyperbolic metrics for the SinKD knowledge distillation framework.

## Overview

The implementation adds hyperbolic geometry support to improve knowledge distillation by:
1. Computing distances in hyperbolic space using Poincaré disk model
2. Using hyperbolic KNN to compute adaptive temperature factors
3. Applying hyperbolic metrics in Sinkhorn divergence computation

## Key Components

### 1. Poincaré Distance (`lossd.py:8-51`)

```python
def poincare_distance(x, y, c=1.0, eps=1e-5):
    """
    Compute Poincaré distance between two sets of vectors in hyperbolic space.

    Args:
        x: Tensor of shape (batch_size_1, dim)
        y: Tensor of shape (batch_size_2, dim)
        c: Curvature parameter (default: 1.0)
        eps: Small constant for numerical stability

    Returns:
        Distance matrix of shape (batch_size_1, batch_size_2)
    """
```

The Poincaré distance is computed as:
```
d^2 = (1/c) * arcosh(1 + 2*((x-y)^2) / ((1-|x|^2)*(1-|y|^2)))
```

### 2. Hyperbolic KNN Temperature (`lossd.py:54-115`)

```python
def compute_hyperbolic_knn_temperature(teacher_logits, student_logits, k=5, c=1.0):
    """
    Compute temperature factor based on hyperbolic KNN distances.

    This function:
    1. Maps logits to probability distributions using softmax
    2. Computes KNN for each sample using hyperbolic distance
    3. Calculates local density based on distances to k-nearest neighbors
    4. Derives temperature factors inversely proportional to density

    Returns:
        temperature_t: Temperature for teacher [scalar]
        temperature_s: Temperature for student [scalar]
    """
```

### 3. Updated Sinkhorn Loss (`lossd.py:145-197`)

The `Sinkhorn` class now supports:
- **Hyperbolic distance metric**: Uses `poincare_distance` instead of Euclidean `torch.cdist`
- **Dynamic temperature**: Accepts optional temperature parameter
- **Curvature parameter**: Configurable curvature `c` (default: 1.0)

```python
class Sinkhorn(nn.Module):
    def __init__(self):
        super(Sinkhorn, self).__init__()
        self.T = 2
        self.curvature = 1.0  # Hyperbolic curvature parameter

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=20, use_hyperbolic=True):
        # use_hyperbolic=True enables hyperbolic distance
        ...
```

### 4. Updated KL Loss (`lossd.py:118-143`)

The `KL` class now accepts optional temperature parameter:

```python
class KL(nn.Module):
    def forward(self, y_s, y_t, mode="classification", temperature=None):
        T = temperature if temperature is not None else self.T
        ...
```

## Usage

### Basic Usage (Fixed Temperature)

```python
from lossd import KL, Sinkhorn

# Standard usage with default fixed temperature
loss_kl = KL()
loss_sk = Sinkhorn()

kl_loss = loss_kl(student_logits, teacher_logits)
sk_loss = loss_sk(student_logits, teacher_logits)
```

### Advanced Usage (Adaptive Hyperbolic Temperature)

```python
from lossd import KL, Sinkhorn, compute_hyperbolic_knn_temperature

# Compute adaptive temperature based on hyperbolic KNN
temp_t, temp_s = compute_hyperbolic_knn_temperature(
    teacher_logits,
    student_logits,
    k=5,  # Number of nearest neighbors
    c=1.0  # Curvature
)

# Use average temperature
avg_temp = (temp_t + temp_s) / 2.0

# Apply temperature to distillation losses
kl_loss = loss_kl(student_logits, teacher_logits, temperature=avg_temp)
sk_loss = loss_sk(student_logits, teacher_logits, temperature=avg_temp)
```

### Training with Hyperbolic Temperature (distillation.py)

The main training loop in `distillation.py` has been updated to use hyperbolic temperature:

```python
use_hyperbolic_temp = True  # Set to False to use fixed temperature

for epoch in range(1, 11):
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        with torch.no_grad():
            outputt = tmodel(**batch)

        # Compute hyperbolic KNN-based temperature
        if use_hyperbolic_temp:
            temp_t, temp_s = compute_hyperbolic_knn_temperature(
                outputt.logits.detach(),
                outputs.logits.detach(),
                k=5,
                c=1.0
            )
            hyperbolic_temp = (temp_t + temp_s) / 2.0
        else:
            hyperbolic_temp = None

        # Compute distillation losses with hyperbolic temperature
        loss_kl = losskl(outputs.logits, outputt.logits, temperature=hyperbolic_temp)
        loss_sk = lossskl(outputs.logits, outputt.logits, temperature=hyperbolic_temp)

        loss1 = 0.1 * loss_kl + 1.0 * loss_sk
        loss = loss1 + outputs.loss
        ...
```

## Configuration Options

### Curvature Parameter (c)

The curvature parameter controls the "bendiness" of the hyperbolic space:
- **c = 1.0**: Default curvature
- **c > 1.0**: More curved (smaller distances)
- **c < 1.0**: Less curved (larger distances)

### Number of Neighbors (k)

For KNN temperature computation:
- **k = 3-5**: Small values focus on very local structure
- **k = 10-20**: Larger values capture broader neighborhood structure
- **Default: k = 5**

### Temperature Normalization

Temperatures are automatically normalized to have mean ≈ 2.0 (similar to original fixed temperature).

## Implementation Details

### Numerical Stability

The implementation includes several safeguards:
1. **Clamping**: Probabilities are clamped to ensure they stay within Poincaré disk (|x|^2 < 1)
2. **Epsilon**: Small constant (1e-5) prevents division by zero
3. **Arcosh argument**: Ensured to be ≥ 1 for numerical stability
4. **Gradient isolation**: Temperature computation uses `.detach()` to avoid affecting gradients

### Computational Complexity

- **Poincaré distance**: O(batch_size² × dim)
- **KNN computation**: O(batch_size² × dim)
- **Overall**: Similar complexity to original Euclidean version

## File Changes

1. **T0/lossd.py**:
   - Added `poincare_distance()` function
   - Added `compute_hyperbolic_knn_temperature()` function
   - Modified `Sinkhorn.sinkhorn_loss()` to support hyperbolic distance
   - Modified `Sinkhorn.forward()` to accept temperature parameter
   - Modified `KL.forward()` to accept temperature parameter

2. **T0/distillation.py**:
   - Imported `compute_hyperbolic_knn_temperature`
   - Updated training loop to compute and use hyperbolic temperature
   - Added error handling with fallback to fixed temperature

## Experimental Results

To compare hyperbolic vs. Euclidean metrics:

```bash
# Run with hyperbolic temperature (enabled)
python T0/distillation.py ...  # use_hyperbolic_temp = True

# Run with fixed temperature (original)
# Set use_hyperbolic_temp = False in distillation.py
python T0/distillation.py ...
```

## Future Improvements

Potential enhancements:
1. **Learnable curvature**: Make curvature parameter trainable
2. **Per-sample temperatures**: Use per-sample temperatures instead of averaged
3. **Alternative embeddings**: Project embeddings to hyperbolic space explicitly
4. **Adaptive k**: Dynamically adjust k based on batch properties

## References

- [Poincaré Embeddings](https://arxiv.org/abs/1705.08039)
- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09386)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [Sinkhorn Divergence](https://arxiv.org/abs/1809.01962)

## Contact

For questions or issues, please refer to the main repository documentation.
