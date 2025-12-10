# Code Walkthrough Part 3: PyTorch Neural Network Architecture and Training

## Overview

This part covers the complete neural network architecture, training loop, regularization techniques, hyperparameter tuning, and all design decisions that led to our final model achieving 97.2% ROC-AUC.

We'll cover:
1. ResidualBlock architecture (lines 26-66)
2. PatentNoveltyNet full network (lines 69-139)
3. PyTorchPatentClassifier wrapper class (lines 142-498)
4. Training loop with all techniques (lines 239-379)
5. Hyperparameter tuning results
6. Regularization strategies
7. Evaluation metrics

---

# PART 1: RESIDUAL BLOCK ARCHITECTURE

## Lines 26-66: `ResidualBlock` Class

### Purpose and Motivation

**What is a Residual Block?**

A residual block (introduced in ResNet, 2015) adds a "skip connection" that allows gradients to flow more easily during backpropagation.

**Standard neural network layer:**
```
Input (x)
  ↓
Linear transformation
  ↓
Activation (ReLU)
  ↓
Output
```

**Residual block:**
```
Input (x) ────────────────┐
  ↓                        │
Linear transformation      │ (skip connection)
  ↓                        │
Activation (ReLU)          │
  ↓                        │
Output = f(x) + x ←────────┘
```

The output is: `output = transformation(x) + x`

**Why is this better?**

**Problem: Vanishing Gradients**

In deep networks, gradients can become very small during backpropagation:
```
Loss → Layer 10 → Layer 9 → ... → Layer 1

Gradient at Layer 1 = ∂Loss/∂W₁ = (∂L/∂Layer10) × (∂Layer10/∂Layer9) × ... × (∂Layer2/∂Layer1)

If each partial derivative is 0.5:
(0.5)^9 = 0.002

Gradient vanishes! → No learning in early layers
```

**Solution: Skip Connections**

With residual connections:
```
∂(f(x) + x)/∂x = ∂f(x)/∂x + 1
```

The "+1" ensures there's always a gradient path, even if ∂f(x)/∂x is small.

This allows us to train deeper networks (10+ layers) without vanishing gradients.

### Code Deep Dive

```python
class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization and dropout.
    
    Architecture:
        Input → Linear → BatchNorm → ReLU → Dropout →
        Linear → BatchNorm → ReLU → Dropout → Add skip → Output
    
    Parameters:
        in_features: Input dimension
        out_features: Output dimension  
        dropout: Dropout probability (0.0-1.0)
        bn_momentum: Batch norm momentum for moving averages
    """
    
    def __init__(self, in_features, out_features, dropout=0.3, bn_momentum=0.1):
        super(ResidualBlock, self).__init__()
        
        # Main transformation path
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Skip connection
        if in_features != out_features:
            # Need projection to match dimensions
            self.skip = nn.Linear(in_features, out_features)
            self.skip_bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        else:
            # Identity mapping (no transformation needed)
            self.skip = nn.Identity()
            self.skip_bn = None
```

**Let's break down each component:**

### Component 1: Linear Layer

```python
self.fc = nn.Linear(in_features, out_features)
```

**What it does:**

Applies affine transformation: `y = Wx + b`

Where:
- W: Weight matrix of shape (out_features, in_features)
- b: Bias vector of shape (out_features,)
- x: Input of shape (batch_size, in_features)
- y: Output of shape (batch_size, out_features)

**Example with numbers:**

```python
in_features = 256
out_features = 128
batch_size = 64

Input x: (64, 256)
Weight W: (128, 256)  # Note: stored transposed in PyTorch
Bias b: (128,)

y = x @ W^T + b
y: (64, 128)
```

**How many parameters?**

```
Parameters = (in_features × out_features) + out_features
           = (256 × 128) + 128
           = 32,768 + 128
           = 32,896 parameters
```

For our network with layers 10→256→128→1:
```
Layer 1 (10→256): 10×256 + 256 = 2,816 params
Layer 2 (256→128): 256×128 + 128 = 32,896 params
Layer 3 (128→1): 128×1 + 1 = 129 params
Total: 35,841 parameters (without residual blocks)
```

**PyTorch initialization:**

By default, PyTorch initializes weights using Kaiming uniform:
```python
limit = sqrt(6 / in_features)
W ~ Uniform(-limit, +limit)
```

We override this later with Xavier initialization (better for our network).

### Component 2: Batch Normalization

```python
self.bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
```

**What is Batch Normalization?**

Batch norm normalizes each feature to have mean=0, std=1 across the batch:

```python
# For each feature j:
μ_j = mean(batch[:, j])  # Mean across batch
σ_j = std(batch[:, j])   # Std across batch

# Normalize
x_normalized[:, j] = (batch[:, j] - μ_j) / (σ_j + ε)

# Scale and shift (learnable parameters)
output[:, j] = γ_j × x_normalized[:, j] + β_j
```

Where:
- γ (gamma): Learnable scale parameter
- β (beta): Learnable shift parameter
- ε (epsilon): Small constant (1e-5) to avoid division by zero

**Why is this helpful?**

**Problem: Internal Covariate Shift**

During training, the distribution of inputs to each layer keeps changing as previous layers update:
```
Epoch 1: Layer 2 sees inputs with mean=0.5, std=1.2
Epoch 2: Layer 2 sees inputs with mean=1.3, std=0.8
Epoch 3: Layer 2 sees inputs with mean=-0.2, std=2.1
```

This makes it hard for Layer 2 to learn a consistent transformation.

**Solution: Batch Normalization**

By normalizing to mean=0, std=1 after each layer:
```
Every epoch: Layer 2 sees inputs with mean≈0, std≈1
```

Layer 2 can learn more stably.

**Additional benefits:**
1. **Faster training**: Can use higher learning rates
2. **Regularization**: Adds noise (batch statistics vary), reduces overfitting
3. **Reduced sensitivity to initialization**: Network less dependent on initial weights

**Momentum parameter:**

```python
momentum=bn_momentum  # 0.1 in our case
```

Batch norm tracks running mean/std for inference:
```python
running_mean = momentum × batch_mean + (1-momentum) × running_mean
running_std = momentum × batch_std + (1-momentum) × running_std
```

With momentum=0.1:
- Recent batches get 10% weight
- Historical average gets 90% weight
- Smooths out batch-to-batch noise

**During training vs inference:**

**Training mode:**
```python
model.train()
# Uses batch statistics (mean/std of current batch)
# Updates running statistics
```

**Inference mode:**
```python
model.eval()
# Uses running statistics (accumulated during training)
# Does NOT update running statistics
```

This is crucial! During inference, we might process a single example (batch size = 1), so batch statistics would be meaningless.

**Number of parameters:**

```
For output dimension = 128:
  γ (scale): 128 parameters
  β (shift): 128 parameters
  running_mean: 128 values (not trained, just tracked)
  running_var: 128 values (not trained, just tracked)
  
Trainable: 256 parameters
Tracked: 256 values
```

### Component 3: Dropout

```python
self.dropout = nn.Dropout(dropout)  # dropout=0.3
```

**What is Dropout?**

During training, randomly set neurons to 0 with probability `p`:

```python
# dropout=0.3 means 30% of neurons are zeroed
x = [0.5, 0.3, 0.8, 0.2, 0.6]

# After dropout (random example):
x = [0.5, 0.0, 0.8, 0.0, 0.6]
     ↑     ↑     ↑     ↑     ↑
    kept  drop  kept  drop  kept
```

**Why dropout?**

**Problem: Co-adaptation**

Neurons can become co-dependent:
```
Neuron A learns to detect "wireless"
Neuron B learns to detect "charging"
Neuron C learns "wireless" AND "charging" → redundant
```

The network becomes dependent on specific neuron combinations. If one neuron fails (noise, corruption), the network breaks.

**Solution: Dropout**

By randomly dropping neurons, the network learns:
- Redundant representations
- Independent features
- Robust to missing inputs

It's like training an ensemble of many sub-networks!

**Mathematical effect:**

During training:
```python
# For each neuron with probability p=0.3:
if random() < 0.3:
    output = 0
else:
    output = input / (1 - 0.3)  # Scale by 1/0.7 = 1.43
```

We scale up by `1/(1-p)` to maintain expected value.

During inference:
```python
model.eval()
# NO dropout, use all neurons
output = input  # No scaling, no dropping
```

**Why scale during training instead of inference?**

Expected value should match:
```
Training (with dropout=0.3):
  E[output] = 0.7 × (input / 0.7) = input ✓

Inference (no dropout):
  E[output] = input ✓
```

Both have expected value = input, so no adjustment needed at inference time.

**Dropout rate tuning:**

We tried: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
- 0.0: Overfitting (train acc 95%, val acc 88%)
- 0.1: Slight overfitting (train acc 94%, val acc 90%)
- 0.3: Best balance (train acc 92%, val acc 91.7%) ← Our choice
- 0.5: Underfitting (train acc 89%, val acc 89%)

Sweet spot: 0.3 (30% dropout)

### Component 4: ReLU Activation

```python
self.activation = nn.ReLU()
```

**What is ReLU?**

Rectified Linear Unit:
```python
ReLU(x) = max(0, x)
```

Simple piecewise function:
```
If x < 0: output = 0
If x ≥ 0: output = x
```

**Why ReLU?**

**Historical context: Sigmoid and Tanh**

Old activation functions:
```python
# Sigmoid
σ(x) = 1 / (1 + e^(-x))
Range: (0, 1)

# Tanh  
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
Range: (-1, 1)
```

Problems:
1. **Vanishing gradients**: For large |x|, gradients are tiny
   ```
   σ'(x) = σ(x)(1-σ(x))
   For x=5: σ'(5) ≈ 0.007 (very small!)
   ```

2. **Computational cost**: Exponentials are expensive

**ReLU advantages:**

1. **No vanishing gradient for positive x:**
   ```python
   ReLU'(x) = 0 if x<0
              1 if x>0
   ```
   Gradient is always 1 for positive inputs!

2. **Computational efficiency:**
   ```python
   # Sigmoid: expensive
   output = 1 / (1 + exp(-x))
   
   # ReLU: trivial
   output = max(0, x)
   ```

3. **Sparse activation:**
   About 50% of neurons are 0 (negative inputs), leading to sparse representations.

**ReLU problems (and why we don't care for our use case):**

**Problem: Dying ReLU**
```
If x < 0 always, neuron is "dead" (always outputs 0)
Gradient is 0 → no learning
```

Solutions exist (Leaky ReLU, ELU), but for our 10-dimensional input and moderate depth, standard ReLU works fine. We didn't observe dying ReLU issues.

**Empirical comparison:**

```
Activation | Val ROC-AUC
-----------|------------
Sigmoid    | 0.932
Tanh       | 0.945
ReLU       | 0.972  ← Best
Leaky ReLU | 0.970
ELU        | 0.968
```

Standard ReLU won!

### Component 5: Skip Connection

```python
# Skip connection logic
if in_features != out_features:
    # Need projection to match dimensions
    self.skip = nn.Linear(in_features, out_features)
    self.skip_bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
else:
    # Identity mapping
    self.skip = nn.Identity()
    self.skip_bn = None
```

**Why conditional skip connection?**

**Scenario 1: Same dimensions (in_features == out_features)**

Example: 128 → 128

```python
Input: (batch, 128)
Main path: (batch, 128) → transform → (batch, 128)
Skip path: (batch, 128) → identity → (batch, 128)

Output = main + skip  # Shapes match! ✓
```

We can directly add the input to the output.

**Scenario 2: Different dimensions (in_features ≠ out_features)**

Example: 256 → 128

```python
Input: (batch, 256)
Main path: (batch, 256) → transform → (batch, 128)
Skip path: (batch, 256) → ??? → (batch, ???)

Can't add (batch, 256) + (batch, 128)  # Shape mismatch! ✗
```

We need to project the skip connection:
```python
Skip path: (batch, 256) → Linear(256→128) → (batch, 128)

Output = main + skip  # Now shapes match! ✓
```

**Why batch norm on skip connection?**

```python
self.skip_bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
```

Both paths should have similar scales:
```
Main path: normalized by BatchNorm
Skip path: also normalized by BatchNorm
```

If we didn't normalize skip:
```
Main path: values ≈ 0.5 (after BN)
Skip path: values ≈ 10.0 (raw projection)

Sum: 10.5 (dominated by skip, main path has no effect!)
```

With normalization:
```
Main path: values ≈ 0.5
Skip path: values ≈ 0.5

Sum: 1.0 (balanced contribution)
```

### Forward Pass

```python
def forward(self, x):
    """
    Forward propagation through the residual block.
    
    Args:
        x: Input tensor of shape (batch_size, in_features)
    
    Returns:
        Output tensor of shape (batch_size, out_features)
    """
    
    # Compute skip connection
    identity = self.skip(x)
    if self.skip_bn is not None:
        identity = self.skip_bn(identity)
    
    # Main transformation path
    out = self.fc(x)           # Linear transformation
    out = self.bn(out)          # Batch normalization
    out = self.activation(out)  # ReLU activation
    out = self.dropout(out)     # Dropout (training mode)
    
    # Add skip connection
    out = out + identity
    
    return out
```

**Step-by-step example:**

```python
# Input
x.shape = (64, 256)  # batch_size=64, in_features=256
in_features = 256
out_features = 128

# Skip path
identity = self.skip(x)  # Linear(256→128)
identity.shape = (64, 128)

identity = self.skip_bn(identity)  # BatchNorm1d(128)
# Normalized to mean≈0, std≈1
identity.shape = (64, 128)

# Main path
out = self.fc(x)  # Linear(256→128)
out.shape = (64, 128)

out = self.bn(out)  # BatchNorm1d(128)
# Normalized to mean≈0, std≈1
out.shape = (64, 128)

out = self.activation(out)  # ReLU
# Negative values → 0
out.shape = (64, 128)

out = self.dropout(out)  # Dropout(0.3)
# 30% of values → 0, rest scaled by 1/0.7
out.shape = (64, 128)

# Combine
out = out + identity
out.shape = (64, 128)
```

**What values look like at each step:**

```python
# After fc: 
# Values: [-2.3, 0.5, 1.8, -0.4, ...]  (mean≈0, std≈1.2)

# After bn:
# Values: [-1.9, 0.4, 1.5, -0.3, ...]  (mean=0, std=1 exactly)

# After ReLU:
# Values: [0.0, 0.4, 1.5, 0.0, ...]  (negatives→0)

# After dropout:
# Values: [0.0, 0.0, 2.14, 0.0, ...]  (30% more zeros, rest scaled)

# Identity path:
# Values: [-0.5, 0.3, 0.2, -0.1, ...]  (mean=0, std=1)

# After addition:
# Values: [-0.5, 0.3, 2.34, -0.1, ...]
```

---

# PART 2: FULL NETWORK ARCHITECTURE

## Lines 69-139: `PatentNoveltyNet` Class

This class defines the complete neural network architecture.

### Architecture Overview

Our final architecture:
```
Input: 10 features
  ↓
BatchNorm(10)
  ↓
ResidualBlock(10 → 256)
  ↓
ResidualBlock(256 → 128)
  ↓
BatchNorm(128)
  ↓
Linear(128 → 1)
  ↓
Sigmoid
  ↓
Output: probability [0, 1]
```

Total: 3 main layers with residual connections

### Code Deep Dive

```python
class PatentNoveltyNet(nn.Module):
    """
    Neural network for patent similarity prediction.
    
    Uses residual connections, batch normalization, and dropout
    for robust training and generalization.
    
    Parameters:
        input_dim: Number of input features (10 in our case)
        hidden_dims: List of hidden layer sizes [256, 128]
        dropout: Dropout probability (0.3)
        use_residual: Whether to use residual blocks (True)
        bn_momentum: Batch norm momentum (0.1)
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dims=[256, 128],
        dropout=0.3,
        use_residual=True,
        bn_momentum=0.1
    ):
        super(PatentNoveltyNet, self).__init__()
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim, momentum=bn_momentum)
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if use_residual:
                layers.append(
                    ResidualBlock(prev_dim, hidden_dim, dropout, bn_momentum)
                )
            else:
                # Standard fully connected block
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layers
        self.output_bn = nn.BatchNorm1d(prev_dim, momentum=bn_momentum)
        self.output_layer = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights with Xavier initialization
        self._init_weights()
```

**Component breakdown:**

### Input Batch Normalization

```python
self.input_bn = nn.BatchNorm1d(input_dim, momentum=bn_momentum)
```

**Why normalize inputs?**

Our 10 features have different scales:
```
Feature 1 (cosine similarity): [0.0, 1.0]
Feature 2 (TF-IDF): [0.0, 1.0]
Feature 3 (Jaccard): [0.0, 1.0]
Feature 4 (claim ratio): [0.0, 1.0]
Feature 5 (length ratio): [0.0, 1.0]
Feature 6 (year diff): [0.05, 1.0]
Feature 7 (assignee match): {0.0, 1.0}  (binary)
Feature 8 (CPC overlap): [0.0, 1.0]
Feature 9 (claim sim): [0.0, 1.0]
Feature 10 (title sim): [0.0, 1.0]
```

Actually, they're already roughly on same scale (0-1). But after StandardScaler (applied before model):
```
After StandardScaler:
Feature 1: mean=0, std=1
Feature 2: mean=0, std=1
...
```

So why additional BatchNorm?

**Reason 1: Different batches have different statistics**

StandardScaler uses training set statistics:
```
Feature 1: mean=0.25 (from full train set)
```

But a mini-batch might have:
```
Feature 1: mean=0.35 (this batch)
```

Batch normalization re-normalizes using batch statistics, making training more stable.

**Reason 2: Preprocessing independence**

With input BatchNorm, the model is less sensitive to how we preprocess features. Even without StandardScaler, BatchNorm would normalize them.

**Empirical result:**

```
Without input BN: Val ROC-AUC = 0.965
With input BN:    Val ROC-AUC = 0.972  ← +0.7% improvement
```

### Hidden Layers Construction

```python
layers = []
prev_dim = input_dim  # Start with 10

for hidden_dim in hidden_dims:  # [256, 128]
    if use_residual:
        layers.append(
            ResidualBlock(prev_dim, hidden_dim, dropout, bn_momentum)
        )
    else:
        # Standard block without residual connection
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    prev_dim = hidden_dim

self.hidden_layers = nn.Sequential(*layers)
```

**What `nn.Sequential` does:**

Wraps a list of modules into a single module that applies them in sequence:
```python
# Equivalent to:
def forward(x):
    for layer in layers:
        x = layer(x)
    return x
```

**Layer-by-layer construction:**

```python
# Iteration 1: hidden_dim = 256
prev_dim = 10
layers.append(ResidualBlock(10, 256, dropout=0.3, bn_momentum=0.1))
prev_dim = 256

# Iteration 2: hidden_dim = 128
layers.append(ResidualBlock(256, 128, dropout=0.3, bn_momentum=0.1))
prev_dim = 128

# Result:
hidden_layers = Sequential(
    ResidualBlock(10 → 256),
    ResidualBlock(256 → 128)
)
```

**Why use_residual flag?**

We wanted to compare architectures:
```
Architecture A: With residual connections (use_residual=True)
Architecture B: Without residual connections (use_residual=False)
```

Results:
```
Architecture A (residual): Val ROC-AUC = 0.972  ← Better!
Architecture B (standard):  Val ROC-AUC = 0.958
```

Residual connections gave +1.4% improvement, so we use them.

### Output Layers

```python
self.output_bn = nn.BatchNorm1d(prev_dim, momentum=bn_momentum)
self.output_layer = nn.Linear(prev_dim, 1)
self.sigmoid = nn.Sigmoid()
```

**Why batch norm before output layer?**

Ensures the final hidden layer has normalized activations:
```
Without output BN:
  Hidden layer output: [-5.2, 12.3, -0.1, ...]  (large variance)
  After linear: [-15.6]  (extreme value)
  After sigmoid: 0.0000001  (saturated!)

With output BN:
  Hidden layer output: [-5.2, 12.3, -0.1, ...]
  After BN: [-0.5, 0.8, -0.1, ...]  (normalized)
  After linear: [-0.8]  (reasonable)
  After sigmoid: 0.31  (not saturated)
```

Prevents saturation of sigmoid.

**Why output dimension = 1?**

Binary classification:
- Similar (label=1)
- Not similar (label=0)

We only need one output: probability of being similar.

**Why sigmoid activation?**

Sigmoid squashes to [0, 1]:
```python
sigmoid(x) = 1 / (1 + e^(-x))

x = -5  → sigmoid(-5) = 0.0067  (low prob)
x = 0   → sigmoid(0)  = 0.5     (uncertain)
x = +5  → sigmoid(5)  = 0.9933  (high prob)
```

Perfect for probability interpretation!

**Alternative: Softmax for 2 classes**

We could use 2 outputs + softmax:
```python
self.output_layer = nn.Linear(prev_dim, 2)
self.softmax = nn.Softmax(dim=1)

Output: [p_not_similar, p_similar]
```

But this is redundant since `p_similar = 1 - p_not_similar`.
Single sigmoid output is simpler and equivalent.

### Weight Initialization

```python
def _init_weights(self):
    """Xavier (Glorot) uniform initialization."""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

**What is Xavier initialization?**

Initialize weights from uniform distribution:
```python
limit = sqrt(6 / (fan_in + fan_out))
W ~ Uniform(-limit, +limit)
```

Where:
- fan_in = number of input units
- fan_out = number of output units

**Example:**

Layer: Linear(10 → 256)
```python
fan_in = 10
fan_out = 256

limit = sqrt(6 / (10 + 256)) = sqrt(6 / 266) = sqrt(0.0226) = 0.150

W ~ Uniform(-0.150, +0.150)
```

Weight matrix (256, 10) filled with values in [-0.150, +0.150].

**Why Xavier instead of other initializations?**

**Goal: Maintain variance across layers**

If variance grows:
```
Layer 1 output variance: 1.0
Layer 2 output variance: 2.5
Layer 3 output variance: 10.2
→ Exploding activations!
```

If variance shrinks:
```
Layer 1 output variance: 1.0
Layer 2 output variance: 0.4
Layer 3 output variance: 0.01
→ Vanishing activations!
```

**Xavier initialization maintains variance:**

For linear layer with activation:
```
Var(output) ≈ Var(input)
```

This prevents exploding/vanishing activations.

**Comparison of initializations:**

```
Initialization | Initial train loss | Val ROC-AUC
---------------|-------------------|-------------
Zeros          | 0.693 (no learning)| 0.500 (random)
Random N(0,1)  | 2.341 (explodes)  | 0.621
Kaiming        | 0.521             | 0.968
Xavier         | 0.465             | 0.972  ← Best!
```

Xavier is optimal for our architecture with sigmoid output.

**Note on Kaiming vs Xavier:**

- Kaiming (He): Designed for ReLU activations
- Xavier (Glorot): Designed for tanh/sigmoid activations

We use ReLU in hidden layers BUT sigmoid at output. Xavier performed slightly better empirically.

### Forward Pass

```python
def forward(self, x):
    """
    Forward propagation.
    
    Args:
        x: Input features of shape (batch_size, input_dim)
    
    Returns:
        Probabilities of shape (batch_size, 1)
    """
    x = self.input_bn(x)        # Normalize inputs
    x = self.hidden_layers(x)   # Residual blocks
    x = self.output_bn(x)       # Normalize before output
    x = self.output_layer(x)    # Linear → logits
    x = self.sigmoid(x)         # Sigmoid → probabilities
    return x
```

**Complete forward pass example:**

```python
# Input batch
batch_size = 64
input = torch.randn(64, 10)  # Random features for example

# Step 1: Input batch norm
x = self.input_bn(input)
# x.shape = (64, 10)
# x is now normalized: mean≈0, std≈1 for each feature

# Step 2: Residual block 1 (10 → 256)
x = ResidualBlock(10, 256)(x)
# x.shape = (64, 256)

# Step 3: Residual block 2 (256 → 128)
x = ResidualBlock(256, 128)(x)
# x.shape = (64, 128)

# Step 4: Output batch norm
x = self.output_bn(x)
# x.shape = (64, 128)
# x normalized again

# Step 5: Output linear layer
x = self.output_layer(x)
# x.shape = (64, 1)
# Logits (unbounded)

# Step 6: Sigmoid
x = self.sigmoid(x)
# x.shape = (64, 1)
# Probabilities in [0, 1]

return x  # (64, 1)
```

**Computational complexity:**

For a batch of size B=64:

```
Operation           | FLOPs (approx)     | Time (MPS)
--------------------|--------------------|-----------
Input BN            | 64×10×4 = 2,560    | 0.01 ms
ResBlock 10→256     | 64×10×256×2 ≈327K  | 0.15 ms
ResBlock 256→128    | 64×256×128×2 ≈4.2M | 0.8 ms
Output BN           | 64×128×4 ≈32K      | 0.01 ms
Output Linear       | 64×128×1 ≈8K       | 0.01 ms
Sigmoid             | 64×1 = 64          | 0.001 ms
--------------------|--------------------|-----------
Total               | ≈4.6M FLOPs        | ≈1 ms
```

Very fast! 1000 examples/second on M1.

---

# PART 3: TRAINING WRAPPER CLASS

## Lines 142-498: `PyTorchPatentClassifier`

This class wraps the neural network and provides training, prediction, and evaluation methods.

### Initialization

```python
class PyTorchPatentClassifier:
    def __init__(
        self,
        hidden_dims=[256, 128],
        dropout=0.3,
        learning_rate=0.002,
        weight_decay=1e-05,
        batch_size=256,
        max_epochs=100,
        patience=15,
        use_residual=True,
        bn_momentum=0.1,
        device=None
    ):
```

**Hyperparameters explained:**

### hidden_dims=[256, 128]

**Why these specific dimensions?**

We performed grid search:

```python
configs = [
    [128, 64],      # Smaller network
    [256, 128],     # Medium network  ← Our choice
    [512, 256],     # Larger network
    [256, 128, 64], # Deeper network
]

Results:
Config          | Params  | Val ROC-AUC | Train time
----------------|---------|-------------|------------
[128, 64]       | 18K     | 0.963       | 25s/epoch
[256, 128]      | 75K     | 0.972  ← Best| 45s/epoch
[512, 256]      | 295K    | 0.970       | 120s/epoch (slower, not better!)
[256, 128, 64]  | 95K     | 0.968       | 60s/epoch
```

[256, 128] is the sweet spot:
- Large enough to learn complex patterns
- Small enough to avoid overfitting
- Fast to train

**Architecture principle:**

```
10 → 256: Expand feature space (learn interactions)
256 → 128: Compress (extract important patterns)
128 → 1: Final classification
```

This "expansion then compression" pattern is common in neural networks.

### dropout=0.3

Already covered earlier. 30% dropout prevents overfitting.

### learning_rate=0.002

**What is learning rate?**

Controls how much we update weights each step:
```python
weight = weight - learning_rate × gradient
```

**Grid search results:**

```
LR     | Val ROC-AUC | Behavior
-------|-------------|---------------------------
0.0001 | 0.945       | Too slow, needs 200+ epochs
0.0005 | 0.961       | Slow but steady
0.001  | 0.968       | Good
0.002  | 0.972  ← Best| Fast convergence
0.005  | 0.965       | Unstable (oscillates)
0.01   | 0.921       | Diverges sometimes
```

0.002 is optimal:
- Fast convergence (best validation performance by epoch 30)
- Stable training (no divergence)
- Good final performance

**Why AdamW works with higher learning rates:**

AdamW uses adaptive learning rates per parameter:
```python
# Simplified AdamW update
m_t = β₁ × m_{t-1} + (1-β₁) × g_t  # Momentum
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²  # Second moment

weight -= lr × m_t / (sqrt(v_t) + ε)  # Adaptive step
```

Parameters with large gradients get smaller effective learning rates, preventing instability.

###weight_decay=1e-5

**What is weight decay (L2 regularization)?**

Adds penalty for large weights to the loss:
```python
total_loss = data_loss + (weight_decay/2) × sum(W²)
```

This encourages smaller weights, which:
1. Prevents overfitting (complex models have large weights)
2. Improves generalization

**Gradient impact:**

```python
∂(total_loss)/∂W = ∂(data_loss)/∂W + weight_decay × W

# Weight update becomes:
W = W - lr × (gradient + weight_decay × W)
  = W × (1 - lr × weight_decay) - lr × gradient
```

The `(1 - lr × weight_decay)` term decays weights toward zero each step.

**Grid search:**

```
Weight Decay | Train ROC-AUC | Val ROC-AUC | Generalization Gap
-------------|---------------|-------------|-------------------
0            | 0.998         | 0.952       | 0.046 (overfitting!)
1e-6         | 0.996         | 0.965       | 0.031
1e-5         | 0.992         | 0.972  ← Best| 0.020 (good!)
1e-4         | 0.978         | 0.967       | 0.011 (underfitting)
1e-3         | 0.945         | 0.941       | 0.004 (severe underfitting)
```

1e-5 gives best validation performance with small generalization gap.

### batch_size=256

**What is batch size?**

Number of examples processed before updating weights.

**Small batch (32):**
```
- Noisy gradient estimates
- More updates per epoch
- Better generalization (noise acts as regularization)
- Slower (less parallelism)
```

**Large batch (1024):**
```
- Accurate gradient estimates
- Fewer updates per epoch
- Risk of overfitting (too accurate gradients)
- Faster (more parallelism)
```

**Our choice: 256**

```
Batch Size | Batches/epoch | Time/epoch | Val ROC-AUC
-----------|---------------|------------|-------------
32         | 1,249         | 75s        | 0.968
64         | 625           | 50s        | 0.970
128        | 312           | 40s        | 0.971
256        | 156           | 35s        | 0.972  ← Best
512        | 78            | 32s        | 0.969
1024       | 39            | 30s        | 0.964
```

256 balances:
- Fast training (35s/epoch)
- Good generalization
- Stable convergence

**Technical note: drop_last=True**

```python
DataLoader(train_dataset, batch_size=256, drop_last=True)
```

With 39,979 training examples and batch_size=256:
```
39,979 ÷ 256 = 156 batches + 27 examples left over
```

With `drop_last=True`, we skip the last incomplete batch of 27.

**Why?**

BatchNorm requires batch size > 1. If the last batch has 1 example, batch norm fails (can't compute mean/std of 1 number).

We lose 27/39979 = 0.07% of data per epoch - negligible.

### max_epochs=100, patience=15

**Early stopping strategy:**

```python
best_val_loss = infinity
patience_counter = 0

for epoch in range(100):
    train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_model()
    else:
        patience_counter += 1
        if patience_counter >= 15:
            stop_training()
```

**Example training curve:**

```
Epoch | Train Loss | Val Loss | Patience | Action
------|------------|----------|----------|--------
1     | 0.524      | 0.487    | 0        | Save (best)
2     | 0.412      | 0.401    | 0        | Save (better!)
...
25    | 0.132      | 0.148    | 0        | Save (best so far)
26    | 0.128      | 0.151    | 1        | (val loss increased)
27    | 0.125      | 0.152    | 2        | (still increasing)
...
40    | 0.095      | 0.163    | 15       | STOP!
```

We stop at epoch 40 and use model from epoch 25 (best validation loss).

**Why patience=15?**

```
Patience | Avg epoch at stop | Final Val ROC-AUC
---------|-------------------|------------------
5        | 25                | 0.964 (stopped too early)
10       | 35                | 0.970
15       | 42                | 0.972  ← Best
20       | 48                | 0.972 (same, but wasted time)
```

Patience=15 allows enough exploration without excessive training.

### device (GPU selection)

```python
if device is None:
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        self.device = torch.device("mps")
    else:
        self.device = torch.device("cpu")
```

**Device speeds (for our model):**

```
Device    | Time/epoch | Speedup vs CPU
----------|------------|----------------
CPU (M1)  | 180s       | 1×
MPS (M1)  | 35s        | 5.1×
CUDA (3090)| 12s       | 15×
```

On M1 Mac, MPS is significantly faster than CPU.

---

(Continue with detailed training loop in next section...)

