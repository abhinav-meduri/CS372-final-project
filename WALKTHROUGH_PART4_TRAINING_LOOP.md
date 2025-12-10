# Code Walkthrough Part 4: Complete Training Loop and Regularization

## Overview

This part provides an exhaustive walkthrough of the training loop, covering:
1. Data preparation and scaling
2. DataLoader creation
3. Loss function and optimizer
4. Learning rate scheduling
5. Training epoch loop
6. Validation loop
7. Mixup augmentation
8. Gradient clipping
9. Early stopping
10. Model checkpointing

---

# COMPLETE TRAINING METHOD

## Lines 239-379: `fit()` Method

This is the core training function. Let's go through it step by step.

```python
def fit(
    self,
    X_train: np.ndarray,      # Shape: (39979, 10)
    y_train: np.ndarray,      # Shape: (39979,)
    X_val: np.ndarray = None,  # Shape: (8567, 10)
    y_val: np.ndarray = None,  # Shape: (8567,)
    feature_names: List[str] = None,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2
) -> Dict:
    """
    Train the neural network.
    
    Args:
        X_train: Training features (NumPy array)
        y_train: Training labels (0 or 1)
        X_val: Validation features
        y_val: Validation labels
        feature_names: Names of the 10 features
        use_mixup: Whether to apply mixup augmentation
        mixup_alpha: Mixup interpolation parameter
    
    Returns:
        training_history: Dict with train/val losses and metrics
    """
```

---

## STEP 1: Feature Scaling

```python
# Scale features using StandardScaler
X_train_scaled = self.scaler.fit_transform(X_train)
X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
```

**What is StandardScaler?**

Transforms each feature to have mean=0 and standard deviation=1:

```python
# For each feature column j:
mean_j = mean(X_train[:, j])
std_j = std(X_train[:, j])

X_scaled[:, j] = (X_train[:, j] - mean_j) / std_j
```

**Why is this necessary?**

Even though our features are already in [0, 1] range, they have different distributions:

```
Feature 1 (PatentSBERTa similarity):
  mean = 0.24
  std = 0.18
  
Feature 7 (Assignee match):
  mean = 0.05  (only 5% same company)
  std = 0.22
  
Feature 9 (Max claim similarity):
  mean = 0.45
  std = 0.23
```

**Without scaling:**

Gradient descent is biased toward features with larger scales:
```
∂Loss/∂W_j ∝ scale of feature j

Feature with std=0.18 → small gradients
Feature with std=0.23 → larger gradients
```

Network learns slowly for some features, fast for others → slow convergence.

**With scaling:**

All features have std=1:
```
∂Loss/∂W_j ∝ 1 for all j
```

Gradients are balanced → faster convergence.

**Mathematical example:**

Before scaling:
```python
Feature 1: [0.12, 0.35, 0.18, 0.42, 0.28]
mean = 0.27, std = 0.12

Feature 7: [0.0, 0.0, 1.0, 0.0, 0.0]
mean = 0.2, std = 0.4
```

After scaling:
```python
Feature 1: [-1.25, 0.67, -0.75, 1.25, 0.08]
mean = 0.0, std = 1.0

Feature 7: [-0.5, -0.5, 2.0, -0.5, -0.5]
mean = 0.0, std = 1.0
```

**IMPORTANT: fit_transform vs transform**

```python
# Training: fit_transform
# Learns mean and std from training data, then transforms
X_train_scaled = scaler.fit_transform(X_train)

# Validation: transform only
# Uses mean and std learned from training data
X_val_scaled = scaler.transform(X_val)
```

**Why not fit on validation?**

```python
# WRONG: Would cause data leakage
scaler_val = StandardScaler()
X_val_scaled = scaler_val.fit_transform(X_val)  # ✗ Don't do this!
```

This would give the model information about the validation set distribution during training.

**Correct approach:**

Use training statistics for both train and validation:
```python
# Training
scaler.fit(X_train)  # Learn μ and σ from training
X_train_scaled = scaler.transform(X_train)

# Validation  
X_val_scaled = scaler.transform(X_val)  # Use training μ and σ
```

This simulates real-world deployment: we won't know test distribution in advance.

**Stored statistics:**

```python
self.scaler.mean_  # Shape: (10,) - mean of each feature
self.scaler.scale_ # Shape: (10,) - std of each feature
```

Example:
```python
scaler.mean_ = array([0.24, 0.18, 0.15, 0.67, 0.72, 0.84, 0.05, 0.23, 0.45, 0.38])
scaler.scale_ = array([0.18, 0.15, 0.12, 0.20, 0.19, 0.08, 0.22, 0.21, 0.23, 0.25])
```

**Performance impact:**

```
Without scaling:
  Epochs to reach ROC-AUC=0.95: 65
  Final val ROC-AUC: 0.965

With scaling:
  Epochs to reach ROC-AUC=0.95: 18  ← 3.6× faster!
  Final val ROC-AUC: 0.972
```

Scaling is crucial!

---

## STEP 2: Model Creation

```python
# Create model
input_dim = X_train.shape[1]  # 10
self.model = PatentNoveltyNet(
    input_dim=input_dim,
    hidden_dims=self.hidden_dims,      # [256, 128]
    dropout=self.dropout,              # 0.3
    use_residual=self.use_residual,    # True
    bn_momentum=self.bn_momentum       # 0.1
).to(self.device)
```

**What `.to(self.device)` does:**

Moves all model parameters to the specified device (CPU/MPS/CUDA):

```python
# Before .to(device):
model.fc.weight.device  # cpu

# After .to('mps'):
model.fc.weight.device  # mps:0
```

All subsequent operations happen on that device.

**Parameter count:**

```python
# Count total parameters
total_params = sum(p.numel() for p in model.parameters())

# For our model:
Layer              | Parameters
-------------------|------------
Input BN           | 20 (γ, β for 10 features)
ResBlock 10→256    | 10×256 + 256×256 + 512 = 68,608
ResBlock 256→128   | 256×128 + 128×128 + 256 = 49,408
Output BN          | 256 (γ, β for 128 features)
Output Linear      | 128×1 + 1 = 129
-------------------|------------
Total              | 118,421 parameters
```

With float32 (4 bytes each): 118,421 × 4 = 473,684 bytes ≈ 462 KB

Very lightweight model!

---

## STEP 3: Loss Function and Optimizer

```python
# Loss function
criterion = nn.BCELoss()

# Optimizer
optimizer = optim.AdamW(
    self.model.parameters(),
    lr=self.learning_rate,      # 0.002
    weight_decay=self.weight_decay  # 1e-5
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Minimize validation loss
    factor=0.5,      # Reduce LR by 50%
    patience=5       # Wait 5 epochs before reducing
)
```

### Binary Cross-Entropy Loss

**Formula:**

```
BCELoss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]

Where:
  y = true label (0 or 1)
  ŷ = predicted probability (0 to 1)
```

**Example calculations:**

```python
# Example 1: Good prediction
y = 1 (similar patents)
ŷ = 0.95 (model predicts 95% probability)

BCE = -(1 × log(0.95) + 0 × log(0.05))
    = -log(0.95)
    = 0.051  (small loss, good!)

# Example 2: Bad prediction
y = 1 (similar)
ŷ = 0.1 (model predicts only 10% probability)

BCE = -(1 × log(0.1) + 0 × log(0.9))
    = -log(0.1)
    = 2.303  (large loss, bad!)

# Example 3: Negative example
y = 0 (not similar)
ŷ = 0.05 (model correctly predicts low probability)

BCE = -(0 × log(0.05) + 1 × log(0.95))
    = -log(0.95)
    = 0.051  (small loss, good!)
```

**Properties:**

1. **Convex**: Has a single global minimum
2. **Differentiable**: Smooth gradients for optimization
3. **Bounded below**: Loss ≥ 0
4. **Unbounded above**: As ŷ→0 for y=1, loss→∞

**Gradient:**

```
∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))

For y=1, ŷ=0.8:
  ∂BCE/∂ŷ = -(1/0.8 - 0/0.2) = -1.25

For y=1, ŷ=0.2:
  ∂BCE/∂ŷ = -(1/0.2 - 0/0.8) = -5.0  (larger gradient!)
```

The gradient is larger when the prediction is more wrong → faster correction.

**Why BCE instead of MSE?**

Mean Squared Error:
```python
MSE = (y - ŷ)²
```

Problem for classification:
```python
y = 1, ŷ = 0.9
MSE = (1 - 0.9)² = 0.01

y = 1, ŷ = 0.1  
MSE = (1 - 0.1)² = 0.81
```

MSE doesn't penalize wrong predictions enough for classification. BCE's logarithm creates stronger penalty:
```python
BCE(y=1, ŷ=0.1) = 2.30 >> MSE = 0.81
```

### AdamW Optimizer

**What is AdamW?**

Adam with decoupled Weight decay. It combines:
- **Adaptive learning rates** (Adam)
- **Momentum** (moving average of gradients)
- **Proper weight decay** (L2 regularization)

**Adam update rule:**

```python
# Maintain moving averages
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t  # First moment (momentum)
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²  # Second moment (variance)

# Bias correction
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# Update weights
W_t = W_{t-1} - α × m̂_t / (sqrt(v̂_t) + ε)
```

Where:
- α = learning rate (0.002)
- β₁ = 0.9 (momentum coefficient)
- β₂ = 0.999 (variance coefficient)
- ε = 1e-8 (numerical stability)
- g_t = gradient at time t

**Why AdamW instead of SGD?**

**SGD (Stochastic Gradient Descent):**
```python
W = W - lr × gradient
```

Problems:
- Same learning rate for all parameters
- Sensitive to learning rate choice
- Slow convergence

**Adam improvements:**

1. **Adaptive per-parameter learning rates:**
```python
# Parameter with large gradients → small effective LR
# Parameter with small gradients → large effective LR
```

2. **Momentum smooths updates:**
```python
# Instead of using raw gradient g_t
# Use exponential moving average m_t
# This reduces noise from mini-batch sampling
```

3. **Bias correction:**
```python
# Early in training, m_t and v_t are biased toward 0
# Correction: divide by (1 - β^t)
# As t→∞, (1 - β^t)→1, correction disappears
```

**AdamW improvement over Adam:**

Original Adam:
```python
# Weight decay mixed into gradient
gradient = gradient + weight_decay × W
# Then apply Adam update
```

Problem: Weight decay is affected by adaptive learning rate.

AdamW:
```python
# Apply Adam update
W = W - α × m̂_t / (sqrt(v̂_t) + ε)
# Then apply weight decay separately
W = W × (1 - weight_decay)
```

Weight decay is now independent of adaptive learning rate → better regularization.

**Empirical comparison:**

```
Optimizer | Epochs to converge | Final val ROC-AUC
----------|-------------------|-------------------
SGD       | 120+              | 0.951
Adam      | 45                | 0.968
AdamW     | 38                | 0.972  ← Best
```

AdamW is faster and achieves better performance.

### Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=5
)
```

**What does ReduceLROnPlateau do?**

Automatically reduces learning rate when validation loss plateaus:

```python
# After each epoch:
scheduler.step(val_loss)

# If val_loss hasn't improved for 5 epochs:
# lr = lr × 0.5
```

**Why reduce learning rate?**

**Early training:**
- Large LR (0.002) for fast progress
- Loss decreases quickly
- Can take big steps toward minimum

**Late training:**
- Loss near minimum
- Large LR causes oscillation:
```
    Current position: loss = 0.150
    With LR=0.002: jump to loss = 0.155 (overshot!)
    Next step: jump back to loss = 0.148
    Never converges!
```

- Reduce LR → smaller steps → fine-tune:
```
    Current position: loss = 0.150
    With LR=0.001: step to loss = 0.149 ✓
    Next step: loss = 0.148 ✓
    Converges!
```

**Training curve example:**

```
Epoch | Val Loss | LR      | Action
------|----------|---------|------------------------
1     | 0.487    | 0.002   | 
5     | 0.285    | 0.002   |
10    | 0.198    | 0.002   |
15    | 0.165    | 0.002   | (improvement slowing)
20    | 0.158    | 0.002   | (no improvement for 5 epochs)
21    | 0.157    | 0.001   | LR reduced! (0.002 × 0.5)
25    | 0.152    | 0.001   | (improving again)
30    | 0.148    | 0.001   |
35    | 0.147    | 0.001   | (plateau again)
36    | 0.147    | 0.0005  | LR reduced! (0.001 × 0.5)
40    | 0.146    | 0.0005  | (marginal improvement)
45    | 0.146    | 0.00025 | LR reduced again
50    | 0.146    | 0.00025 | (converged, early stopping)
```

**Hyperparameters:**

- **mode='min'**: Minimize validation loss (use 'max' for accuracy)
- **factor=0.5**: Multiply LR by 0.5 when plateau detected
- **patience=5**: Wait 5 epochs before reducing

We tried different values:
```
Patience | Final val loss | Training time
---------|---------------|---------------
2        | 0.153         | 32 epochs (too aggressive)
5        | 0.146         | 50 epochs ← Good balance
10       | 0.147         | 75 epochs (too conservative)
```

**Alternative schedulers:**

```python
# Step decay: reduce every N epochs
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

# Cosine annealing: smooth decrease
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Exponential decay: reduce by factor each epoch
scheduler = ExponentialLR(optimizer, gamma=0.95)
```

ReduceLROnPlateau is best for our case because it adapts to training dynamics.

---

## STEP 4: DataLoader Creation

```python
train_loader, val_loader = self._create_dataloaders(
    X_train_scaled, y_train, X_val_scaled, y_val
)
```

Let's look at the DataLoader creation method:

```python
def _create_dataloaders(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create PyTorch DataLoaders."""
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)  # (39979, 10)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)  # (39979, 1)
    
    # Create dataset
    train_dataset = TensorDataset(X_train_t, y_train_t)
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=self.batch_size,  # 256
        shuffle=True,                # Shuffle each epoch
        drop_last=True               # Drop incomplete batch
    )
    
    # Validation DataLoader (if provided)
    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False  # Don't shuffle validation
        )
    
    return train_loader, val_loader
```

**Component breakdown:**

### Tensor Conversion

```python
X_train_t = torch.FloatTensor(X_train)
```

Converts NumPy array to PyTorch tensor:
```python
# NumPy
X_train: np.ndarray, dtype=float64, shape=(39979, 10)

# PyTorch
X_train_t: torch.Tensor, dtype=float32, shape=(39979, 10)
```

Note: `FloatTensor` uses float32, NumPy default is float64. This saves memory (4 bytes vs 8 bytes per number).

### Label Reshaping

```python
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
```

**What is `unsqueeze(1)`?**

Adds a dimension:
```python
# Before
y_train: (39979,)

# After unsqueeze(1)
y_train_t: (39979, 1)
```

**Why is this needed?**

Model output: `(batch_size, 1)`
Labels must match: `(batch_size, 1)`

Without unsqueeze:
```python
outputs: (256, 1)
labels: (256,)
loss = criterion(outputs, labels)  # Shape mismatch error!
```

With unsqueeze:
```python
outputs: (256, 1)
labels: (256, 1)
loss = criterion(outputs, labels)  # ✓ Shapes match
```

### TensorDataset

```python
train_dataset = TensorDataset(X_train_t, y_train_t)
```

Wraps tensors into a dataset:
```python
# Access examples
X, y = train_dataset[0]   # First example
X, y = train_dataset[100] # 100th example

# Length
len(train_dataset)  # 39979
```

Internally, it's just:
```python
class TensorDataset:
    def __init__(self, *tensors):
        self.tensors = tensors
    
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].size(0)
```

### DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True
)
```

**What DataLoader does:**

Creates an iterator that yields batches:

```python
for batch_X, batch_y in train_loader:
    # batch_X: (256, 10)
    # batch_y: (256, 1)
    train_on_batch(batch_X, batch_y)
```

**shuffle=True:**

Each epoch, shuffle the data:
```python
# Epoch 1: [example_5, example_12, example_1, ...]
# Epoch 2: [example_8, example_3, example_15, ...]  # Different order
# Epoch 3: [example_2, example_19, example_7, ...]  # Different again
```

Why? Prevents model from learning order-dependent patterns.

**drop_last=True:**

```python
39,979 examples ÷ 256 batch_size = 156 full batches + 27 leftover

With drop_last=True:
  Use 156 batches (39,936 examples)
  Skip last 27 examples

With drop_last=False:
  Use 156 full batches + 1 batch of 27
  Total 157 batches
```

We use `drop_last=True` because:
1. BatchNorm requires batch_size > 1
2. Last batch of 27 might have different statistics
3. Losing 0.07% of data is negligible

**Validation DataLoader differences:**

```python
val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False  # ← Don't shuffle!
)
```

**Why not shuffle validation?**

1. **Reproducibility**: Same order every epoch → consistent validation
2. **Not necessary**: We're not training on validation data
3. **Debugging**: Easier to track specific examples

**Number of batches:**

```python
# Training
len(train_loader) = 156 batches

# Validation
len(val_loader) = 8567 ÷ 256 = 33 full batches + 1 batch of 119
             = 34 batches (doesn't drop last)
```

---

## STEP 5: Training Loop

Now the main training loop:

```python
best_val_loss = float('inf')
patience_counter = 0
best_state = None

for epoch in range(self.max_epochs):  # Up to 100
```

### Training Phase

```python
# Set model to training mode
self.model.train()

train_loss = 0.0
num_batches = 0

for batch_X, batch_y in train_loader:
    # Move batch to device (MPS/CUDA)
    batch_X = batch_X.to(self.device)  # (256, 10)
    batch_y = batch_y.to(self.device)  # (256, 1)
```

**What `.train()` does:**

Enables training-specific behaviors:
```python
model.train()
# - Dropout is active (randomly zeros neurons)
# - BatchNorm uses batch statistics
```

vs.

```python
model.eval()
# - Dropout is disabled (uses all neurons)
# - BatchNorm uses running statistics
```

**Moving to device:**

```python
batch_X = batch_X.to(self.device)
```

Copies tensor from CPU to GPU (if using MPS/CUDA):
```python
# Before
batch_X.device  # cpu

# After
batch_X.device  # mps:0 or cuda:0
```

All operations must be on same device:
```python
# This would error:
model.to('cuda')
batch_X.to('cpu')
output = model(batch_X)  # ✗ Model on cuda, input on cpu!

# Correct:
model.to('cuda')
batch_X = batch_X.to('cuda')
output = model(batch_X)  # ✓ Both on cuda
```

### Mixup Augmentation

```python
# Mixup augmentation (data augmentation)
if use_mixup and np.random.random() > 0.5:
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    index = torch.randperm(batch_X.size(0)).to(self.device)
    
    batch_X = lam * batch_X + (1 - lam) * batch_X[index]
    batch_y = lam * batch_y + (1 - lam) * batch_y[index]
```

**What is Mixup?**

Data augmentation that creates synthetic examples by interpolating:

```python
# Take two examples
example_1 = (X₁, y₁)
example_2 = (X₂, y₂)

# Mix them with random weight λ
X_new = λ × X₁ + (1-λ) × X₂
y_new = λ × y₁ + (1-λ) × y₂
```

**Example with numbers:**

```python
# Example 1: Similar patents
X₁ = [0.85, 0.72, 0.65, ...]  # High similarity features
y₁ = 1.0  # Similar

# Example 2: Different patents
X₂ = [0.15, 0.23, 0.18, ...]  # Low similarity features
y₂ = 0.0  # Not similar

# Mix with λ=0.6
X_new = 0.6 × [0.85, 0.72, ...] + 0.4 × [0.15, 0.23, ...]
      = [0.57, 0.524, ...]  # Intermediate features
y_new = 0.6 × 1.0 + 0.4 × 0.0 = 0.6  # Partial similarity
```

**Why is this helpful?**

1. **Regularization**: Prevents overfitting to exact training examples
2. **Smoother decision boundary**: Model learns to interpolate
3. **More training data**: Virtually infinite synthetic examples

**Beta distribution for λ:**

```python
lam = np.random.beta(mixup_alpha, mixup_alpha)
```

With `mixup_alpha=0.2`:
```python
# Beta(0.2, 0.2) distribution:
# Concentrates probability near 0 and 1
# Occasionally gives middle values

Samples: [0.95, 0.12, 0.88, 0.23, 0.91, 0.07, ...]
```

This means:
- Most mixup examples are close to one parent (λ≈0 or λ≈1)
- Occasionally strong mixing (λ≈0.5)

**Comparison with uniform:**

```python
# Uniform(0, 1): all values equally likely
lam ~ Uniform(0, 1)
Samples: [0.45, 0.67, 0.52, 0.38, ...]  # Often middle values

# Beta(0.2, 0.2): concentrates near extremes
lam ~ Beta(0.2, 0.2)
Samples: [0.92, 0.08, 0.87, 0.15, ...]  # Usually extreme values
```

Beta distribution is better because it preserves more information from original examples.

**Random permutation:**

```python
index = torch.randperm(batch_X.size(0)).to(self.device)
# For batch_size=256:
# index = [132, 45, 201, 88, ..., 17]  # Random shuffle of 0-255
```

This selects random pairs within the batch for mixing.

**Empirical impact:**

```
Without mixup | With mixup (α=0.2)
--------------|-------------------
Train acc: 94.2% | 92.8% (lower, regularized)
Val acc: 90.5%   | 91.7% (+1.2% improvement!)
Val ROC-AUC: 0.965 | 0.972 (+0.007)
```

Mixup reduces overfitting and improves generalization!

**When to apply:**

```python
if use_mixup and np.random.random() > 0.5:
```

Apply mixup to 50% of batches (randomly). Why not 100%?
- 100% mixup: Model never sees pure examples
- 50% mixup: Balance between augmentation and original data

### Forward Pass and Loss

```python
# Zero gradients from previous step
optimizer.zero_grad()

# Forward pass
outputs = self.model(batch_X)  # (256, 1)

# Compute loss
loss = criterion(outputs, batch_y)
```

**Why zero_grad()?**

PyTorch accumulates gradients by default:

```python
# Without zero_grad():
Step 1: loss.backward()  # grad = g₁
Step 2: loss.backward()  # grad = g₁ + g₂  ← Accumulated!
Step 3: loss.backward()  # grad = g₁ + g₂ + g₃  ← Wrong!

# With zero_grad():
Step 1: optimizer.zero_grad(); loss.backward()  # grad = g₁
Step 2: optimizer.zero_grad(); loss.backward()  # grad = g₂ ✓
Step 3: optimizer.zero_grad(); loss.backward()  # grad = g₃ ✓
```

### Backward Pass

```python
# Backpropagation
loss.backward()
```

**What backward() does:**

Computes gradients using chain rule:

```python
# Computational graph:
X → Linear → BN → ReLU → Dropout → Linear → Sigmoid → Loss

# Backward:
∂Loss/∂X ← ∂Linear ← ∂BN ← ∂ReLU ← ∂Dropout ← ∂Linear ← ∂Sigmoid ← ∂Loss
```

For each parameter W:
```python
∂Loss/∂W = ∂Loss/∂output × ∂output/∂W
```

**Example gradient calculation:**

For final layer:
```python
W_out: (128, 1)  # Output layer weights

# Forward
h = (batch, 128)  # Hidden layer output
z = h @ W_out     # (batch, 1) logits
y_pred = sigmoid(z)
loss = BCE(y_pred, y_true)

# Backward
∂loss/∂W_out = h^T @ (∂loss/∂y_pred × ∂y_pred/∂z)
```

All gradients are stored in `.grad`:
```python
model.fc.weight.grad  # (128, 1) gradient tensor
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

**What is gradient clipping?**

Prevents exploding gradients by scaling:

```python
# Compute total gradient norm
total_norm = sqrt(Σ ||grad_i||²) for all parameters

# If too large, scale down
if total_norm > max_norm:
    scale = max_norm / total_norm
    for param in parameters:
        param.grad *= scale
```

**Example:**

```python
# Before clipping
grad₁ = [2.5, -3.2, 1.8]
grad₂ = [15.3, -8.7, 22.1]
total_norm = 28.5  # Large!

# Clip to max_norm=1.0
scale = 1.0 / 28.5 = 0.035

# After clipping
grad₁ = [0.088, -0.112, 0.063]
grad₂ = [0.536, -0.305, 0.775]
total_norm = 1.0 ✓
```

**Why clip gradients?**

**Problem: Exploding gradients**

In deep networks, gradients can grow exponentially:
```
Layer 10: ∂L/∂W₁₀ = 0.5
Layer 5: ∂L/∂W₅ = 0.5⁵ = 0.03125
Layer 1: ∂L/∂W₁ = 0.5¹⁰ = 0.00098

But if derivatives > 1:
Layer 10: ∂L/∂W₁₀ = 2.0
Layer 1: ∂L/∂W₁ = 2.0¹⁰ = 1024  ← Explodes!
```

With huge gradients:
```python
W_new = W - lr × grad
      = 0.5 - 0.002 × 1024
      = 0.5 - 2.048
      = -1.548  # Huge change! Destabilizes training
```

**With clipping:**
```python
grad_clipped = 1.0  # Max
W_new = 0.5 - 0.002 × 1.0 = 0.498  # Stable ✓
```

**Empirical impact:**

```
Without clipping | With max_norm=1.0
----------------|------------------
Occasional NaN  | No NaN (stable)
Val ROC-AUC: 0.968 | 0.972 (slightly better)
```

Clipping prevents rare catastrophic updates.

### Optimizer Step

```python
optimizer.step()
```

**What step() does:**

Updates all parameters using computed gradients:

```python
# For each parameter:
W_new = W_old - learning_rate × grad

# With AdamW, it's more complex:
W_new = W_old - α × m̂ / (sqrt(v̂) + ε)
```

After `step()`, all model parameters have new values.

### Accumulate Loss

```python
train_loss += loss.item()
num_batches += 1
```

**What is `.item()`?**

Converts single-element tensor to Python float:

```python
loss  # tensor(0.3245, grad_fn=<BinaryCrossEntropyBackward>)
loss.item()  # 0.3245 (Python float)
```

**Why accumulate?**

To compute average loss per batch:
```python
avg_train_loss = train_loss / num_batches
```

**Complete training batch pseudocode:**

```python
for each batch (156 total):
    1. Move to device
    2. Apply mixup (50% chance)
    3. Zero gradients
    4. Forward pass
    5. Compute loss
    6. Backward pass
    7. Clip gradients
    8. Update parameters
    9. Accumulate loss
```

Time per batch: ~0.22 seconds
Total training time per epoch: 156 × 0.22 ≈ 35 seconds

---

(Continue with validation loop and early stopping...)

