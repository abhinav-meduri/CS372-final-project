"""
PyTorch Neural Network Classifier for Patent Novelty

Features:
- Dropout layers for regularization
- Batch normalization for training stability
- Residual connections for better gradient flow
- Learning rate scheduling
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from pathlib import Path
import json
import pickle
from typing import Tuple, Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block with batch norm and dropout.
    
    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    dropout : float, default=0.3
        Dropout probability
    bn_momentum : float, default=0.1
        Batch normalization momentum
    """
    
    def __init__(self, in_features, out_features, dropout=0.3, bn_momentum=0.1):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Skip connection (identity or projection)
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
            self.skip_bn = nn.BatchNorm1d(out_features, momentum=bn_momentum)
        else:
            self.skip = nn.Identity()
            self.skip_bn = None
    
    def forward(self, x):
        identity = self.skip(x)
        if self.skip_bn is not None:
            identity = self.skip_bn(identity)
        
        out = self.fc(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + identity
        return out


class PatentNoveltyNet(nn.Module):
    """Definition of a neural network for patent novelty classification
    in PyTorch, inheriting from the torch.nn.Module base class.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dims : list of int, default=[128, 64, 32]
        Number of hidden units in each layer
    dropout : float, default=0.3
        Dropout probability
    use_residual : bool, default=True
        Whether to use residual connections
    bn_momentum : float, default=0.1
        Batch normalization momentum
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3, use_residual=True, bn_momentum=0.1):
        super(PatentNoveltyNet, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim, momentum=bn_momentum)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if use_residual:
                layers.append(ResidualBlock(prev_dim, hidden_dim, dropout, bn_momentum))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=bn_momentum))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_bn = nn.BatchNorm1d(prev_dim, momentum=bn_momentum)
        self.output_layer = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Compute logits for batch x by forward propagation.
        
        Parameters
        ----------
        x : tensor, shape = [n_examples, n_features]
            Input features
        
        Returns
        -------
        tensor, shape = [n_examples, 1]
            Output probabilities
        """
        x = self.input_bn(x)
        x = self.hidden_layers(x)
        x = self.output_bn(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


class PyTorchPatentClassifier:
    """PyTorch-based patent novelty classifier with modern deep learning techniques.
    
    Parameters
    ----------
    hidden_dims : list of int, default=[128, 64, 32]
        Number of hidden units in each layer
    dropout : float, default=0.3
        Dropout probability
    learning_rate : float, default=0.001
        Learning rate for optimizer
    weight_decay : float, default=1e-4
        L2 regularization parameter
    batch_size : int, default=256
        Minibatch size
    max_epochs : int, default=100
        Maximum number of training epochs
    patience : int, default=15
        Number of epochs to wait before early stopping
    use_residual : bool, default=True
        Whether to use residual connections
    bn_momentum : float, default=0.1
        Batch normalization momentum
    device : str, optional
        Device to use ('cuda', 'mps', or 'cpu'). If None, auto-detects.
    """
    
    def __init__(
        self,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=100,
        patience=15,
        use_residual=True,
        bn_momentum=0.1,
        device=None
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_residual = use_residual
        self.bn_momentum = bn_momentum
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.feature_names = None
        
        logger.info(f"PyTorch Classifier initialized (device: {self.device})")
    
    def _create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create PyTorch DataLoaders."""
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
        use_mixup: bool = True,
        mixup_alpha: float = 0.2
    ) -> Dict:
        """
        Train the model with optional mixup augmentation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Names of features
            use_mixup: Whether to use mixup augmentation
            mixup_alpha: Mixup interpolation strength
        """
        self.feature_names = feature_names
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Create model
        input_dim = X_train.shape[1]
        self.model = PatentNoveltyNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_residual=self.use_residual,
            bn_momentum=self.bn_momentum
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        logger.info(f"Training PyTorch model...")
        logger.info(f"  Architecture: {self.hidden_dims}")
        logger.info(f"  Dropout: {self.dropout}")
        logger.info(f"  Residual connections: {self.use_residual}")
        logger.info(f"  Mixup augmentation: {use_mixup}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Mixup augmentation
                if use_mixup and np.random.random() > 0.5:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    index = torch.randperm(batch_X.size(0)).to(self.device)
                    batch_X = lam * batch_X + (1 - lam) * batch_X[index]
                    batch_y = lam * batch_y + (1 - lam) * batch_y[index]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
            self.training_history["train_loss"].append(avg_train_loss)
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        all_preds.extend(outputs.cpu().numpy())
                        all_labels.extend(batch_y.cpu().numpy())
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = accuracy_score(
                    np.array(all_labels) > 0.5,
                    np.array(all_preds) > 0.5
                )
                
                self.training_history["val_loss"].append(avg_val_loss)
                self.training_history["val_acc"].append(val_acc)
                
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 5 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.max_epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, "
                        f"Val Acc: {val_acc:.4f}"
                    )
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        logger.info("Training complete!")
        
        return self.training_history
    
    def predict_proba(self, X):
        """Predict probabilities for each row in X for each class"""
        """Predict probabilities."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy()
        
        return np.hstack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class for each row in X"""
        """Predict classes."""
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }
    
    def save(self, path: str):
        """Save model and scaler."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "use_residual": self.use_residual,
            "bn_momentum": self.bn_momentum,
            "input_dim": self.model.input_bn.num_features
        }, path / "pytorch_model.pt")
        
        # Save scaler
        with open(path / "scaler_pytorch.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Save training history
        with open(path / "training_history_pytorch.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and scaler."""
        path = Path(path)
        
        # Load model config and weights
        checkpoint = torch.load(path / "pytorch_model.pt", map_location=self.device)
        
        self.hidden_dims = checkpoint["hidden_dims"]
        self.dropout = checkpoint["dropout"]
        self.use_residual = checkpoint["use_residual"]
        self.bn_momentum = checkpoint.get("bn_momentum", 0.1)
        
        self.model = PatentNoveltyNet(
            input_dim=checkpoint["input_dim"],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_residual=self.use_residual,
            bn_momentum=self.bn_momentum
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load scaler
        with open(path / "scaler_pytorch.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    # Quick test
    print("Testing PyTorch classifier...")
    
    # Generate dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 13)
    y_train = (np.random.rand(1000) > 0.5).astype(float)
    X_val = np.random.randn(200, 13)
    y_val = (np.random.rand(200) > 0.5).astype(float)
    
    classifier = PyTorchPatentClassifier(
        hidden_dims=[64, 32],
        dropout=0.3,
        max_epochs=20
    )
    
    classifier.fit(X_train, y_train, X_val, y_val, use_mixup=True)
    metrics = classifier.evaluate(X_val, y_val)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

