import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Optional
import pickle

# Mineral class mapping (10 common minerals)
MINERAL_MAPPING = {
    0: 'quartz',
    1: 'feldspar', 
    2: 'pyroxene',
    3: 'olivine',
    4: 'amphibole',
    5: 'mica',
    6: 'calcite',
    7: 'chlorite',
    8: 'serpentine',
    9: 'hematite'
}

class SpectralCNN(nn.Module):
    """Simple 1D CNN for spectral classification."""
    
    def __init__(self, n_bands: int = 109, n_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        
        # 1D Conv layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate size after convolutions
        conv_size = n_bands // 8  # After 3 pooling layers (2^3 = 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * conv_size, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # Input: (batch_size, n_bands)
        x = x.unsqueeze(1)  # Add channel dim: (batch_size, 1, n_bands)
        
        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


def create_synthetic_usgs_data(n_samples: int = 1000, n_bands: int = 109, 
                              n_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic USGS-like spectral data for demonstration.
    In practice, replace this with actual USGS spectral library loading.
    """
    np.random.seed(42)
    
    # Generate synthetic spectral signatures with different patterns per mineral
    X = []
    y = []
    
    for class_id in range(n_classes):
        class_samples = n_samples // n_classes
        
        # Create base spectral pattern for this mineral class
        base_spectrum = np.random.rand(n_bands) * 0.5 + 0.2
        
        # Add class-specific spectral features
        if class_id == 0:  # quartz - high reflectance
            base_spectrum[40:60] += 0.3
        elif class_id == 1:  # feldspar - absorption around band 70
            base_spectrum[65:75] -= 0.2
        elif class_id == 2:  # pyroxene - characteristic dips
            base_spectrum[30:35] -= 0.15
            base_spectrum[80:85] -= 0.15
        # Add more specific patterns for other minerals...
        
        for _ in range(class_samples):
            # Add noise and variation
            spectrum = base_spectrum + np.random.normal(0, 0.05, n_bands)
            spectrum = np.clip(spectrum, 0, 1)  # Keep in reflectance range
            
            X.append(spectrum)
            y.append(class_id)
    
    return np.array(X), np.array(y)


def train_spectral_cnn(X: np.ndarray, y: np.ndarray, 
                      model_save_path: str = "/teamspace/studios/this_studio/isro-spectral-unmixing/models/spectral_cnn.pth",
                      epochs: int = 50, batch_size: int = 32, lr: float = 0.001) -> SpectralCNN:
    """
    Train the spectral CNN classifier.
    
    Args:
        X: Spectral data (n_samples, n_bands)
        y: Class labels (n_samples,)
        model_save_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
    
    Returns:
        Trained model
    """
    # Create model directory
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Split data (simple train/test)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SpectralCNN(n_bands=X.shape[1], n_classes=len(np.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        if (epoch + 1) % 10 == 0:
            # Test accuracy
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_y.size(0)
                    test_correct += (predicted == batch_y).sum().item()
            
            train_acc = 100 * correct / total
            test_acc = 100 * test_correct / test_total
            print(f'Epoch [{epoch+1}/{epochs}], Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
            model.train()
    
    # Save model and mapping
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_bands': X.shape[1],
            'n_classes': len(np.unique(y))
        },
        'mineral_mapping': MINERAL_MAPPING
    }, model_save_path)
    
    print(f"Model saved to: {model_save_path}")
    return model


def load_model_and_predict(model_path: str, endmembers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load trained model and predict mineral classes for endmembers.
    
    Args:
        model_path: Path to saved model
        endmembers: Endmember spectra (n_endmembers, n_bands)
    
    Returns:
        predicted_classes: Class indices (n_endmembers,)
        probabilities: Class probabilities (n_endmembers, n_classes) 
        mineral_names: Dictionary mapping class indices to names
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    mineral_mapping = checkpoint['mineral_mapping']
    
    # Initialize model
    model = SpectralCNN(
        n_bands=model_config['n_bands'],
        n_classes=model_config['n_classes']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare endmember data
    if endmembers.ndim != 2:
        raise ValueError("Endmembers must be 2D array (n_endmembers, n_bands)")
    
    endmembers_tensor = torch.FloatTensor(endmembers).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(endmembers_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
    
    # Convert to numpy
    predicted_classes = predicted_classes.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # Print results
    print("\nMineral Classification Results:")
    print("-" * 50)
    for i, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
        mineral_name = mineral_mapping[pred_class]
        confidence = probs[pred_class] * 100
        print(f"Endmember {i+1}: {mineral_name} (confidence: {confidence:.1f}%)")
        
        # Show top 3 predictions
        top3_indices = np.argsort(probs)[-3:][::-1]
        print(f"  Top 3: ", end="")
        for j, idx in enumerate(top3_indices):
            name = mineral_mapping[idx]
            prob = probs[idx] * 100
            print(f"{name}({prob:.1f}%)", end="")
            if j < 2:
                print(", ", end="")
        print("\n")
    
    return predicted_classes, probabilities, mineral_mapping


if __name__ == "__main__":
    # Generate synthetic training data
    print("Generating synthetic USGS-like spectral data...")
    X, y = create_synthetic_usgs_data(n_samples=2000, n_bands=109, n_classes=10)
    
    # Train model
    print("Training spectral CNN...")
    model = train_spectral_cnn(X, y)
    
    # Test with some sample endmembers
    print("\nTesting with sample endmembers...")
    test_endmembers = X[:4]  # Use first 4 samples as test endmembers
    
    pred_classes, probs, mapping = load_model_and_predict(
        "/teamspace/studios/this_studio/isro-spectral-unmixing/models/spectral_cnn.pth",
        test_endmembers
    )