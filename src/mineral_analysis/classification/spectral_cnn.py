import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Optional
from collections import Counter
# Mineral class mapping (10 common minerals)
MINERAL_MAPPING = {
   0 : 'Pyroxene',
   1 : 'Plagioclase',
   2 : 'Olivine',
   3 : 'Ilmenite',
}
num_classes=4
class SpectralCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SpectralCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(3456, 128)  # after 2 pool layers
        self.fc2 = nn.Linear(128, num_classes)

        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        # x: (B, 1, 109)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 32, 54)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, 64, 27)
        x = F.relu(self.bn3(self.conv3(x)))             # (B, 128, 27)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))        # (B, 128)
        return self.fc2(x)
    
class SpectrumDataset(Dataset):
    def __init__(self, padded, mask, labels):
        self.padded = torch.tensor(padded, dtype=torch.float32)
        self.mask   = torch.tensor(mask, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        reflectance = self.padded[idx].unsqueeze(0).float()  # (1, L)
        mask = self.mask[idx].float()                        # (L,)
        label = torch.tensor(self.labels[idx]).long()
        return reflectance, mask, label
def augment_spectrum(x, noise_std=0.01, scale_range=(0.95, 1.05)):
    """
    x: numpy array of shape (L,)
    noise_std: Gaussian noise std
    scale_range: multiplicative random scaling
    """
    x_aug = x * np.random.uniform(*scale_range)
    x_aug += np.random.normal(0, noise_std, size=x.shape)
    return np.clip(x_aug, 0, 1)  # reflectance should be 0-1

def oversample_and_augment(reflectances, labels):
    """
    reflectances: list or np.array of shape (N, L)
    labels: list or np.array of shape (N,)
    """
    reflectances = list(reflectances)
    labels = list(labels)
    
    # find class counts
    class_counts = Counter(labels)
    max_count = max(class_counts.values())
    
    new_reflectances = []
    new_labels = []
    
    for cls in class_counts:
        cls_indices = [i for i, l in enumerate(labels) if l == cls]
        cls_samples = [reflectances[i] for i in cls_indices]
        n_to_add = max_count - class_counts[cls]
        
        # duplicate and augment
        for _ in range(n_to_add):
            sample = cls_samples[np.random.randint(len(cls_samples))]
            sample_aug = augment_spectrum(sample)
            new_reflectances.append(sample_aug)
            new_labels.append(cls)
    
    # concatenate original + augmented
    all_reflectances = np.vstack([reflectances, new_reflectances])
    all_labels = np.hstack([labels, new_labels])
    
    # shuffle
    indices = np.arange(len(all_labels))
    np.random.shuffle(indices)
    
    return all_reflectances[indices], all_labels[indices]

from mineral_analysis.classification.relab_data_utils import get_relab_iirs_data
padded, labels, mask = get_relab_iirs_data()

all_reflectance_balanced, all_labels_balanced = oversample_and_augment(padded, labels)
print("Balanced shape:", all_reflectance_balanced.shape, all_labels_balanced.shape)
# Split into 80% train, 20% test
dataset = SpectrumDataset(all_reflectance_balanced, mask, all_labels_balanced)
train_size = int(0.8 * len(dataset))
test_size  = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

labels_np = np.array(labels)
class_counts = np.bincount(labels_np)
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels_np]

# split indices first
num_total = len(labels_np)
train_size = int(0.8 * num_total)
test_size = num_total - train_size

indices = np.arange(num_total)
np.random.shuffle(indices)
train_idx, test_idx = indices[:train_size], indices[train_size:]

# make train/test samplers separately
train_sampler = WeightedRandomSampler(sample_weights[train_idx], num_samples=len(train_idx), replacement=True)
test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
test_loader  = DataLoader(dataset, batch_size=32, sampler=test_sampler)
# -------------------------
# 4️⃣  Training Setup
# -------------------------
def masked_cross_entropy_loss(logits, target, mask):
    """
    logits: (B, num_classes)
    target: (B,)
    mask: (B, L)
    """
    # Reduce mask over spectral dimension to single validity weight per sample
    # (e.g. if 80% of wavelengths are valid, weight that sample by 0.8)
    sample_weights = mask.mean(dim=1)  # (B,)

    ce_loss = F.cross_entropy(logits, target, reduction='none')  # (B,)
    weighted_loss = (ce_loss * sample_weights).mean()
    return weighted_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(set(labels))

model = SpectralCNN(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

def train_test_loop(model):
    num_epochs = 30
    train_losses, test_losses = [], []

    # -------------------------
    # 5️⃣  Training Loop
    # -------------------------
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = masked_cross_entropy_loss(logits, y, mask)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_test_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for x, mask, y in test_loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                logits = model(x)
                loss = masked_cross_entropy_loss(logits, y, mask)
                total_test_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        accuracy = 100.0 * correct / total

        print(f"Epoch {epoch+1:02d}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}, Acc={accuracy:.2f}%")

    # -------------------------
    # 6️⃣  Plot Loss Curves
    # -------------------------
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(test_losses, label="Test Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("/teamspace/studios/this_studio/isro-spectral-unmixing/src/mineral_analysis/classification/plots", exist_ok=True)
    plt.savefig(f"/teamspace/studios/this_studio/isro-spectral-unmixing/src/mineral_analysis/classification/plots/loss_curves_{model.__class__.__name__}.png", dpi=300)
    plt.show()

    # -------------------------
    # 7️⃣  Save Model
    # -------------------------
    model_save_path: str = f"/teamspace/studios/this_studio/isro-spectral-unmixing/models/{model.__class__.__name__}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'n_bands': padded.shape[1],
                'n_classes': num_classes
            },
            'mineral_mapping': MINERAL_MAPPING
        }, model_save_path)

    print("Model and loss plot saved.")

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, mask, y in test_loader:
            out = model(x.to(device))
            preds = out.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    print(classification_report(y_true, y_pred, target_names=['Pyroxene','Plagioclase','Olivine','Ilmenite']))
    print(confusion_matrix(y_true, y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=['Pyroxene','Plagioclase','Olivine','Ilmenite'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f"/teamspace/studios/this_studio/isro-spectral-unmixing/src/mineral_analysis/classification/plots/confusion_matrix_{model.__class__.__name__}.png", dpi=300)

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
