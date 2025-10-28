import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("Crop_recommendation.csv")
features = df.drop("label", axis=1)
labels = df["label"]

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X = torch.tensor(features_scaled, dtype=torch.float32)
y = torch.tensor(labels_encoded, dtype=torch.long)

# Split data
train_size = int(0.8 * len(X))
val_size = len(X) - train_size

X_train, X_val = torch.utils.data.random_split(TensorDataset(X, y), [train_size, val_size])
train_loader = DataLoader(X_train, batch_size=32, shuffle=True)
val_loader = DataLoader(X_val, batch_size=32)

# Model
class CropClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CropClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = CropClassifier(input_size=X.shape[1], hidden_size=128, num_classes=len(le.classes_))

# Compute class weights
y_train_tensor = y[list(X_train.indices)]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_tensor), y=y_train_tensor.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training with early stopping
best_val_loss = float('inf')
patience = 20
epochs_no_improve = 0
max_epochs = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, max_epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_accuracy = 100 * correct / total
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == y_batch).sum().item()
            total_val += y_batch.size(0)

    val_accuracy = 100 * correct_val / total_val
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch}/{max_epochs}] - Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pt")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"\nEarly stopping on epoch {epoch}")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pt"))

# Final evaluation
model.eval()
X_test_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(labels_encoded, dtype=torch.long).to(device)

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    final_accuracy = 100 * correct / total

print(f"\nFinal Model Accuracy: {final_accuracy:.2f}%")

# Classification report with zero_division=0 to suppress warnings
print("\nClassification Report:\n")
print(classification_report(y_test_tensor.cpu(), predicted.cpu(), target_names=le.classes_, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test_tensor.cpu(), predicted.cpu())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
