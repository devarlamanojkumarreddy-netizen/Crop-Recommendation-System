import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Preprocessed_Crop_Data.csv")

# Feature columns and target
numerical_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[numerical_cols]
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Module definitions
class Module1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, x): return self.model(x)

class Module2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, x): return self.model(x)

class Module3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, x): return self.model(x)

class ModularNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod1 = Module1()
        self.mod2 = Module2()
        self.mod3 = Module3()
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(label_encoder.classes_))
        )
    def forward(self, x1, x2, x3):
        o1 = self.mod1(x1)
        o2 = self.mod2(x2)
        o3 = self.mod3(x3)
        out = torch.cat((o1, o2, o3), dim=1)
        return self.fusion(out)

# Initialize model
model = ModularNN()

# Optimizer, loss, scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)

# Training function
def train_model(model, optimizer, criterion, X_train, y_train, epochs=100, patience=10):
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train[0], X_train[1], X_train[2])
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # Accuracy
        _, preds = torch.max(outputs, 1)
        acc = accuracy_score(y_train.numpy(), preds.numpy())

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}, Accuracy: {acc * 100:.2f}%")

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping on epoch {epoch+1}")
            break

    return model

# Train the model
model = train_model(
    model, optimizer, criterion,
    (X_train[:, 0:3], X_train[:, 3:6], X_train[:, 6:]),
    y_train, epochs=100
)

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test[:, 0:3], X_test[:, 3:6], X_test[:, 6:])
    _, predicted = torch.max(outputs, 1)
    acc = accuracy_score(y_test.numpy(), predicted.numpy())
    print(f"\nFinal Model Accuracy: {acc * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test.numpy(), predicted.numpy())
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test.numpy(), predicted.numpy(), target_names=label_encoder.classes_))

# Save model
torch.save(model.state_dict(), 'best_modular_model_with_lr_scheduler.pth')
print("\nBest model saved successfully!")

# User prediction
def predict_user_input(model, scaler, label_encoder):
    print("\nEnter values to predict the suitable crop:")
    N = float(input("Nitrogen (N): "))
    P = float(input("Phosphorus (P): "))
    K = float(input("Potassium (K): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("pH value: "))
    rainfall = float(input("Rainfall (mm): "))

    # Create input array and scale
    user_input = [[N, P, K, temperature, humidity, ph, rainfall]]
    user_input_scaled = scaler.transform(user_input)

    # Convert to tensor segments
    x1 = torch.tensor(user_input_scaled[:, 0:3], dtype=torch.float32)
    x2 = torch.tensor(user_input_scaled[:, 3:6], dtype=torch.float32)
    x3 = torch.tensor(user_input_scaled[:, 6:], dtype=torch.float32)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(x1, x2, x3)
        _, pred = torch.max(output, 1)
        predicted_crop = label_encoder.inverse_transform(pred.numpy())

    print(f"\n🌾 Recommended Crop: **{predicted_crop[0]}**")

# Run user prediction
predict_user_input(model, scaler, label_encoder)
