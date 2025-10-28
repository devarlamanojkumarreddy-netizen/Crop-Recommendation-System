import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Handle missing data using KNN imputation(method of filling in missing values in a dataset using the K-Nearest Neighbors algorithm)
imputer = KNNImputer(n_neighbors=5)
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# Encode categorical labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Apply PCA for feature reduction (keep 95% variance)
pca = PCA(n_components=0.95)
X_resampled = pca.fit_transform(X_resampled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define improved deep learning model(feedforward neural networks for multiclass classification)
class CropClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CropClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.layer4 = nn.Linear(64, num_classes)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.layer4(x)  # No softmax, since CrossEntropyLoss applies it automatically
        return x

# Initialize model
input_size = X_train.shape[1]
num_classes = len(np.unique(y))
model = CropClassifier(input_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Train the model
epochs = 100
batch_size = 64

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

for epoch in range(epochs):
    model.train()
    
    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Learning rate adjustment
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'crop_classifier_improved.pth')
print("Improved Model saved successfully!")

