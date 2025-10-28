import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px


# Load your dataset (assuming df is already defined)
df = pd.read_csv("Crop_recommendation.csv")

# Define features (soil data) and target (crop type)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LightGBM Classifier
model = lgb.LGBMClassifier(n_estimators=100, random_state=42)

# Perform cross-validation to get the accuracies at each fold
train_accuracies = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
test_accuracies = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')

# Plotting the accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), train_accuracies, label='Train Accuracy', color='blue', marker='o')
plt.plot(range(1, 6), test_accuracies, label='Test Accuracy', color='red', marker='o')
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy for Each Fold')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Evaluate the model's accuracy on the test set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Final accuracy on the test set
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy on Test Set: {final_accuracy * 100:.2f}%")

# Feature plotting function
def feat_plot(feature):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    feature.plot(kind='hist')
    plt.title(f'{feature.name} histogram plot')
    
    plt.subplot(1, 3, 2)
    mu, sigma = scipy.stats.norm.fit(feature)
    sns.displot(feature, kde=True)
    plt.axvline(mu, linestyle='--', color='green')
    plt.axvline(sigma, linestyle='--', color='red')
    plt.title(f'{feature.name} distribution plot')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(feature)
    plt.title(f'{feature.name} box plot')
    plt.show()

# Apply the feature plot for all numerical features(A pair plot (or scatter plot matrix) is a useful visualization to observe the distributions and relationships between each pair of numerical features)
num_feat = df.select_dtypes(exclude='object')
for i in num_feat.columns:
    feat_plot(num_feat[i])

# Outlier removal function using IQR (Interquartile Range)
def remove_outliers(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        cleaned_df = cleaned_df[(cleaned_df[col] >= (Q1 - 1.5 * IQR)) & (cleaned_df[col] <= (Q3 + 1.5 * IQR))]
    return cleaned_df

# Remove outliers from the numerical columns
columns_to_clean = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
cleaned_df = remove_outliers(df, columns_to_clean)

# Box plots before and after removing outliers
fig = go.Figure()

# Add box plots for each feature before removing outliers
for col in columns_to_clean:
    fig.add_trace(go.Box(y=df[col], name=f'Before: {col}'))

# Add box plots for each feature after removing outliers
for col in columns_to_clean:
    fig.add_trace(go.Box(y=cleaned_df[col], name=f'After: {col}'))

# Update layout
fig.update_layout(
    title="Comparison of Numerical Features Before and After Removing Outliers",
    yaxis_title="Value",
    xaxis_title="Feature",
    boxmode='group'
)

# Show plot
fig.show()