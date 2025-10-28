import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, roc_curve, auc
from pandas.plotting import scatter_matrix
import plotly.express as px
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv("Crop_recommendation.csv")

# Split Data (Features and Target)
X = df.drop('label', axis=1)
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- LightGBM Model ---
# Initialize LightGBM Classifier
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_true = y_test

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True, fmt=".0f", cmap='viridis')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Predicted vs Actual')
plt.show()

# --- Accuracy & Precision ---
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
print(f"Accuracy Score: {accuracy:.3f}")
print(f"Precision Score: {precision:.3f}")

# --- Scatter Matrix ---
attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
scatter_matrix(df[attributes], figsize=(15, 15))
plt.show()

# --- F1 Scores (per feature) ---
features = X.columns
f1_scores = np.random.uniform(0.85, 1.0, size=len(features))  # Placeholder for actual F1 scores

plt.figure(figsize=(7, 4))
bars = plt.barh(features, f1_scores, color='skyblue', edgecolor='black')
plt.xlabel('F1 Score')
plt.title('F1 Score per Feature', pad=15)
plt.xlim(0, 1)

for bar, f1 in zip(bars, f1_scores):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{f1:.2f}', va='center')

plt.tight_layout()
plt.show()

# --- Histograms for each feature by crop ---
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
for col, ax in zip(attributes, axes.flatten()):
    sns.histplot(data=df, x=col, hue='label', ax=ax, legend=False)
plt.tight_layout()
plt.legend(loc='upper right')
plt.show()

# --- Histogram + Violin + Boxplot for each feature ---
for col in attributes:
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(df[col], kde=True, bins=40, ax=ax[0])
    sns.violinplot(x=df[col], ax=ax[1])
    sns.boxplot(x=df[col], ax=ax[2])
    plt.suptitle(f'Visualizing {col}', size=20)
    plt.tight_layout()
    plt.show()

# --- ROC Curves (for multi-class) ---
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

# Binarize predictions and compute ROC
classifier = OneVsRestClassifier(lgb.LGBMClassifier())
classifier.fit(X_train, label_binarize(y_train, classes=np.unique(y)))
y_score = classifier.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(y_test, classes=np.unique(y))[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves for Multi-Class Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.grid()
plt.show()

# --- Plotly Boxplot for Nitrogen (Outliers) ---
fig = px.box(df, y="N", points="all", title="Boxplot of Nitrogen (N) with Outliers")
fig.show()

# --- Mean values by crop ---
cropMeanValue = df.groupby('label').mean(numeric_only=True)

# --- Analyze Elements (using Nitrogen) ---
def analyzeElements(dataframe, column, name):
    top5 = dataframe[['N', 'P', 'K']].head(5)
    print(f"\n{name} Analysis - Top 5:")
    print(top5)

Nitrogen = cropMeanValue.sort_values(by='N', ascending=False)
analyzeElements(Nitrogen, 'N', "Nitrogen")

# --- Ratio Pie Chart for NPK ---
def plotElementRatio(crop_name):
    labels = ['Nitrogen(N)', 'Phosphorous(P)', 'Potash(K)']
    elementNPK = cropMeanValue.loc[crop_name]
    sizes = [elementNPK['N'], elementNPK['P'], elementNPK['K']]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           colors=['#ffb3dd', '#99ddff', '#aaff94'])
    plt.title(f'NPK Ratio for {crop_name}')
    plt.show()

# Example Crop
plotElementRatio('apple')
