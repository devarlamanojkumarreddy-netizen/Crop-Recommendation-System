# ========== Import Libraries ==========
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.theme import Theme

# ========== Load Dataset ==========
df = pd.read_csv("Crop_recommendation.csv")

# ========== Basic Info ==========
print(df.head())
print(df.tail())
print(df.shape)
print(df.describe())
print(df.columns)
print(df.info())
print(df.dtypes)

df['label'].value_counts()
df['label'].nunique()

print("Unique Crops:", df['label'].unique())
print("Missing Values:\n", df.isnull().sum())

print("Number of various crops: ", len(df['label'].unique()))
print("List of crops: ", df['label'].unique())

# ========== Feature Correlation Heatmap ==========
plt.figure(figsize=(12, 8))
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# ========== Skewness Analysis & Box Plots ==========
numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
print("\nSkewness Analysis:")
for feature in numerical_features:
    print(f"{feature}: {df[feature].skew():.3f}")
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.show()

# ========== Pair Plot ==========
sns.pairplot(df[numerical_features + ['label']], hue='label', diag_kind='kde', palette='husl')
plt.suptitle("Pair Plot of Crop Features", y=1.02)
plt.show()

# ========== Crop Distribution ==========
plt.figure(figsize=(10, 10))
label_counts = df['label'].value_counts()
plt.pie(label_counts, labels=label_counts.index, autopct="%.2f%%", startangle=125)
plt.title('Crop Type Distribution')
plt.show()

# ========== Outlier Removal ==========
Q1 = df[numerical_features].quantile(0.25)
Q3 = df[numerical_features].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numerical_features] < (Q1 - 1.5 * IQR)) | (df[numerical_features] > (Q3 + 1.5 * IQR))).any(axis=1)]

# ========== Feature Ranges by Crop ==========
def print_feature_ranges(df, numerical_features):
    custom_theme = Theme({
        "info": "bright_yellow",
        "title": "bright_white on blue",
        "crop": "bright_green",
        "feature": "bright_cyan",
        "header": "bright_magenta",
        "border": "bright_blue",
        "stats": "bright_yellow",
        "highlight": "bright_white",
        "value": "bright_green"
    })
    console = Console(theme=custom_theme)
    title = Text("🌾 FEATURE RANGES BY CROP 🌾", style="title")
    console.print("\n")
    console.print(Panel(title, box=box.DOUBLE, padding=(1, 30), style="border", title="Analysis Report"))
    console.print("\n")
    for crop in sorted(df['label'].unique()):
        crop_data = df[df['label'] == crop]
        crop_title = Text(f"\n📊 Crop: {crop.upper()}", style="crop")
        console.print(crop_title)
        table = Table(show_header=True, header_style="header", box=box.ROUNDED, border_style="border", show_lines=True,
                      title=f"Statistics for {crop.upper()}", title_style="highlight")
        table.add_column("Feature", style="feature")
        table.add_column("Min", justify="right", style="value")
        table.add_column("Max", justify="right", style="value")
        table.add_column("Range", justify="right", style="value")
        table.add_column("Mean", justify="right", style="value")
        for feature in numerical_features:
            min_val = crop_data[feature].min()
            max_val = crop_data[feature].max()
            mean_val = crop_data[feature].mean()
            range_val = max_val - min_val
            table.add_row(feature, f"{min_val:.2f}", f"{max_val:.2f}", f"{range_val:.2f}", f"{mean_val:.2f}")
        console.print(Panel(table, border_style="border"))
        console.print("\n" + "─" * 80 + "\n")

print_feature_ranges(df, numerical_features)




