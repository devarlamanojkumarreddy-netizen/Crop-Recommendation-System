#Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.theme import Theme
import warnings
warnings.filterwarnings('ignore')

# Set the style and color palette
sns.set_palette("viridis")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 20)))


import pickle
import sklearn
print("Pickle module version:", pickle.format_version)
print("Pickle module version:", sklearn.__version__)


#Read the dataset
df = pd.read_csv("Crop_recommendation.csv")

df.info()


#Display the First Few Rows of the Crop Recommendation Dataset
df.head()

#Check for Missing Values
print(df.isnull().sum())

#Summary Statistics of the Crop Recommendation Dataset
df.describe()

def format_summary_statistics(df):
    """
    Create a professionally formatted summary of statistics by crop for each feature.

    Parameters:
    df (pandas.DataFrame): Input dataframe with crop data

    Returns:
    None: Prints formatted statistics
    """
    # Set pandas display options for better formatting
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    print("\n" + "="*80)
    print("CROP-WISE SUMMARY STATISTICS")
    print("="*80)

    for feature in features:
        # Create summary statistics
        stats_df = df.groupby('label')[feature].describe()

        # Add additional statistics
        stats_df['range'] = stats_df['max'] - stats_df['min']
        stats_df['cv'] = (stats_df['std'] / stats_df['mean']) * 100  # Coefficient of variation

        # Rename columns for better readability
        stats_df = stats_df.rename(columns={
            'count': 'Count',
            'mean': 'Mean',
            'std': 'Std Dev',
            'min': 'Min',
            '25%': '25th Perc',
            '50%': 'Median',
            '75%': '75th Perc',
            'max': 'Max',
            'range': 'Range',
            'cv': 'CV (%)'
        })

        print(f"\n{'-'*80}")
        print(f"Feature: {feature.upper()}")
        print(f"{'-'*80}")

        # Format the DataFrame for display
        styled_df = stats_df.style\
            .format({
                'Count': '{:.0f}',
                'Mean': '{:.2f}',
                'Std Dev': '{:.2f}',
                'Min': '{:.2f}',
                '25th Perc': '{:.2f}',
                'Median': '{:.2f}',
                '75th Perc': '{:.2f}',
                'Max': '{:.2f}',
                'Range': '{:.2f}',
                'CV (%)': '{:.2f}'
            })\
            .background_gradient(cmap='viridis', subset=['Mean', 'Median'])\
            .highlight_max(color='lightgreen', subset=['Max'])\
            .highlight_min(color='lightsalmon', subset=['Min'])

        print(styled_df)

        # Print feature insights
        print("\nKey Insights:")
        print(f"• Highest {feature} requirement: {stats_df['Max'].idxmax()} crop")
        print(f"• Lowest {feature} requirement: {stats_df['Min'].idxmin()} crop")
        print(f"• Most variable {feature} (highest CV): {stats_df['CV (%)'].idxmax()} crop")
        print(f"• Most consistent {feature} (lowest CV): {stats_df['CV (%)'].idxmin()} crop")

# Execute the function
format_summary_statistics(df)

# Reset display options to default
pd.reset_option('display.float_format')
pd.reset_option('display.max_columns')
pd.reset_option('display.width')
format_summary_statistics(df)