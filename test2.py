import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore')
# Set the style and color palette
sns.set_palette("viridis")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.viridis(np.linspace(0, 1, 20)))

import pickle
import sklearn
print("Pickle module version:", pickle.format_version)
print("Pickle module version:", sklearn.__version__)





import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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




# Importing Data
df = pd.read_csv("Crop_recommendation.csv")

df.shape

df.head()

df.tail()

df.describe()

df['label'].unique()

df.columns

df.isnull().sum()

df['label'].value_counts()

df.info()

df['label'].nunique()

df.dtypes

sns.pairplot(df,hue='label')
plt.show()

print("Number of various crops: ", len(df['label'].unique()))
print("List of crops: ", df['label'].unique())

plt.figure(figsize=(12, 8))
correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()



#Feature Ranges and Statistical Analysis by Crop
def print_feature_ranges(df, numerical_features):
    """
    Print beautifully formatted feature ranges with dark mode optimized colors.

    Parameters:
    df (pandas.DataFrame): Input dataframe with crop data
    numerical_features (list): List of numerical feature names
    """
    # Custom theme for dark mode
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
    data_rows = []

    # Create title
    title = Text("🌾 FEATURE RANGES BY CROP 🌾", style="title")
    title_panel = Panel(
        title,
        box=box.DOUBLE,
        padding=(1, 30),
        style="border",
        title="Analysis Report"
    )
    console.print("\n")
    console.print(title_panel)
    console.print("\n")

    # Process each crop
    for crop in sorted(df['label'].unique()):
        crop_data = df[df['label'] == crop]

        # Create crop header
        crop_title = Text(f"\n📊 Crop: {crop.upper()}", style="crop")
        console.print(crop_title)

        # Create table for this crop
        table = Table(
            show_header=True,
            header_style="header",
            box=box.ROUNDED,
            border_style="border",
            show_lines=True,
            title=f"Statistics for {crop.upper()}",
            title_style="highlight"
        )

        # Add columns
        table.add_column("Feature", style="feature", justify="left")
        table.add_column("Minimum", style="value", justify="right")
        table.add_column("Maximum", style="value", justify="right")
        table.add_column("Range", style="value", justify="right")
        table.add_column("Mean", style="value", justify="right")

        # Process features for this crop
        for feature in numerical_features:
            min_val = crop_data[feature].min()
            max_val = crop_data[feature].max()
            mean_val = crop_data[feature].mean()
            range_val = max_val - min_val

            # Add row to table
            table.add_row(
                feature,
                f"{min_val:,.2f}",
                f"{max_val:,.2f}",
                f"{range_val:,.2f}",
                f"{mean_val:,.2f}"
            )

            # Store data for DataFrame
            data_rows.append({
                'Crop': crop,
                'Feature': feature,
                'Min': min_val,
                'Max': max_val,
                'Range': range_val,
                'Mean': mean_val
            })

        # Print table with border
        console.print(Panel(table, border_style="border"))

        # Print summary statistics for this crop
        summary_text = Text("\n📈 Key Statistics:", style="stats")
        console.print(summary_text)

        # Find feature with highest range
        max_range_feature = max(numerical_features, key=lambda x: crop_data[x].max() - crop_data[x].min())
        max_range_value = crop_data[max_range_feature].max() - crop_data[max_range_feature].min()

        # Find feature with highest variability (coefficient of variation)
        cv_values = {feature: crop_data[feature].std() / crop_data[feature].mean() * 100
                    for feature in numerical_features}
        most_variable_feature = max(cv_values, key=cv_values.get)

        # Print statistics with improved formatting
        console.print(Panel(
            "\n".join([
                f"[highlight]•[/highlight] Most variable feature: [feature]{max_range_feature}[/feature] (Range: [value]{max_range_value:,.2f}[/value])",
                f"[highlight]•[/highlight] Highest CV: [feature]{most_variable_feature}[/feature] (CV: [value]{cv_values[most_variable_feature]:.2f}%[/value])",
                f"[highlight]•[/highlight] Average {max_range_feature}: [value]{crop_data[max_range_feature].mean():,.2f}[/value]"
            ]),
            title="Summary",
            border_style="border",
            box=box.ROUNDED
        ))

        console.print("\n" + "─" * 80 + "\n")

    return pd.DataFrame(data_rows)

# Execute the function
numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
feature_ranges_df = print_feature_ranges(df, numerical_features)


#Skewness Analysis
print("Skewness Analysis:")
print("=================")
numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for feature in numerical_features:
    skewness = df[feature].skew()
    print(f"{feature}: {skewness:.3f}")

numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.xlabel(feature)
    plt.show()
    
    
sns.scatterplot(x = 'rainfall', y= 'humidity', data=df,
               style='label',
               hue='label')
    
plt.figure(figsize=(12, 8))
sns.pairplot(df[numerical_features])
plt.suptitle('Pair Plot of Numerical Features', y=1.02)
plt.show()


sns.pairplot(df, hue = 'label')


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title('Correlation Heatmap')
plt.show()

#Seperating features and target label
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []

# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

#Visualizing data

label_counts = df['label'].value_counts()

plt.figure(figsize=(10, 10))
plt.pie(label_counts, labels=label_counts.index, autopct="%.2f%%", startangle=125)
plt.title('Crop type Distribution')
plt.show()


# Remove Outliers (Only apply to numeric columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns  # Select only numeric columns

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1




x=df.drop(['label'],axis=1)

y=df['label']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

x_train.shape

x_test.shape

#evaluate(light_gbm)

#train the deep learning model with lgbm(training model)

# Predictive System that should be user input

# Standardization

# Train Test Split

#Model building

#Visualizing data

# Split Data to Training and Validation set

# Train and evaluate all models

# Performance Measure

#Making a prediction

label = ['KNN', 'Decision Tree','Random Forest','Naive Bayes']
Test = [knn_test_accuracy, dt_test_accuracy,rf_test_accuracy,
        nb_test_accuracy]
Train = [knn_train_accuracy,  dt_train_accuracy, rf_train_accuracy,
         nb_train_accuracy]

f, ax = plt.subplots(figsize=(20,7)) 
X_axis = np.arange(len(label))
plt.bar(X_axis - 0.2,Test, 0.4, label = 'Test', color=('midnightblue'))
plt.bar(X_axis + 0.2,Train, 0.4, label = 'Train', color=('mediumaquamarine'))

plt.xticks(X_axis, label)
plt.xlabel("ML algorithms")
plt.ylabel("Accuracy")
plt.title("Testing vs Training Accuracy")
plt.legend()
plt.show()

#Distribution of each crop

#Seperating features and target label

#Import required  libraries

#accuracy after prediction and compare to the predicted

#Classification Metrices

#LightGBM Model Building and Training

#Split dataset into training and test set

#Declare independent and target variables

#Correlation between different features

#boxplot
#pairplot

#Data Visualization and analysis- most required and least required for each feature

#KDE plot for each feature

#Feature Ranges and Statistical Analysis by Crop

#Scatter Plots of Feature Relationships

#Visualize the Distribution of Features

#CROP-WISE SUMMARY STATISTICS or Summary Statistics of the Crop Recommendation Dataset

#

