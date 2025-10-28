import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Assuming 'df' is your DataFrame containing the dataset
df = pd.read_csv("Crop_recommendation.csv")

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Inspect column names to ensure the correct columns exist
print("Column Names in df:", df.columns)

# Check for missing columns
required_columns = ['K', 'P', 'N', 'temperature', 'humidity', 'ph', 'rainfall']  # Corrected column names
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing Columns: {missing_columns}")
else:
    print("All required columns are present.")

# Data Preparation
df_copy = df.round()
label_map = {j: i for i, j in enumerate(df['label'].unique())}
df_copy['label'] = df['label'].replace(label_map)

# Binning the features (only if the columns exist)
for col in ['K', 'P', 'N', 'temperature', 'humidity', 'rainfall']:
    if col in df.columns:
        # Adjust the binning for each feature
        df_copy[f'{col}_bin'] = pd.cut(df_copy[col], bins=20) if col == 'K' else pd.cut(df_copy[col], bins=7)
        df_copy[f'{col}_bin'] = df_copy[f'{col}_bin'].apply(lambda x: str(x))

# Creating the parallel categories plot
df_copy['label_'] = df['label']
fig = px.parallel_categories(df_copy[['label_', 'K_bin', 'P_bin', 'N_bin', 'ph']], color_continuous_scale=px.colors.sequential.Inferno)
fig.show()

# Function to check multicollinearity(Multicollinearity refers to the situation where two or more independent variables in a regression model are highly correlated)
def check_multicollinearity(df_x):
    corr = df_x.corr()
    corr = pd.DataFrame(np.tril(corr, k=-1), columns=df_x.columns, index=df_x.columns)  # Lower triangular matrix
    corr = corr.replace(0.000000, np.NAN)
    count_of_total_correlation_values = corr.count().sum()

    for i in [0.5, 0.6, 0.7, 0.8, 0.9]:
        data_corr = corr[abs(corr) > i]
        count_greater_than_thresh = data_corr.count().sum()
        print(f'Percent Values Greater than {i} correlation: {count_greater_than_thresh / count_of_total_correlation_values}')
    
    return corr

# Function to plot correlation matrix(how features correlate with each other and help detect potential issues like multicollinearity.)
def plot_corr(corr, threshold=0.5):
    data_corr = corr[abs(corr) > threshold]
    sns.heatmap(data_corr, annot=True, cmap="YlGnBu")
    plt.show()

# Feature set
X = df[['K', 'P', 'N', 'temperature', 'humidity', 'rainfall']]  # Corrected feature names
y = df['label']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

# Feature importances plot
fig = go.Figure()
fig.add_trace(go.Bar(
    x=X.columns,
    y=model.feature_importances_,
    marker=dict(color='rgb(55, 83, 109)'),
    name='Feature Importance'
))

fig.update_layout(
    title="Feature Importance",
    xaxis_title="Features",
    yaxis_title="Importance",
    template='plotly_dark'
)

fig.show()