import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report
from plotly.subplots import make_subplots
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("Crop_recommendation.csv")

# Prepare the data
X = df.drop('label', axis=1)  # Features
y = df['label']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train LGBM Model ---
model = lgb.LGBMClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- Train and Test Set Scores ---
print('Training set accuracy: {:.4f}'.format(model.score(X_train, y_train)))
print('Test set accuracy: {:.4f}'.format(model.score(X_test, y_test)))

# --- Accuracy Calculation with LightGBM ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100  # Use accuracy_score instead of r2_score

print(f'Accuracy of LGBM on test set: {accuracy:.2f}%')

# --- Crop Type Analysis by Seasons ---
print("Summer Crops")
print(df[(df['temperature'] > 30) & (df['humidity'] > 50)]['label'].unique())
print("------------------------------------------------------------------------------")
print("Winter Crops")
print(df[(df['temperature'] < 20) & (df['humidity'] > 30)]['label'].unique())
print("------------------------------------------------------------------------------")
print("Rainy Crops")
print(df[(df['rainfall'] > 200) & (df['humidity'] > 50)]['label'].unique())

# --- Accuracy of Multiple Models ---
accuracy_models = {
    "LGBM Accuracy": accuracy
}

for k, v in accuracy_models.items():
    print(f"{k}: {v:.2f}%")

# --- Stripplot for pH vs Label ---
sns.stripplot(y="label", x="ph", hue="label", orient="h", data=df, size=5)
plt.show()

# --- Scatterplot for All Data Points ---
plt.figure(figsize=(20, 15))
sns.scatterplot(data=df)
plt.show()

# --- Parallel Coordinates Plot (Plotly) ---
df2 = df.copy()
dfg = pd.DataFrame({'label': df2['label'].unique()})
dfg['dummy'] = dfg.index
df2 = pd.merge(df2, dfg, on='label', how='left')

dimensions = [
    dict(range=[0, df2['dummy'].max()], tickvals=dfg['dummy'], ticktext=dfg['label'], label='Crops', values=df2['dummy']),
    dict(range=[df2['N'].min(), df2['N'].max()], label='N', values=df2['N']),
    dict(range=[df2['P'].min(), df2['P'].max()], label='P', values=df2['P']),
    dict(range=[df2['K'].min(), df2['K'].max()], label='K', values=df2['K']),
    dict(range=[df2['temperature'].min(), df2['temperature'].max()], label='Temperature', values=df2['temperature']),
    dict(range=[df2['humidity'].min(), df2['humidity'].max()], label='Humidity', values=df2['humidity']),
    dict(range=[df2['ph'].min(), df2['ph'].max()], label='pH', values=df2['ph']),
    dict(range=[df2['rainfall'].min(), df2['rainfall'].max()], label='Rainfall', values=df2['rainfall'])
]

fig = go.Figure(data=go.Parcoords(line=dict(color=df2['dummy'], colorscale='magma'), dimensions=dimensions))
fig.update_layout(
    height=550, width=1000, title='Crop Feature Distribution', title_font_size=20, title_x=0.5, title_y=0.95
)
fig.show()

# --- Boxplots for Numeric Features ---
numeric = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numeric)

fig = make_subplots(rows=1, cols=len(df_num.columns))
featuresNum = df_num.columns

plt.figure(figsize=(50, 40))
for i in range(0, len(featuresNum)):
    trace = go.Box(y=df[featuresNum[i]], name=featuresNum[i])
    fig.append_trace(trace, row=1, col=i + 1)

fig.show()
