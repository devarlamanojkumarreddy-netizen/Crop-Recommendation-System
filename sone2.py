import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming 'df' is your DataFrame containing the dataset
df = pd.read_csv("Crop_recommendation.csv")

# Strip spaces from column names
df.columns = df.columns.str.strip()

# 1. Visualize the impact of different agricultural conditions on crops using bar plots
plt.rcParams['figure.figsize'] = (15, 8)

plt.subplot(2, 4, 1)
sns.barplot(x=df['N'], y=df['label'])
plt.ylabel(' ')
plt.xlabel('Ratio of Nitrogen', fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2, 4, 2)
sns.barplot(x=df['P'], y=df['label'])
plt.ylabel(' ')
plt.xlabel('Ratio of Phosphorous', fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2, 4, 3)
sns.barplot(x=df['K'], y=df['label'])
plt.ylabel(' ')
plt.xlabel('Ratio of Potassium', fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2, 4, 4)
sns.barplot(x=df['temperature'], y=df['label'])
plt.ylabel(' ')
plt.xlabel('Temperature', fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2, 4, 5)
sns.barplot(x=df['humidity'], y=df['label'])
plt.ylabel(' ')
plt.xlabel('Humidity', fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2, 4, 6)
sns.barplot(x=df['ph'], y=df['label'])
plt.ylabel(' ')
plt.xlabel('pH of Soil', fontsize=10)
plt.yticks(fontsize=10)

plt.subplot(2, 4, 7)
sns.barplot(x=df['rainfall'], y=df['label'])
plt.ylabel(' ')
plt.xlabel('Rainfall', fontsize=10)
plt.yticks(fontsize=10)

plt.suptitle('Visualizing the Impact of Different Conditions on Crops', fontsize=15)
plt.show()

# 2. Scatter plot to explore relationship between temperature, humidity, and rainfall( type of data visualization that displays individual data points on a two-dimensional grid. )
sns.relplot(x='temperature', y='humidity', hue='rainfall', data=df)

# 3. Exploratory Analysis - Creating box plots and histograms
# Box-Plot for each feature(Box plots are particularly useful for identifying outliers and understanding the spread of the data)
plt.figure(figsize=(16, 10))
plt.subplot(1, 7, 1)
sns.boxplot(data=df['N'])
plt.title('Box Plot for Nitrogen')

plt.subplot(1, 7, 2)
sns.boxplot(data=df['P'])
plt.title('Box Plot for Phosphorus')

plt.subplot(1, 7, 3)
sns.boxplot(data=df['K'])
plt.title('Box Plot for Potassium')

plt.subplot(1, 7, 4)
sns.boxplot(data=df['temperature'])
plt.title('Box Plot for Temperature')

plt.subplot(1, 7, 5)
sns.boxplot(data=df['humidity'])
plt.title('Box Plot for Humidity')

plt.subplot(1, 7, 6)
sns.boxplot(data=df['ph'])
plt.title('Box Plot for pH')

plt.subplot(1, 7, 7)
sns.boxplot(data=df['rainfall'])
plt.title('Box Plot for Rainfall')

plt.show()

# 4. Histograms for features(It groups the data into bins (or intervals) and plots the frequency (count) of data points in each bin)
sns.histplot(data=df, x="N", kde=True, bins=30, color="blue")
plt.title("Distribution of Nitrogen in Soil")
plt.show()

sns.histplot(data=df, x="P", kde=True, bins=30, color="green")
plt.title("Distribution of Phosphorus in Soil")
plt.show()

sns.histplot(data=df, x="K", kde=True, bins=30, color="red")
plt.title("Distribution of Potassium in Soil")
plt.show()

# 5. Density plots for features(A density plot (also known as a Kernel Density Estimate (KDE) plot) is a smoothed version of a histogram. It estimates the probability density function (PDF) of a continuous random variable.)
df_long = df.melt(id_vars=['label'], value_vars=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], var_name='Feature', value_name='Value')
sns.kdeplot(data=df_long, x='Value', hue='Feature', common_norm=False)
plt.title('Density Plots for Agricultural Conditions')
plt.show()

# 6. Interesting Facts
print("Crops which require very High ratio of Nitrogen Content in soil:", df[df['N'] > 120]['label'].unique())
print("Crops which require very High ratio of Phosphorous Content in soil:", df[df['P'] > 100]['label'].unique())
print("Crops which require very High ratio of Potassium Content in soil:", df[df['K'] > 200]['label'].unique())
print("Crops which require very High Rainfall:", df[df['rainfall'] > 200]['label'].unique())
print("Crops which require very Low Rainfall:", df[df['rainfall'] < 40]['label'].unique())
print("Crops which require very Low Temperature:", df[df['temperature'] < 10]['label'].unique())
print("Crops which require very High Temperature:", df[df['temperature'] > 40]['label'].unique())
print("Crops which require very Low Humidity:", df[df['humidity'] < 20]['label'].unique())
print("Crops which require very Low pH:", df[df['ph'] < 4]['label'].unique())
print("Crops which require very High pH:", df[df['ph'] > 8]['label'].unique())

# 7. Crop Rotation Visualization with NetworkX
crop_rotation = {
    "Maize": "Pigeonpeas",
    "Pigeonpeas": "Rice",
    "Rice": "Kidneybeans",
    "Kidneybeans": "Chickpea",
    "Chickpea": "Maize"  # Loop back to maize
}

# Create directed graph for crop rotation(we can represent crops as nodes, and the edges would indicate the sequence in which crops should be grown.)
G = nx.DiGraph()
for crop, next_crop in crop_rotation.items():
    G.add_edge(crop, next_crop)

# Plot crop rotation graph(Crop rotation helps maintain soil health and optimize yields by alternating crops with different nutrient requirements and pest vulnerabilities)
plt.figure(figsize=(8, 5))
pos = nx.circular_layout(G)  # Circular layout for clarity
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="black", node_size=3000, font_size=10, font_weight="bold")
plt.title("Crop Rotation Flowchart", fontsize=14)
plt.show()

# 8. LightGBM Model for Crop Prediction
# Prepare features and target variable
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

# Plot Feature Importance using LightGBM
importances = model.feature_importances_
feature_names = X.columns

# Plot the feature importances. 
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance from LightGBM')
plt.show()

# 9. Scatter plot of Nitrogen (N) vs. Phosphorus (P) in soil per crop type
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="N", y="P", hue="label", palette="Set2", s=100)
plt.xlabel("Nitrogen Content in Soil")
plt.ylabel("Phosphorus Content in Soil")
plt.title("Soil Nutrient Composition by Crop Type")
plt.legend(title="Crop Type", bbox_to_anchor=(1, 1))
plt.show()

# 10. Histogram to visualize soil pH distribution for different crops
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="ph", hue="label", kde=True, bins=30, palette="coolwarm")
plt.xlabel("Soil pH Level")
plt.ylabel("Frequency")
plt.title("Soil pH Levels Across Different Crops")
plt.show()