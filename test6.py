import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Crop_recommendation.csv")

# Descriptive Statistics: Summary for all crops
print("Average Ratio of Nitrogen in the Soil : {0:.2f}".format(df['N'].mean()))
print("Average Ratio of Phosphorous in the Soil : {0:.2f}".format(df['P'].mean()))
print("Average Ratio of Potassium in the Soil : {0:.2f}".format(df['K'].mean()))
print("Average Temperature in Celsius : {0:.2f}".format(df['temperature'].mean()))
print("Average Relative Humidity in % : {0:.2f}".format(df['humidity'].mean()))
print("Average PH Value of the soil : {0:.2f}".format(df['ph'].mean()))
print("Average Rainfall in mm : {0:.2f}".format(df['rainfall'].mean()))

# Summary Statistics for each Crop

def summary(crops=list(df['label'].value_counts().index)):
    x = df[df['label'] == crops]
    print("---------------------------------------------")
    print("Statistics for Nitrogen")
    print("Minimum Nitrogen required :", x['N'].min())
    print("Average Nitrogen required :", x['N'].mean())
    print("Maximum Nitrogen required :", x['N'].max()) 
    print("---------------------------------------------")
    print("Statistics for Phosphorous")
    print("Minimum Phosphorous required :", x['P'].min())
    print("Average Phosphorous required :", x['P'].mean())
    print("Maximum Phosphorous required :", x['P'].max()) 
    print("---------------------------------------------")
    print("Statistics for Potassium")
    print("Minimum Potassium required :", x['K'].min())
    print("Average Potassium required :", x['K'].mean())
    print("Maximum Potassium required :", x['K'].max()) 
    print("---------------------------------------------")
    print("Statistics for Temperature")
    print("Minimum Temperature required : {0:.2f}".format(x['temperature'].min()))
    print("Average Temperature required : {0:.2f}".format(x['temperature'].mean()))
    print("Maximum Temperature required : {0:.2f}".format(x['temperature'].max()))
    print("---------------------------------------------")
    print("Statistics for Humidity")
    print("Minimum Humidity required : {0:.2f}".format(x['humidity'].min()))
    print("Average Humidity required : {0:.2f}".format(x['humidity'].mean()))
    print("Maximum Humidity required : {0:.2f}".format(x['humidity'].max()))
    print("---------------------------------------------")
    print("Statistics for PH")
    print("Minimum PH required : {0:.2f}".format(x['ph'].min()))
    print("Average PH required : {0:.2f}".format(x['ph'].mean()))
    print("Maximum PH required : {0:.2f}".format(x['ph'].max()))
    print("---------------------------------------------")
    print("Statistics for Rainfall")
    print("Minimum Rainfall required : {0:.2f}".format(x['rainfall'].min()))
    print("Average Rainfall required : {0:.2f}".format(x['rainfall'].mean()))
    print("Maximum Rainfall required : {0:.2f}".format(x['rainfall'].max()))




# Interesting Facts and Patterns
print("Some Interesting Patterns")
print("---------------------------------")
print("Crops which requires very High Ratio of Nitrogen Content in Soil:", df[df['N'] > 120]['label'].unique())
print("Crops which requires very High Ratio of Phosphorous Content in Soil:", df[df['P'] > 100]['label'].unique())
print("Crops which requires very High Ratio of Potassium Content in Soil:", df[df['K'] > 200]['label'].unique())
print("Crops which requires very High Rainfall:", df[df['rainfall'] > 200]['label'].unique())
print("Crops which requires very Low Temperature:", df[df['temperature'] < 10]['label'].unique())
print("Crops which requires very High Temperature:", df[df['temperature'] > 40]['label'].unique())
print("Crops which requires very Low Humidity:", df[df['humidity'] < 20]['label'].unique())
print("Crops which requires very Low pH:", df[df['ph'] < 4]['label'].unique())
print("Crops which requires very High pH:", df[df['ph'] > 9]['label'].unique())



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx




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

# 2. Scatter plot to explore relationship between temperature, humidity, and rainfall
sns.relplot(x='temperature', y='humidity', hue='rainfall', data=df)

# 3. Exploratory Analysis - Creating box plots and histograms

# Box-Plot for each feature
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

# 4. Histograms for features
sns.histplot(data=df, x="N", kde=True, bins=30, color="blue")
plt.title("Distribution of Nitrogen in Soil")
plt.show()

sns.histplot(data=df, x="P", kde=True, bins=30, color="green")
plt.title("Distribution of Phosphorus in Soil")
plt.show()

sns.histplot(data=df, x="K", kde=True, bins=30, color="red")
plt.title("Distribution of Potassium in Soil")
plt.show()

# 5. Density plots for features
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



# 9. Crop Rotation Visualization with NetworkX
crop_rotation = {
    "Maize": "Pigeonpeas",
    "Pigeonpeas": "Rice",
    "Rice": "Kidneybeans",
    "Kidneybeans": "Chickpea",
    "Chickpea": "Maize"  # Loop back to maize
}

# Create directed graph for crop rotation
G = nx.DiGraph()
for crop, next_crop in crop_rotation.items():
    G.add_edge(crop, next_crop)

# Plot crop rotation graph
plt.figure(figsize=(8, 5))
pos = nx.circular_layout(G)  # Circular layout for clarity
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="black", node_size=3000, font_size=10, font_weight="bold")
plt.title("Crop Rotation Flowchart", fontsize=14)
plt.show()


# Filter the dataframe to only include numeric columns (N, P, K)
numeric_columns = ['N', 'P', 'K']

# Ensure that you're selecting only the rows for the crop and calculating the mean for the numeric columns
crop_npk = df[df['label'] == crop][numeric_columns].mean()

print(crop_npk)


# 10. Scatter plot of Nitrogen (N) vs. Phosphorus (P) in soil per crop type
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="N", y="P", hue="label", palette="Set2", s=100)
plt.xlabel("Nitrogen Content in Soil")
plt.ylabel("Phosphorus Content in Soil")
plt.title("Soil Nutrient Composition by Crop Type")
plt.legend(title="Crop Type", bbox_to_anchor=(1, 1))
plt.show()

# 11. Histogram to visualize soil pH distribution for different crops
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="ph", hue="label", kde=True, bins=30, palette="coolwarm")
plt.xlabel("Soil pH Level")
plt.ylabel("Frequency")
plt.title("Soil pH Levels Across Different Crops")
# Remove the custom plt.legend() call since it's handled by Seaborn
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px




# Define features (soil data) and target (crop type)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

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
    feature.plot(kind = 'hist')
    plt.title(f'{feature.name} histogram plot')
    
    plt.subplot(1, 3, 2)
    mu, sigma = scipy.stats.norm.fit(feature)
    sns.displot(feature, kde=True)
    plt.axvline(mu, linestyle = '--', color = 'green', )
    plt.axvline(sigma, linestyle = '--', color = 'red')
    plt.title(f'{feature.name} distribution plot')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(feature)
    plt.title(f'{feature.name} box plot')
    plt.show()

# Apply the feature plot for all numerical features
num_feat = df.select_dtypes(exclude='object')
for i in num_feat.columns:
    feat_plot(num_feat[i])

# N, P, K values comparison between crops
fig = go.Figure()
fig.add_trace(go.Bar(
    x=df['label'].unique(),
    y=df.groupby('label')['N'].mean(),
    name='Nitrogen',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=df['label'].unique(),
    y=df.groupby('label')['P'].mean(),
    name='Phosphorous',
    marker_color='lightsalmon'
))
fig.add_trace(go.Bar(
    x=df['label'].unique(),
    y=df.groupby('label')['K'].mean(),
    name='Potash',
    marker_color='crimson'
))

fig.update_layout(title="N, P, K values comparison between crops",
                  plot_bgcolor='white',
                  barmode='group',
                  xaxis_tickangle=-45)

fig.show()

# NPK ratio for rice, cotton, jute, maize, lentil
labels = ['Nitrogen(N)', 'Phosphorous(P)', 'Potash(K)']
fig = make_subplots(rows=1, cols=5, specs=[[{'type':'domain'}, {'type':'domain'},
                                            {'type':'domain'}, {'type':'domain'}, 
                                            {'type':'domain'}]])



crop_data = df[df['label'] == crop]

if not crop_data.empty:
    crop_npk = crop_data[['N', 'P', 'K']].mean()
    print(f"Average NPK for {crop}:")
    print(crop_npk)
else:
    print(f"No data found for crop: {crop}")


for crop in ['rice', 'cotton', 'jute', 'maize', 'lentil']:
    crop_npk = df[df['label'] == crop].mean()[['N', 'P', 'K']]
    values = [crop_npk['N'], crop_npk['P'], crop_npk['K']]
    fig.add_trace(go.Pie(labels=labels, values=values, name=crop), 1, len(fig.data) + 1)

fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(
    title_text="NPK ratio for rice, cotton, jute, maize, lentil",
    annotations=[dict(text=crop, x=0.06+(i*0.2), y=0.8, font_size=15, showarrow=False) for i, crop in enumerate(['Rice', 'Cotton', 'Jute', 'Maize', 'Lentil'])]
)

fig.show()

# NPK ratio for fruits
specs = [[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}],
         [{'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=5, specs=specs)

fruit_crops = ['apple', 'banana', 'grapes', 'orange', 'mango', 'coconut', 'papaya', 'pomegranate', 'watermelon', 'muskmelon']
for i, fruit in enumerate(fruit_crops):
    fruit_npk = df[df['label'] == fruit].mean()[['N', 'P', 'K']]
    values = [fruit_npk['N'], fruit_npk['P'], fruit_npk['K']]
    fig.add_trace(go.Pie(labels=labels, values=values, name=fruit, marker_colors=['rgb(255, 128, 0)', 'rgb(0, 153, 204)', 'rgb(173, 173, 133)']), 
                  row=(i // 5) + 1, col=(i % 5) + 1)

fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(
    title_text="NPK ratio for fruits",
    annotations=[dict(text=fruit.capitalize(), x=0.06+(i*0.2), y=0.8 - (i//5)*0.5, font_size=15, showarrow=False) for i, fruit in enumerate(fruit_crops)]
)

fig.show()

# Scatter plot for crops based on temperature and humidity
crop_scatter = df[df['label'].isin(['rice', 'jute', 'cotton', 'maize', 'lentil'])]
fig = px.scatter(crop_scatter, x="temperature", y="humidity", color="label", symbol="label")
fig.update_layout(plot_bgcolor='white')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Create box plots for each numerical feature before and after removing outliers
fig = go.Figure()

# Add box plots for each feature before removing outliers
for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
    fig.add_trace(go.Box(y=df[col], name=f'Before: {col}'))

# Add box plots for each feature after removing outliers (you will need to create 'cleaned_df' before this step)
# Assuming 'cleaned_df' is your cleaned dataset
for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
    fig.add_trace(go.Box(y=cleaned_df[col], name=f'After: {col}'))

# Update layout
fig.update_layout(title="Comparison of Numerical Features Before and After Removing Outliers",
                  yaxis_title="Value",
                  xaxis_title="Feature",
                  boxmode='group')

# Show plot
fig.show()



import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame containing the dataset

# Data Preparation
data_copy = data.round()
label_map = {j: i for i, j in enumerate(data['label'].unique())}
data_copy['label'] = data['label'].replace(label_map)

# Binning the features
data_copy['potassium_bin'] = pd.cut(data_copy['potassium'], bins=20)
data_copy['potassium_bin'] = data_copy['potassium_bin'].apply(lambda x: str(x))
data_copy['phosphorus_bin'] = pd.cut(data_copy['phosphorus'], bins=7)
data_copy['phosphorus_bin'] = data_copy['phosphorus_bin'].apply(lambda x: str(x))
data_copy['nitrogen_bin'] = pd.cut(data_copy['nitrogen'], bins=7)
data_copy['nitrogen_bin'] = data_copy['nitrogen_bin'].apply(lambda x: str(x))
data_copy['humidity_bin'] = pd.cut(data_copy['humidity'], bins=8)
data_copy['humidity_bin'] = data_copy['humidity_bin'].apply(lambda x: str(x))
data_copy['temperature_bin'] = pd.cut(data_copy['temperature'], bins=7)
data_copy['temperature_bin'] = data_copy['temperature_bin'].apply(lambda x: str(x))
data_copy['rainfall_bin'] = pd.cut(data_copy['rainfall'], bins=9)
data_copy['rainfall_bin'] = data_copy['rainfall_bin'].apply(lambda x: str(x))

# Creating the parallel categories plot
data_copy['label_'] = data['label']
fig = px.parallel_categories(data_copy[['label_', 'potassium_bin', 'phosphorus_bin', 'nitrogen_bin', 'ph']], color_continuous_scale=px.colors.sequential.Inferno)
fig.show()

# Function to check multicollinearity
def check_multicollinearity(data_x):
    corr = data_x.corr()
    corr = pd.DataFrame(np.tril(corr, k=-1), columns=data_x.columns, index=data_x.columns)  # Lower triangular matrix
    corr = corr.replace(0.000000, np.NAN)
    count_of_total_correlation_values = corr.count().sum()

    for i in [0.5, 0.6, 0.7, 0.8, 0.9]:
        data_corr = corr[abs(corr) > i]
        count_greater_than_thresh = data_corr.count().sum()
        print(f'Percent Values Greater than {i} correlation: {count_greater_than_thresh / count_of_total_correlation_values}')
    
    return corr

# Function to plot correlation matrix
def plot_corr(corr, threshold=0.5):
    data_corr = corr[abs(corr) > threshold]
    sns.heatmap(data_corr, annot=True, cmap="YlGnBu")
    plt.show()

# Checking correlation and plotting
corr = check_multicollinearity(X)
plot_corr(corr)

# Params Checking (Model Coefficients)
import plotly.graph_objs as go
df_coeff = pd.DataFrame(model.coef_, columns=X.columns, index=label_map.keys())
# df_coeff['intercept'] = model.intercept_  # Uncomment to use intercept as well

# Plotting the coefficients
fig = go.Figure()
cols = df_coeff.columns
for index in df_coeff.index:
    fig.add_trace(go.Scatter(y=df_coeff.loc[index, :].values, x=cols, mode='lines', name=index))

fig.update_layout(template='plotly_dark')
fig.show()
