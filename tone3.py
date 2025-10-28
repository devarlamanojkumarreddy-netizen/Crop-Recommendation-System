import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv("Crop_recommendation.csv")

# --- Correlation Network Visualization ---
def corr_network(G, corr_direction, min_correlation):
    H = G.copy()

    for s1, s2, weight in G.edges(data=True):       
        if corr_direction == "positive":
            if weight["weight"] < min_correlation or s1 == s2:
                H.remove_edge(s1, s2)
        else:
            if weight["weight"] > min_correlation or s1 == s2:
                H.remove_edge(s1, s2)

    edges, weights = zip(*nx.get_edge_attributes(H, 'weight').items())
    weights = tuple([(1 + abs(x))**2 for x in weights])
   
    d = dict(nx.degree(H))
    nodelist = list(d.keys())  # Corrected from keys() to list()
    node_sizes = list(d.values())  # Corrected from values() to list()
    
    positions = nx.circular_layout(H)
    
    plt.figure(figsize=(15, 15))
    nx.draw_networkx_nodes(H, positions, node_color='#d100d1', nodelist=nodelist,
                           node_size=tuple([x**3 for x in node_sizes]), alpha=0.8)
    nx.draw_networkx_labels(H, positions, font_size=14)

    if corr_direction == "positive":
        edge_colour = plt.cm.summer 
    else:
        edge_colour = plt.cm.autumn
        
    nx.draw_networkx_edges(H, positions, edgelist=edges, style='solid',
                           width=weights, edge_color=weights, edge_cmap=edge_colour,
                           edge_vmin=min(weights), edge_vmax=max(weights))
    plt.axis('off')
    plt.show()

# --- Feature Correlations ---
featuresNum = ['N', 'P', 'K']

for feature in featuresNum:
    fig = go.Figure()
    fig.update_layout(autosize=False, width=1280, height=720)
    fig.add_trace(go.Scatter(y=df[feature], mode='lines', name=feature))
    fig.show()

# --- Plot Pie Chart for Crop Class Distribution ---
label_counts = df['label'].value_counts()

# Ensure it's a clean dataframe for plotting
fig = px.pie(labels=label_counts.index,  # Use the index for the 'names'
             values=label_counts.values,  # Use the values for the count of each label
             color_discrete_sequence=px.colors.sequential.RdBu, 
             title="Crop Class Distribution")
fig.show()


# --- Scatter Matrix Plot ---
fig = px.scatter_matrix(df, dimensions=['N', 'P', 'K'], color="label")
fig.update_layout(autosize=True, width=1920, height=1080)
fig.show()

# --- Split the data and train an LGBM model ---
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the LGBM model
model = lgb.LGBMClassifier()
model = lgb.LGBMClassifier(num_leaves=64, max_depth=10, learning_rate=0.05, min_child_samples=20)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of LGBM model: {accuracy:.4f}")

# --- Explainable AI with LIME ---
exp = LimeTabularExplainer(X_train.values, feature_names=X.columns, 
                           class_names=y.unique(), discretize_continuous=True)

for i in range(5):  # Displaying explanations for 5 samples
    exp.explain_instance(X_test.iloc[i].values, model.predict_proba).show_in_notebook()

# --- Violin Plot for Nitrogen (N) vs Crop Labels ---
plt.figure(figsize=(30, 13))
sns.violinplot(x="label", y="N", data=df, hue="label", dodge=False)
plt.show()

# --- Pairplot ---
sns.pairplot(df, hue='label')
plt.show()

# --- Correlation Heatmap ---
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
