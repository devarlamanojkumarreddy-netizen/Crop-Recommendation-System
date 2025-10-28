import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Crop_recommendation.csv")
# Histogram of all features
df.hist(bins=30, figsize=(12, 8))
plt.suptitle("Histograms of Features", fontsize=20)
plt.tight_layout()
plt.show()

# Print shape and number of duplicate rows
print("Shape of data:", df.shape)
print("Number of duplicate rows:", df.duplicated().sum())

# Visualizations: Histogram with KDE, Violin plot, and Box plot for each feature
for i in df.columns[:-1]:  # Exclude target column
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Histogram with KDE
    sns.histplot(data=df, x=i, kde=True, bins=20, ax=ax[0], color='skyblue')
    ax[0].set_title(f'Histogram and KDE of {i}', fontsize=15)
    ax[0].set_xlabel(i, fontsize=12)
    ax[0].set_ylabel('Frequency', fontsize=12)

    # Violin plot
    sns.violinplot(data=df, y=i, ax=ax[1], color='lightgreen')
    ax[1].set_title(f'Violin Plot of {i}', fontsize=15)
    ax[1].set_ylabel(i, fontsize=12)
    ax[1].set_xlabel("")

    # Box plot
    sns.boxplot(data=df, y=i, ax=ax[2], color='salmon')
    ax[2].set_title(f'Box Plot of {i}', fontsize=15)
    ax[2].set_ylabel(i, fontsize=12)
    ax[2].set_xlabel("")

    plt.suptitle(f'Visualizing Feature: {i}', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sm

df = pd.read_csv("Crop_recommendation.csv")

# Scatter plot: Temperature vs Rainfall
plt.figure(figsize=(8, 6))
plt.scatter(df['temperature'], df['rainfall'], color='green', alpha=0.5)
plt.xlabel('Temperature')
plt.ylabel('Rainfall')
plt.title('Scatter Plot of Temperature vs. Rainfall')
plt.show()

# Histogram of Temperature
plt.figure(figsize=(8, 6))
plt.hist(df['temperature'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.title('Distribution of Temperature')
plt.show()

# Boxplot of Temperature by Crop Label
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='temperature', data=df)
plt.xlabel('Crop Label')
plt.ylabel('Temperature')
plt.title('Box Plot of Temperature for Each Crop Label')
plt.xticks(rotation=45)
plt.show()

# Count plot of crop labels
plt.figure(figsize=(10, 6))
sns.countplot(x='label', hue='label', data=df, palette='Set3', legend=False)
plt.xlabel('Crop Label')
plt.ylabel('Count')
plt.title('Count of Each Crop Label')
plt.xticks(rotation=45)
plt.show()

# Boxplot comparison of N, P, K
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['N', 'P', 'K']], palette='Set3')
plt.title('Comparison of N, P, K')
plt.xlabel('Nutrient')
plt.ylabel('Value')
plt.show()

# Melt the data for grouped bar plot
dfm = df.melt(id_vars=['label'], value_vars=['N', 'P', 'K'], var_name='Nutrient', value_name='Value')

# Grouped bar plot of N, P, K by crop
plt.figure(figsize=(12, 8))
sns.barplot(data=dfm, x='label', y='Value', hue='Nutrient', dodge=True)
plt.xlabel('Crops')
plt.ylabel('Values')
plt.title('Comparison of N, P, K across different crops')
plt.xticks(rotation=90)
plt.legend(title='Nutrient', title_fontsize='14')
plt.tight_layout()
plt.show()

# Boxplots for each feature by crop label (first 6 features)
plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns[:-2], 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='label', y=column, data=df)
    plt.xticks(rotation=90)
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()

# General boxplots for all numeric columns (excluding label)
plt.figure(figsize=(12, 12))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    df[[col]].boxplot()
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Q-Q plots for checking normality
plt.figure(figsize=(12, 12))
for i, col in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sm.probplot(df[col], dist='norm', plot=plt)
    plt.title(f'Q-Q Plot of {col}')
plt.tight_layout()
plt.show()


from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- LIME EXPLANATION ---
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Split into features and target
X = df.drop('label', axis=1)
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (e.g., LightGBM)
from lightgbm import LGBMClassifier
model = LGBMClassifier()
model.fit(X_train, y_train)

# Now use LIME
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=model.classes_,
    mode='classification'
)

# Initialize LIME for tabular data
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=model.classes_,
    discretize_continuous=True
)

# Explain a specific instance
instance_idx = 0
instance = X_test.iloc[instance_idx]
exp = explainer.explain_instance(instance.values, model.predict_proba, num_features=len(X_test.columns))

# Plot LIME feature importance(LIME (Local Interpretable Model-agnostic Explanations) is a technique used to explain machine learning model predictions by approximating the complex model with an interpretable one locally (around a specific prediction).)
exp.as_pyplot_figure()
plt.title('LIME Feature Importance')
plt.tight_layout()
plt.show()

# --- MODEL EVALUATION ---

# Accuracy on training data
train_accuracy = model.score(X_train, y_train)
print("Train Accuracy:", train_accuracy)

# Accuracy on test data
test_accuracy = model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Accuracy on whole dataset
y_all_pred = model.predict(X)
whole_data_accuracy = accuracy_score(y, y_all_pred)
print("Whole Data Accuracy:", whole_data_accuracy)

# Confusion matrix for test data
y_test_pred = model.predict(X_test)
cm_test = confusion_matrix(y_test, y_test_pred)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=model.classes_)
disp_test.plot(cmap='Blues')
plt.title("Confusion Matrix - Test Data")
plt.show()

# Confusion matrix for whole data
cm_whole = confusion_matrix(y, y_all_pred)
disp_whole = ConfusionMatrixDisplay(confusion_matrix=cm_whole, display_labels=model.classes_)
disp_whole.plot(cmap='Greens')
plt.title("Confusion Matrix - Whole Data")
plt.show()

# --- JOINT PLOT (Rainfall vs Humidity) ---(A Joint Plot is a visualization that combines both a scatter plot and the univariate distributions (like histograms or kernel density estimates) of two numerical variables)
sns.jointplot(
    x="rainfall",
    y="humidity",
    data=df[(df['temperature'] < 40) & (df['rainfall'] > 40)],
    height=10,
    hue="label"
)
plt.suptitle("Joint Plot: Rainfall vs Humidity (Filtered by Temperature & Rainfall)", y=1.02)
plt.show()

# --- Cross-validation Score ---(used to evaluate the performance of a model by splitting the data into several subsets, training the model on some subsets, and testing it on the remaining ones.)
score = cross_val_score(model, X, y, cv=5)
print('Cross-validation score:', score)
print('Mean CV score:', score.mean())

# --- Box Plot of pH per Crop ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(30, 15))
sns.boxplot(x='label', y='ph', data=df)
plt.xticks(rotation=90)
plt.title("Box Plot of pH Across Crop Labels")
plt.xlabel("Crop")
plt.ylabel("pH Level")
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from pandas.plotting import scatter_matrix
import plotly.express as px

df = pd.read_csv("Crop_recommendation.csv")

# --- Confusion Matrix Heatmap ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True, linewidth=0.5, fmt=".0f", cmap='viridis')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Predicted vs Actual')
plt.show()

# --- Histogram of Temperature ---
plt.figure(figsize=(8, 5))
sns.histplot(df['temperature'], color='Purple')
plt.title('Histogram of Temperature')
plt.show()

# --- Scatter Matrix ---
attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "num"]
scatter_matrix(df[attributes], figsize=(15, 15))
plt.show()

# --- Sample Accuracy and Precision Scores (assuming they're calculated before) ---
accuracy = 0.975
precision = 0.9761914318624844
print(f"Accuracy Score: {accuracy}")
print(f"Precision Score: {precision}")

# --- Mean Values Grouped by Label ---
grouped = df.groupby(by='label').mean().reset_index()
print(grouped)

# --- Top 5 Crops Requiring Highest Amount of Each Element ---
for i in grouped.columns[1:]:
    print(f'\nTop 5 Most {i} requiring crops:')
    for j, k in grouped.sort_values(by=i, ascending=False)[['Label', i]].head(5).values:
        print(f'{j} --> {k}')
    print('***************************************')

# --- Top 5 Crops Requiring Least Amount of Each Element ---
for i in grouped.columns[1:]:
    print(f'\nTop 5 Least {i} requiring crops:')
    for j, k in grouped.sort_values(by=i)[['Label', i]].head(5).values:
        print(f'{j} --> {k}')
    print('*************************************')

# --- F1 Score Bar Plot ---
plt.figure(figsize=(7, 4))
bars = plt.barh(features, f1_scores, color='skyblue', edgecolor='black')

plt.xlabel('F1 Score')
plt.title('F1 Score for Each Soil Metric and Weather Feature', pad=15)
plt.xlim(0, 1)

for bar, f1 in zip(bars, f1_scores):
    plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
             f'{f1:.2f}', va='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()

# --- Histograms for Features by Label ---
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for col, ax in zip(data.columns, axes.flatten()):
    sns.histplot(data=data, x=col, hue='label', ax=ax, legend=False)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# --- Histogram, Violin, and Boxplot for each column ---
plt.style.use('ggplot')
sns.set_palette("husl")

for i in df.columns[:-1]:
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(data=df, x=i, kde=True, bins=40, ax=ax[0])
    sns.violinplot(data=df, x=i, ax=ax[1])
    sns.boxplot(data=df, x=i, ax=ax[2])
    plt.suptitle(f'Visualizing {i}', size=20)
    plt.tight_layout()
    plt.show()

# --- ROC Curves for Multi-Class Classification ---
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve class %d (area = %0.2f)' % (i, roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# --- Plotly Boxplot for Outliers ---
fig = px.box(data, y="N", points="all", title="Boxplot of Nitrogen (N) with Outliers")
fig.show()

# --- Analyze Elements: Sorted by Nitrogen ---
Nitrogen = cropMeanValue.sort_values(by='N', ascending=False)
analyzeElements(Nitrogen, 'N', "Nitrogen")  # Assuming `analyzeElements` is a defined function

# --- Ratio of Elements Pie Chart ---
def plotElementRatio(name):
    labels = ['Nitrogen(N)', 'Phosphorous(P)', 'Potash(K)']
    elementNPK = cropMeanValue[cropMeanValue.index == name]
    sizes = [elementNPK['N'].values[0], elementNPK['P'].values[0], elementNPK['K'].values[0]]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           colors=['#ffb3dd', '#99ddff', '#aaff94'])
    plt.title(name)
    plt.show()

# Plot for example crop
plotElementRatio('apple')


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
import plotly.express as px
from plotly.subplots import make_subplots

df = pd.read_csv("Crop_recommendation.csv")

# --- Train and Test Set Scores ---
print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))

# --- Random Forest Model Training and Evaluation ---
accuracy = []
training_accuracy = []
testing_accuracy = []

for i in range(1, 51):
    model = RandomForestClassifier(max_depth=i, random_state=42)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy.append(r2_score(Y_test, Y_pred) * 100)
    training_accuracy.append(model.score(X_train, Y_train) * 100)
    testing_accuracy.append(model.score(X_test, Y_test) * 100)

plt.plot(list(range(1, 51)), accuracy, label='accuracy')
plt.plot(list(range(1, 51)), training_accuracy, label='training_accuracy')
plt.plot(list(range(1, 51)), testing_accuracy, label='testing_accuracy')
plt.legend()
plt.show()

# Find the best depth (with highest accuracy)
i = accuracy.index(max(accuracy))
print(i, accuracy[i], training_accuracy[i], testing_accuracy[i])

# --- Crop Type Analysis by Seasons ---
print("Summer Crops")
print(data[(data['temperature'] > 30) & (data['humidity'] > 50)]['label'].unique())
print("------------------------------------------------------------------------------")
print("Winter Crops")
print(data[(data['temperature'] < 20) & (data['humidity'] > 30)]['label'].unique())
print("------------------------------------------------------------------------------")
print("Rainy Crops")
print(data[(data['rainfall'] > 200) & (data['humidity'] > 50)]['label'].unique())

# --- Checking Overfitting with LGBM ---
print('LGBM Accuracy on training set: ', LGBM.score(x_train, y_train))
print('LGBM Accuracy on test set: ', LGBM.score(x_test, y_test))

# --- Accuracy of Multiple Models ---
accuracy_models = dict(zip(model, acc))  # assuming 'acc' contains accuracy scores for models
for k, v in accuracy_models.items():
    print(k, ' : ', v)

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


import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import seaborn as sns

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
    nodelist = d.keys()
    node_sizes = d.values()
    
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

# Call the function for positive and negative correlation
corr_network(G, corr_direction="positive", min_correlation=0)
corr_network(G, corr_direction="negative", min_correlation=0)

# --- Plotting Feature Correlations (Scatter Plots) ---
numeric = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numeric)
featuresNum = ['N', 'P', 'K']

for i in range(-1, len(featuresNum)-3):
    fig = go.Figure()
    fig.update_layout(autosize=False, width=1280, height=720)
    trace1 = go.Scatter(y=df[featuresNum[i + 1]], name=featuresNum[i + 1])
    fig.add_trace(trace1)
    trace2 = go.Scatter(y=df[featuresNum[i + 2]], name=featuresNum[i + 2])
    fig.add_trace(trace2)
    trace3 = go.Scatter(y=df[featuresNum[i + 3]], name=featuresNum[i + 3])
    fig.add_trace(trace3)
    fig.show()

# --- Plot Pie Charts for Crop Composition ---
featuresNum = df.columns

plt.figure(figsize=(50, 40))
for i in range(0, len(featuresNum)-1):
    fig = px.pie(df, values=featuresNum[i], names='label', 
                 color_discrete_sequence=px.colors.sequential.RdBu, 
                 title="Composition of crops with " + featuresNum[i])
    fig.show()

# --- Plot Pie Chart for Crop Class Distribution ---
fig = px.pie(df, values=df['label'].value_counts().values, 
             names=df['label'].value_counts().index, 
             color_discrete_sequence=px.colors.sequential.RdBu, 
             title="Composition of crops class")
fig.show()

# --- Scatter Matrix Plot ---
fig = px.scatter_matrix(df, dimensions=['N', 'P', 'K'], color="label")
fig.update_layout(autosize=True, width=1920, height=1080)
fig.show()

# --- Explainable AI with LIME ---
exp = LimeTabularExplainer(X_test.values, feature_names=list(X.columns), 
                           discretize_continuous=True, class_names=X.columns)
_, doc_nums = np.unique(np.array(y_test), return_index=True)

for num in doc_nums:
    print('Actual Label:', np.array(y_test)[num])
    print('Predicted Label:', y_pred_test[num])
    exp.explain_instance(X_test.iloc[num].values, model.predict_proba).show_in_notebook()

# --- Violin Plot for Nitrogen (N) vs Crop Labels ---
plt.figure(figsize=(30, 13))
sns.violinplot(x="label", y="N", data=crop, hue="label", dodge=False)
plt.show()
import pandas as pd

# Load your dataset (adjust path as needed)
Crop_recommendation = pd.read_csv('Crop_recommendation.csv')

# Now this will work
for i in Crop_recommendation.columns[:-1]:  # Exclude target column
    print(i)

sns.pairplot(df, hue='label')
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
