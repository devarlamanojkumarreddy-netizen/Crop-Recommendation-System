import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Load data
df = pd.read_csv("Crop_recommendation.csv")

# -------------------------------
# General Dataset Statistics
# -------------------------------
print("Average Ratio of Nitrogen in the Soil : {:.2f}".format(df['N'].mean()))
print("Average Ratio of Phosphorous in the Soil : {:.2f}".format(df['P'].mean()))
print("Average Ratio of Potassium in the Soil : {:.2f}".format(df['K'].mean()))
print("Average Temperature in Celsius : {:.2f}".format(df['temperature'].mean()))
print("Average Relative Humidity in % : {:.2f}".format(df['humidity'].mean()))
print("Average PH Value of the soil : {:.2f}".format(df['ph'].mean()))
print("Average Rainfall in mm : {:.2f}".format(df['rainfall'].mean()))

# -------------------------------
# Summary for a specific crop
# -------------------------------
def summary(crop):
    x = df[df['label'] == crop]
    print("\nSummary statistics for", crop)
    print("------------------------------------------------")
    print("Nitrogen - min: {}, mean: {:.2f}, max: {}".format(x['N'].min(), x['N'].mean(), x['N'].max()))
    print("Phosphorous - min: {}, mean: {:.2f}, max: {}".format(x['P'].min(), x['P'].mean(), x['P'].max()))
    print("Potassium - min: {}, mean: {:.2f}, max: {}".format(x['K'].min(), x['K'].mean(), x['K'].max()))
    print("Temperature - min: {:.2f}, mean: {:.2f}, max: {:.2f}".format(x['temperature'].min(), x['temperature'].mean(), x['temperature'].max()))
    print("Humidity - min: {:.2f}, mean: {:.2f}, max: {:.2f}".format(x['humidity'].min(), x['humidity'].mean(), x['humidity'].max()))
    print("pH - min: {:.2f}, mean: {:.2f}, max: {:.2f}".format(x['ph'].min(), x['ph'].mean(), x['ph'].max()))
    print("Rainfall - min: {:.2f}, mean: {:.2f}, max: {:.2f}".format(x['rainfall'].min(), x['rainfall'].mean(), x['rainfall'].max()))
    print("------------------------------------------------")

# Example usage:
summary('rice')

# -------------------------------
# Interesting Patterns
# -------------------------------
print("\nSome Interesting Patterns")
print("---------------------------------")
print("Crops with high Nitrogen (>120):", df[df['N'] > 120]['label'].unique())
print("Crops with high Phosphorous (>100):", df[df['P'] > 100]['label'].unique())
print("Crops with high Potassium (>200):", df[df['K'] > 200]['label'].unique())
print("Crops with high Rainfall (>200):", df[df['rainfall'] > 200]['label'].unique())
print("Crops with low Temperature (<10°C):", df[df['temperature'] < 10]['label'].unique())
print("Crops with high Temperature (>40°C):", df[df['temperature'] > 40]['label'].unique())
print("Crops with low Humidity (<20%):", df[df['humidity'] < 20]['label'].unique())
print("Crops with low pH (<4):", df[df['ph'] < 4]['label'].unique())
print("Crops with high pH (>9):", df[df['ph'] > 9]['label'].unique())

# -------------------------------
# LGBM Classifier
# -------------------------------
# Prepare data
X = df.drop('label', axis=1)
y = df['label']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LGBM Model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy and Report
print("\nAccuracy of LGBM model: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Feature Importance Plot( shows how important each feature (or variable) is for making predictions in a machine learning model.)
# -------------------------------
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances from LGBM")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()