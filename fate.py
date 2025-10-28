import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
crop = pd.read_csv("Crop_recommendation.csv")  # Change path if needed

# Basic EDA(Exploratory Data Analysis)
print(crop.shape)
print(crop.info())
print(crop.isnull().sum())
print(crop.duplicated().sum())
print(crop.describe())
print(crop['label'].value_counts())

# Visualize one of the features
sns.histplot(crop['N'])
plt.title('Distribution of Nitrogen Content')
plt.show()

# Encode labels to numeric values
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)
crop.drop('label', axis=1, inplace=True)

# Train-test split
from sklearn.model_selection import train_test_split
x = crop.drop('crop_num', axis=1)
y = crop['crop_num']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling(used to normalize or standardize the range of independent variables or features in a dataset)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train multiple models and print accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

models = {
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'SVC': SVC(),
    'KNeighbors': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'ExtraTree': ExtraTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

print("Model Accuracy Scores:")
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")

# Final model (Random Forest)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)

# Crop dictionary reverse lookup
crop_dict_reverse = {v: k.capitalize() for k, v in crop_dict.items()}

# Prediction function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features = scaler.transform(features)  # scale the input
    prediction = rfc.predict(features)
    return prediction[0]

# Take user input
print("\nEnter values for prediction:")
N = float(input("Enter Nitrogen content (N): "))
P = float(input("Enter Phosphorus content (P): "))
K = float(input("Enter Potassium content (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall (mm): "))

# Predict and display result
predicted_crop_id = recommendation(N, P, K, temperature, humidity, ph, rainfall)
predicted_crop_name = crop_dict_reverse.get(predicted_crop_id, "Unknown Crop")

print(f"\n Recommended Crop: {predicted_crop_name}")
