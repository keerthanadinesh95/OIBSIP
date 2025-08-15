# Iris Flower Classification - Oasis Infobyte Internship
# Author: Keerthana Dinesh

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving/loading the model

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("First 5 rows of dataset:")
print(df.head())

# Data visualization
sns.pairplot(df, hue='species')
plt.show()

# Split data
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "iris_classifier.pkl")
print("Model saved as iris_classifier.pkl")

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Load the saved model (example)
loaded_model = joblib.load("iris_classifier.pkl")
sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Example measurement
prediction = loaded_model.predict(sample_data)
print(f"Prediction for sample data {sample_data}: {prediction[0]}")
