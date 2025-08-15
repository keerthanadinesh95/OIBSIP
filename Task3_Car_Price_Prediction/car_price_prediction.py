# car_price_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# Load the dataset
df = pd.read_csv("car_data.csv")

# Display first few rows
print("First 5 rows of dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Encoding categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df_encoded.drop("Selling_Price", axis=1)
y = df_encoded["Selling_Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()

# Save the trained model
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as car_price_model.pkl")

# Example usage of saved model
# (Uncomment below lines if you want to test loading)
# with open("car_price_model.pkl", "rb") as file:
#     loaded_model = pickle.load(file)
#     sample_input = [[5.59, 2014, 27000, 0, 0, 1, 0, 0]]  # Example
#     print("Sample Prediction:", loaded_model.predict(sample_input))

