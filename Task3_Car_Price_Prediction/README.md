# 🚗 Task 3 - Car Price Prediction
**OASIS INFOBYTE Internship (Data Science)**

## 📌 Overview
This project aims to predict the **selling price** of cars based on various features such as year of manufacture, kilometers driven, fuel type, seller type, and transmission.  
We use **Linear Regression** as our predictive model.

## 📂 Dataset
- File: `car_data.csv`
- Contains features like:
  - `Year`
  - `Present_Price`
  - `Kms_Driven`
  - `Fuel_Type`
  - `Seller_Type`
  - `Transmission`
  - `Selling_Price` (Target variable)

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pickle (for model saving)

## 🔍 Steps Performed
1. **Data Loading** – Read dataset using Pandas.
2. **Data Preprocessing** – Handled missing values, encoded categorical variables.
3. **Train/Test Split** – 80% training and 20% testing data.
4. **Model Training** – Applied Linear Regression.
5. **Evaluation** – Measured R² Score, MSE, and RMSE.
6. **Visualization** – Plotted Actual vs Predicted Prices.
7. **Model Saving** – Exported trained model as `car_price_model.pkl`.

## 📊 Model Performance
- **R² Score:** ~0.84 *(varies with dataset)*
- **RMSE:** ~1.2 *(varies with dataset)*

## 📷 Outputs
- Car Price Distribution
- Actual vs Predicted Car Prices

## 💾 How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run script
python car_price_prediction.py
