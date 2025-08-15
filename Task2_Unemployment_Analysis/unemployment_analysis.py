# 📌 Task 2: Unemployment Analysis
# OASIS INFOBYTE INTERNSHIP

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------
# 1️⃣ Detect unemployment file (Excel or CSV)
# ----------------------
folder = os.path.dirname(os.path.abspath(__file__))
file_name = None
for f in os.listdir(folder):
    if "unemployment" in f.lower() and f.lower().endswith((".xlsx", ".xls", ".csv")):
        file_name = f
        break

if not file_name:
    raise FileNotFoundError("❌ No file containing 'unemployment' found in the folder.")

file_path = os.path.join(folder, file_name)

# ----------------------
# 2️⃣ Load dataset based on file type
# ----------------------
if file_name.lower().endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    df = pd.read_excel(file_path)

print(f"✅ Loaded {file_name} successfully with {df.shape[0]} rows and {df.shape[1]} columns.\n")
print(df.head(), "\n")

# ----------------------
# 3️⃣ Data Cleaning
# ----------------------
print("📊 Missing values:\n", df.isnull().sum(), "\n")
print("📌 Data types:\n", df.dtypes, "\n")

df.dropna(how='all', inplace=True)
df.fillna(method='ffill', inplace=True)

# Standardize column names
df.columns = df.columns.str.strip()
df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour Participation Rate'
}, inplace=True)

# Convert Date column if exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df.to_csv("unemployment_clean.csv", index=False)
print("💾 Cleaned file saved as unemployment_clean.csv")


# ----------------------
# 4️⃣ Summary Statistics
# ----------------------
print("📈 Summary statistics:\n", df.describe(), "\n")

# ----------------------
# 5️⃣ Visualization
# ----------------------
sns.set(style="whitegrid")

# Bar Plot: Unemployment Rate by Region & Area
if 'Region' in df.columns and 'Area' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Region', y='Unemployment Rate', hue='Area', data=df, errorbar=None)
    plt.xticks(rotation=90)
    plt.title('Unemployment Rate by Region & Area')
    plt.tight_layout()
    plt.show()

# Line Plot: Trend over Time
if 'Date' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Date', y='Unemployment Rate', hue='Area', data=df)
    plt.title('Unemployment Trend Over Time')
    plt.tight_layout()
    plt.show()

# Heatmap: Correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

print("✅ Analysis & visualizations completed.")
