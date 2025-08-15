# Task 2 - Unemployment Analysis 📊
**OASIS INFOBYTE Internship**

## 📌 Objective
The objective of this task was to analyze unemployment data, clean and preprocess it, and generate visual insights using Python.

## 📂 Files in this folder
- `unemployment_analysis.py` → Python script for analysis & visualization
- `unemployment_clean.csv` → Cleaned dataset after preprocessing
- `unemployment.xlsx` → Original dataset
- `task2_bar_chart_unemployment_rate.png` → Bar chart visualization
- `task2_line_chart_unemployment_trend.png` → Line chart visualization
- `task2_heatmap_correlation.png` → Correlation heatmap visualization

## 🛠️ Tools & Libraries Used
- **Python**  
- **Pandas** → Data loading & cleaning  
- **Matplotlib** → Plotting visualizations  
- **Seaborn** → Stylish charts & graphs  

## 📊 Steps Performed
1. **Loaded Dataset** (Excel/CSV)  
2. **Data Cleaning**  
   - Removed empty rows  
   - Forward-filled missing values  
   - Standardized column names  
   - Converted date column to datetime format  
3. **Exploratory Data Analysis (EDA)**  
4. **Visualizations**  
   - **Bar Chart** → Unemployment Rate by Region & Area  
   - **Line Chart** → Unemployment Trend Over Time  
   - **Heatmap** → Correlation between employment-related metrics  

## 📈 Key Insights
- Different regions have varying unemployment rates.
- Urban vs Rural areas show noticeable differences in employment patterns.
- Some metrics have strong correlations, as shown in the heatmap.

## 🚀 How to Run
1. Install the required libraries:
```bash
pip install pandas matplotlib seaborn
