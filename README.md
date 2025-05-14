# 🚕 Taxi Fare Prediction - NYC

This project aims to build a regression model to predict the taxi fare in New York City based on pickup and drop-off coordinates, datetime, and trip characteristics.

---

## 🧠 Problem Statement

The objective is to predict the `fare_amount` (in USD) for a taxi trip in NYC using data-driven machine learning techniques. This includes preprocessing, feature engineering, model building, and evaluation.

---

## 📁 Dataset

The dataset includes columns such as:

- `fare_amount` — the target variable (cost of the ride)
- `pickup_datetime`
- `pickup and drop-off coordinates (lat/long)`
- Additional engineered features like hour, weekday, year, and distance

---

## 🔧 Technologies Used

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Plotly, AutoViz
- **Models Used:**
  - Linear Regression
  - Decision Tree Regressor
  - AdaBoost Regressor
  - Gradient Boosting Regressor
  - Random Forest Regressor
  - Support Vector Regressor (SVR)

---

## 🚀 Key Features

- 📊 Data cleaning, outlier removal, and missing value handling
- 🧭 Geospatial distance calculated using the Haversine formula
- 📅 Feature engineering on time-based fields (year, weekday, hour)
- 🧪 Train-test split and evaluation using R² Score
- 🔍 AutoViz used for quick EDA visualization
- 📈 Achieved **85% accuracy** (R² score) using ensemble models

---

## 📉 Model Performance

| Model                   | R² Score (Test) |
|------------------------|-----------------|
| Linear Regression       | ~85%            |
| Decision Tree Regressor | ~83%            |
| AdaBoost Regressor      | ~84%            |
| Gradient Boosting       | ~85%            |
| Random Forest           | ~85%            |
| SVR                     | ~70%            |

> Final model selected based on performance: **Gradient Boosting Regressor**

---

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/taxi-fare-prediction.git
