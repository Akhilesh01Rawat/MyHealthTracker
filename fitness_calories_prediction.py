# ==========================================
# FITNESS CALORIES PREDICTION PROJECT
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("fitness_data.csv")

print("Dataset Shape:", df.shape)

# ==========================================
# DATA CLEANING
# ==========================================

df = df.dropna()

df["ACTIVITY_DATE"] = pd.to_datetime(df["ACTIVITY_DATE"])

df["DAY_NUM"] = df["ACTIVITY_DATE"].dt.dayofweek
df["MONTH_NUM"] = df["ACTIVITY_DATE"].dt.month

print(df.head())

# ==========================================
# BASIC ANALYSIS
# ==========================================

print(df.describe())

# ==========================================
# CORRELATION
# ==========================================

plt.figure(figsize=(8,5))
sns.heatmap(
    df[['STEPS','KM','CALORIES']].corr(),
    annot=True,
    cmap="Blues"
)
plt.title("Correlation Matrix")
plt.show()

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(df["ACTIVITY_DATE"], df["STEPS"])
plt.title("Daily Steps")
plt.xlabel("Date")
plt.ylabel("Steps")
plt.show()

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df[['STEPS','KM','DAY_NUM','MONTH_NUM']]

y = df['CALORIES']

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Rows:", len(X_train))
print("Testing Rows:", len(X_test))

# ==========================================
# LINEAR REGRESSION
# ==========================================

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\nLINEAR REGRESSION")

print("R2 Score:", r2_score(y_test, lr_pred))
print("MAE:", mean_absolute_error(y_test, lr_pred))

# ==========================================
# RANDOM FOREST
# ==========================================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRANDOM FOREST")

print("R2 Score:", r2_score(y_test, rf_pred))
print("MAE:", mean_absolute_error(y_test, rf_pred))

# ==========================================
# FUTURE PREDICTION
# ==========================================

new_data = pd.DataFrame({
    'STEPS':[15000],
    'KM':[10.5],
    'DAY_NUM':[1],
    'MONTH_NUM':[9]
})

prediction = rf.predict(new_data)

print("\nPREDICTED CALORIES")
print(prediction[0])

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

print("\nFeature Importance")
print(importance.sort_values(
    by='Importance',
    ascending=False
))

plt.figure(figsize=(8,4))
sns.barplot(
    x='Importance',
    y='Feature',
    data=importance.sort_values(
        by='Importance',
        ascending=False
    )
)
plt.title("Feature Importance")
plt.show()
