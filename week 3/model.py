import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = "data/sugarcane_yield_dataset.csv"

df = pd.read_csv(r"D:\SIT\Sem.5\Hack 0 Week\week 5 and 6\sugarcane_yield_dataset.csv")

print("Dataset loaded successfully")
print(df.head())


# ============================================================
# REMOVE ID COLUMN
# ============================================================

df = df.drop(columns=["Plot_ID"])


# ============================================================
# ENCODE SOIL TYPE
# ============================================================

df = pd.get_dummies(
    df,
    columns=["Soil_Type"],
    dtype=int
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Yield_tons_ha"])

y = df["Yield_tons_ha"]


print("\nFeatures:")
print(X.columns.tolist())


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# TRAIN MODEL
# ============================================================

model = LinearRegression()

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


# ============================================================
# SAVE MODEL
# ============================================================

model_data = {

    "model": model,

    "scaler": scaler,

    "features": X.columns.tolist()
}


joblib.dump(
    model_data,
    "sugarcane_model.pkl"
)


print("\nModel saved successfully!")
print("File: sugarcane_model.pkl")