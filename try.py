import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_openml


boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data.values
y = boston.target.values.astype(float)

# Split train/test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features for ADALINE & MLP
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_raw)
X_test_std = scaler.transform(X_test_raw)

# Custom ADALINE Regressor
class AdalineRegressor:
    def __init__(self, learning_rate=0.01, epochs=300, random_state=42):
        self.lr = learning_rate
        self.epochs = epochs
        self.rng = np.random.RandomState(random_state)
        self.losses_ = []

    def fit(self, X, y):
        X_wb = np.c_[np.ones((X.shape[0], 1)), X]  
        n_features = X_wb.shape[1]
        self.w_ = self.rng.normal(loc=0.0, scale=0.01, size=n_features)

        for _ in range(self.epochs):
            y_hat = X_wb @ self.w_
            errors = y - y_hat
            grad = -2.0 * X_wb.T @ errors / X_wb.shape[0]
            self.w_ -= self.lr * grad
            self.losses_.append((errors**2).mean())
        return self

    def predict(self, X):
        X_wb = np.c_[np.ones((X.shape[0], 1)), X]
        return X_wb @ self.w_

# Train & Evaluate Models
results = {}

def train_and_eval(name, fit_fn, pred_fn, loss_curve=None):
    t0 = time.perf_counter()
    fit_fn()
    train_time = time.perf_counter() - t0
    y_pred = pred_fn()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results[name] = {
        "MSE": mse,
        "RMSE": rmse,
        "Train Time": train_time,
        "Pred": y_pred,
        "Loss": loss_curve,
    }

# ADALINE
adaline = AdalineRegressor(learning_rate=0.01, epochs=500)
train_and_eval(
    "ADALINE",
    fit_fn=lambda: adaline.fit(X_train_std, y_train),
    pred_fn=lambda: adaline.predict(X_test_std),
    loss_curve=adaline.losses_,
)

# MLP
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500,early_stopping=True, random_state=42)
train_and_eval(
    "MLP",
    fit_fn=lambda: mlp.fit(X_train_std, y_train),
    pred_fn=lambda: mlp.predict(X_test_std),
    loss_curve=getattr(mlp, "loss_curve_", None),
)

# Linear Regression
linreg = LinearRegression()
train_and_eval(
    "Linear Regression",
    fit_fn=lambda: linreg.fit(X_train_raw, y_train),
    pred_fn=lambda: linreg.predict(X_test_raw),
)

# Decision Tree
tree = DecisionTreeRegressor(random_state=42)
train_and_eval(
    "Decision Tree",
    fit_fn=lambda: tree.fit(X_train_raw, y_train),
    pred_fn=lambda: tree.predict(X_test_raw),
)

# Print Results
print("\n=== Test Results (Boston Housing) ===")
for name, res in results.items():
    print(f"\n{name}")
    print(f"  MSE        : {res['MSE']:.4f}")
    print(f"  RMSE       : {res['RMSE']:.4f}")
    print(f"  Train Time : {res['Train Time']:.4f} s")

# Visualizations

# 1) Scatter Plots: Predicted vs Actual
plt.figure(figsize=(12, 10))
for i, (name, res) in enumerate(results.items(), start=1):
    plt.subplot(2, 2, i)
    plt.scatter(y_test, res["Pred"], alpha=0.6)
    y_min, y_max = y_test.min(), y_test.max()
    plt.plot([y_min, y_max], [y_min, y_max], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name}: Predicted vs Actual")
plt.tight_layout()
plt.show()

# 2) Residual Plots: ADALINE & MLP
plt.figure(figsize=(10, 4))
for i, name in enumerate(["ADALINE", "MLP"], start=1):
    res = results[name]
    residuals = y_test - res["Pred"]
    plt.subplot(1, 2, i)
    plt.scatter(res["Pred"], residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Residual Plot")
plt.tight_layout()
plt.show()

# 3) Training Curves: ADALINE & MLP
plt.figure(figsize=(10, 4))
for i, name in enumerate(["ADALINE", "MLP"], start=1):
    plt.subplot(1, 2, i)
    if results[name]["Loss"] is not None:
        plt.plot(results[name]["Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name}: Training Curve")
plt.tight_layout()
plt.show()
