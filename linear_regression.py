# =============================================================
# Task 3: Linear Regression
# Dataset : train.csv (House Prices)
# Run     : python linear_regression.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── 1. LOAD & PREPROCESS ─────────────────────────────────────
df = pd.read_csv("train.csv")

# Features for Multiple LR
FEATURES = ["GrLivArea", "OverallQual", "TotalBsmtSF",
            "GarageArea", "YearBuilt", "FullBath", "TotRmsAbvGrd"]
TARGET = "SalePrice"

data = df[FEATURES + [TARGET]].copy()
data.fillna(data.median(numeric_only=True), inplace=True)

print("=" * 55)
print("  Dataset shape :", df.shape)
print("  Features used :", FEATURES)
print("  Target        :", TARGET)
print("=" * 55)

# ── 2. SPLIT DATA ─────────────────────────────────────────────
X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── 3A. SIMPLE LINEAR REGRESSION (GrLivArea only) ─────────────
X_s_train = X_train[["GrLivArea"]]
X_s_test  = X_test[["GrLivArea"]]

slr = LinearRegression()
slr.fit(X_s_train, y_train)
y_slr_pred = slr.predict(X_s_test)

print("\n─── Simple Linear Regression (GrLivArea → SalePrice) ───")
print(f"  Intercept  : {slr.intercept_:,.2f}")
print(f"  Coefficient: {slr.coef_[0]:,.4f}  ($ per sq ft)")

# ── 3B. MULTIPLE LINEAR REGRESSION (all 7 features) ───────────
scaler = StandardScaler()
X_m_train = scaler.fit_transform(X_train)
X_m_test  = scaler.transform(X_test)

mlr = LinearRegression()
mlr.fit(X_m_train, y_train)
y_mlr_pred = mlr.predict(X_m_test)

print("\n─── Multiple Linear Regression (7 features → SalePrice) ─")
print(f"  Intercept : {mlr.intercept_:,.2f}")
print(f"\n  {'Feature':<18} {'Coefficient':>13}")
print(f"  {'─'*18} {'─'*13}")
for feat, coef in sorted(zip(FEATURES, mlr.coef_),
                          key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat:<18} {coef:>13,.2f}")

# ── 4. EVALUATE ───────────────────────────────────────────────
def evaluate(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{label}]")
    print(f"    MAE  : ${mae:>10,.2f}")
    print(f"    MSE  : ${mse:>15,.2f}")
    print(f"    RMSE : ${rmse:>10,.2f}")
    print(f"    R²   :  {r2:>10.4f}")
    return mae, mse, rmse, r2

print("\n─── Evaluation Metrics ───────────────────────────────────")
mae_s, mse_s, rmse_s, r2_s = evaluate(y_test, y_slr_pred,  "Simple LR  ")
mae_m, mse_m, rmse_m, r2_m = evaluate(y_test, y_mlr_pred, "Multiple LR")

print(f"\n  R² improvement: +{(r2_m - r2_s)/r2_s*100:.1f}%  |  "
      f"RMSE drop: ${rmse_s - rmse_m:,.0f}")

# ── 5. PLOTS ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Task 3 — Linear Regression on House Prices",
             fontsize=15, fontweight="bold")

# Plot 1: Regression line (Simple LR)
ax = axes[0, 0]
ax.scatter(X_s_test, y_test, alpha=0.4, color="steelblue", s=20, label="Actual")
x_line = np.linspace(X_s_test["GrLivArea"].min(),
                     X_s_test["GrLivArea"].max(), 200).reshape(-1, 1)
ax.plot(x_line, slr.predict(x_line), color="red", lw=2, label="Regression line")
ax.set_title("Simple LR — Regression Line")
ax.set_xlabel("GrLivArea (sq ft)")
ax.set_ylabel("Sale Price ($)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (Multiple LR)
ax = axes[0, 1]
ax.scatter(y_test, y_mlr_pred, alpha=0.4, color="orange", s=20)
lim = [min(y_test.min(), y_mlr_pred.min()) - 5000,
       max(y_test.max(), y_mlr_pred.max()) + 5000]
ax.plot(lim, lim, "r--", lw=1.5, label="Perfect fit")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"Multiple LR — Actual vs Predicted  (R²={r2_m:.4f})")
ax.set_xlabel("Actual ($)"); ax.set_ylabel("Predicted ($)")
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 3: Feature coefficients (Multiple LR)
ax = axes[0, 2]
coef_df = pd.DataFrame({"Feature": FEATURES, "Coef": mlr.coef_})\
            .sort_values("Coef")
colors = ["salmon" if c < 0 else "steelblue" for c in coef_df["Coef"]]
ax.barh(coef_df["Feature"], coef_df["Coef"], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Multiple LR — Feature Coefficients")
ax.set_xlabel("Coefficient (standardised)")
ax.grid(True, alpha=0.3, axis="x")

# Plot 4: Residuals — Simple LR
ax = axes[1, 0]
res_s = y_test - y_slr_pred
ax.scatter(y_slr_pred, res_s, alpha=0.4, color="steelblue", s=20)
ax.axhline(0, color="red", lw=1.5, linestyle="--")
ax.set_title("Simple LR — Residuals vs Predicted")
ax.set_xlabel("Predicted ($)"); ax.set_ylabel("Residual ($)")
ax.grid(True, alpha=0.3)

# Plot 5: Residuals — Multiple LR
ax = axes[1, 1]
res_m = y_test - y_mlr_pred
ax.scatter(y_mlr_pred, res_m, alpha=0.4, color="orange", s=20)
ax.axhline(0, color="red", lw=1.5, linestyle="--")
ax.set_title("Multiple LR — Residuals vs Predicted")
ax.set_xlabel("Predicted ($)"); ax.set_ylabel("Residual ($)")
ax.grid(True, alpha=0.3)

# Plot 6: Metrics comparison
ax = axes[1, 2]
metrics = ["MAE ($k)", "RMSE ($k)", "R²"]
vals_s = [mae_s/1000, rmse_s/1000, r2_s]
vals_m = [mae_m/1000, rmse_m/1000, r2_m]
x_pos  = np.arange(3)
w = 0.35
b1 = ax.bar(x_pos - w/2, vals_s, w, label="Simple LR",   color="steelblue", alpha=0.8)
b2 = ax.bar(x_pos + w/2, vals_m, w, label="Multiple LR", color="orange",    alpha=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels(metrics)
ax.set_title("Model Metrics Comparison")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("linear_regression_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → linear_regression_results.png")
print("\nDone ✓")