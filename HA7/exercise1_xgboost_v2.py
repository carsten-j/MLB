# exercise1_xgboost.py
# ------------------------------------------------------------
#  Machine Learning B – Home assignment 7 – Exercise 1
#  Author: <your-name>
#  ------------------------------------------------------------
#  This script solves sub-questions 1–3 (loading & splitting
#  the data set, training a first XGBoost model, running a
#  grid-search for better hyper-parameters and comparing the
#  result to a KNN baseline).
# ------------------------------------------------------------

import json
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

plt.style.use("ggplot")
warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42  # global reproducibility switch
DATA_FILE = "quasars.csv"  # file delivered with the assignment

# ------------------------------------------------------------
# Step 1.  Load the data set and create *one* test split
# ------------------------------------------------------------
print("\nStep 1 - Loading data and creating an 80/20 train-test split")

df = pd.read_csv(DATA_FILE, header=None)

X = df.iloc[:, :-1].values  # first 10 columns  -> features
y = df.iloc[:, -1].values  # last  column     -> target  (red-shift)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

print(f"  Training instances: {X_train.shape[0]}")
print(f"  Test instances    : {X_test.shape[0]}")

# ------------------------------------------------------------
# Step 2.  Train the *given* XGBoost configuration and monitor
#          RMSE on a 90 / 10 sub-split of the training data
# ------------------------------------------------------------
print("\nStep 2 - Fitting the reference XGBoost model and plotting learning curves")

# 90 / 10 split of *training* portion for on-the-fly validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=RANDOM_STATE
)

eval_metric = "rmse"

xgb_ref = XGBRegressor(
    objective="reg:squarederror",
    # parameters specified by the exercise
    colsample_bytree=0.5,
    learning_rate=0.1,
    max_depth=4,
    reg_lambda=1.0,
    n_estimators=500,
    eval_metric=eval_metric,
    # nice-to-have settings
    random_state=RANDOM_STATE,
    verbosity=0,
)

eval_set = [(X_tr, y_tr), (X_val, y_val)]

print("  -> training … (this may take half a minute)")
start = time.time()
xgb_ref.fit(
    X_tr, y_tr, eval_set=eval_set, verbose=False
)  # turn to True to watch progress
print(f"     training finished in {time.time() - start:4.1f}s")

# --- Plot RMSE vs. boosting iteration -----------------------
results = xgb_ref.evals_result()
rmse_tr = results["validation_0"][eval_metric]
rmse_va = results["validation_1"][eval_metric]

plt.figure(figsize=(6, 4))
plt.plot(rmse_tr, label="train")
plt.plot(rmse_va, label="validation")
plt.xlabel("Boosting iteration")
plt.ylabel("RMSE")
plt.title("Learning curves - reference XGBoost")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curves_reference_xgb.pdf", dpi=600, bbox_inches="tight")
print("  -> plot saved as learning_curves_reference_xgb.png")

# --- Evaluate on the *held-out* test set ---------------------
y_pred_ref = xgb_ref.predict(X_test)
rmse_ref = root_mean_squared_error(y_test, y_pred_ref)
r2_ref = r2_score(y_test, y_pred_ref)

print(f"  Test RMSE (reference): {rmse_ref:8.5f}")
print(f"  Test R²   (reference): {r2_ref:8.5f}")

# ------------------------------------------------------------
# Step 3.  Hyper-parameter grid search (3-fold CV) plus KNN = 5
# ------------------------------------------------------------
print("\nStep 3 - Grid-search for better hyper-parameters")

# Enhanced parameter grid with more comprehensive hyperparameter tuning
param_grid = {
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7],
    "learning_rate": [0.01, 0.05, 0.1, 0.15],
    "max_depth": [6, 8, 10, 12],
    "n_estimators": [300, 500, 700],
    "reg_lambda": [1.0, 1.5, 2.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],  # L1 regularization
    "subsample": [0.7, 0.8, 0.9],  # Row sampling
    "min_child_weight": [1, 3, 5],  # Minimum instances per leaf
    "gamma": [0, 0.1, 0.2],  # Minimum loss reduction
}

# ------------------------------------------------------------
# Alternative: Two-stage grid search for better efficiency
# Stage 1: Coarse grid search
print("\n  Stage 1: Coarse grid search")
coarse_param_grid = {
    "colsample_bytree": [0.8, 1.0],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [10, 12],
    "n_estimators": [200, 300],
    "reg_lambda": [0.5, 1.0, 2.0],
    "reg_alpha": [0.0, 0.5, 1.0],
    "subsample": [0.5, 0.7],
    "min_child_weight": [5, 8],
    "gamma": [0, 0.1],
}

# Use the coarse grid instead for faster execution
# Comment out the enhanced grid above and use this for practical purposes
param_grid = coarse_param_grid

xgb_base = XGBRegressor(
    objective="reg:squarederror", random_state=RANDOM_STATE, verbosity=0
)

grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)  # ONLY on *training* part

best_params = grid.best_params_
best_rmse = -grid.best_score_  # negate again
print("\n  Best 3-fold CV RMSE: ", best_rmse)
print("  Best parameter set  : ")
print(json.dumps(best_params, indent=4))

# ---- Train final XGBoost with the discovered configuration --
print("\n  -> retraining best configuration with early stopping")

# Create validation split for early stopping
X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
    X_train, y_train, test_size=0.15, random_state=RANDOM_STATE
)

xgb_best = XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    verbosity=0,
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    eval_metric="rmse",
    **best_params,
)

# Fit with early stopping
xgb_best.fit(
    X_tr_final, y_tr_final, eval_set=[(X_val_final, y_val_final)], verbose=False
)

# Retrain on full training data with optimal number of estimators
optimal_estimators = xgb_best.best_iteration
print(f"  Optimal number of estimators: {optimal_estimators}")

# Final model with optimal estimators on full training data
xgb_best_final = XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    verbosity=0,
    n_estimators=optimal_estimators,
    **{k: v for k, v in best_params.items() if k != "n_estimators"},
)
xgb_best_final.fit(X_train, y_train)

# plot_importance(xgb_best)
# plt.tight_layout()
# plt.savefig("feature_importance_best_xgb.pdf", dpi=600, bbox_inches="tight")

y_pred_best = xgb_best_final.predict(X_test)
rmse_best = root_mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"\n  Test RMSE (best XGB) : {rmse_best:8.5f}")
print(f"  Test R²   (best XGB) : {r2_best:8.5f}")

# ---- Baseline: 5-nearest-neighbours regressor ---------------
print("\n  -> Fitting 5-nearest-neighbours baseline")
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
rmse_knn = root_mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f"\n  Baseline RMSE (k=5): {rmse_knn:8.5f}")
print(f"  Baseline R²        : {r2_knn:8.5f}")

# ------------------------------------------------------------
# Alternative: Two-stage grid search for better efficiency
# Stage 1: Coarse grid search
print("\n  Stage 1: Coarse grid search")
coarse_param_grid = {
    "colsample_bytree": [0.4, 0.6, 0.8],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [6, 8, 10],
    "n_estimators": [300, 500],
    "reg_lambda": [0.5, 1.0, 2.0],
    "reg_alpha": [0.0, 0.5, 1.0],
    "subsample": [0.7, 0.9],
    "min_child_weight": [1, 5],
    "gamma": [0, 0.1],
}

# Use the coarse grid instead for faster execution
# Comment out the enhanced grid above and use this for practical purposes
# param_grid = coarse_param_grid

# ------------------------------------------------------------
# Additional improvements for better RMSE
# ------------------------------------------------------------

# 1. Feature scaling (add to imports: from sklearn.preprocessing import RobustScaler)
# Uncomment to add feature scaling:
# scaler = RobustScaler()  # More robust to outliers than StandardScaler
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 2. Feature engineering ideas:
# - Polynomial features of degree 2 for key features
# - Interaction terms between highly correlated features
# - Log transformations for skewed features

# 3. Ensemble methods:
# - Combine XGBoost with other algorithms (Random Forest, LightGBM)
# - Use different random seeds and average predictions

# 4. Advanced XGBoost techniques:
# - Use 'dart' or 'gblinear' booster types
# - Experiment with different objectives like 'reg:gamma' or 'reg:tweedie'

# ------------------------------------------------------------
#  Final comparison table
# ------------------------------------------------------------
print("\n----------------------------------------------------------")
print("   Model comparison on the *unseen* test set")
print("   (lower RMSE & higher R² are better)")
print("----------------------------------------------------------")
print(f"   Reference XGBoost :  RMSE {rmse_ref:8.5f} | R² {r2_ref:8.5f}")
print(f"   Tuned    XGBoost :  RMSE {rmse_best:8.5f} | R² {r2_best:8.5f}")
print(f"   k-NN (k=5)       :  RMSE {rmse_knn:8.5f} | R² {r2_knn:8.5f}")
print("----------------------------------------------------------\n")
