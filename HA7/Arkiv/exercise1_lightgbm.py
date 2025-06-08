# exercise1_lightgbm.py
# ------------------------------------------------------------
#  Machine Learning B – Home assignment 7 – Exercise 1 (LightGBM version)
#  Author: <your-name>
#  ------------------------------------------------------------
#  This script solves sub-questions 1–3 (loading & splitting
#  the data set, training a first LightGBM model, running a
#  grid-search for better hyper-parameters and comparing the
#  result to a KNN baseline).
# ------------------------------------------------------------

import json
import pathlib
import sys
import time
import warnings

import lightgbm as lgb  # For callbacks if needed, e.g. early_stopping
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor

plt.style.use("ggplot")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # LightGBM can produce some FutureWarnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
RANDOM_STATE = 42  # global reproducibility switch
DATA_FILE = "quasars.csv"  # file delivered with the assignment

# ------------------------------------------------------------
# Step 1.  Load the data set and create *one* test split
# ------------------------------------------------------------
print("\nStep 1 – Loading data and creating an 80 / 20 train-test split")
if not pathlib.Path(DATA_FILE).exists():
    sys.exit(f"❌  Could not find {DATA_FILE}. Put the file next to the script.")

df = pd.read_csv(DATA_FILE)
X = df.iloc[:, :-1].values  # first 10 columns  -> features
y = df.iloc[:, -1].values  # last  column     -> target  (red-shift)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

print(f"  Training instances: {X_train.shape[0]}")
print(f"  Test instances    : {X_test.shape[0]}")

# ------------------------------------------------------------
# Step 2.  Train the *given* LightGBM configuration and monitor
#          RMSE on a 90 / 10 sub-split of the training data
# ------------------------------------------------------------
print("\nStep 2 – Fitting the reference LightGBM model and plotting learning curves")

# 90 / 10 split of *training* portion for on-the-fly validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=RANDOM_STATE
)

eval_metric_name = "rmse"

lgbm_ref = LGBMRegressor(
    objective="regression",  # Equivalent to reg:squarederror
    # parameters specified by the exercise (adapted for LightGBM)
    feature_fraction=0.5,  # XGBoost: colsample_bytree
    learning_rate=0.1,
    max_depth=4,
    lambda_l2=1.0,  # XGBoost: reg_lambda
    n_estimators=500,
    metric=eval_metric_name,  # LightGBM uses 'metric'
    # nice-to-have settings
    random_state=RANDOM_STATE,
    verbose=-1,  # Suppress LightGBM output
)

eval_set = [(X_tr, y_tr), (X_val, y_val)]
eval_names = ["train", "validation"]

print("  -> training … (this may take a moment)")
start = time.time()
lgbm_ref.fit(
    X_tr,
    y_tr,
    eval_set=eval_set,
    eval_names=eval_names,
    # verbose is deprecated in fit, set in constructor or use callbacks
    # For explicit verbosity control during fit, use callbacks like lgb.log_evaluation(period=100)
    callbacks=[lgb.log_evaluation(period=0)],  # Suppress iteration messages
)
print(f"     training finished in {time.time() - start:4.1f}s")

# --- Plot RMSE vs. boosting iteration -----------------------
results = lgbm_ref.evals_result_
# Accessing results: results['train']['rmse'] and results['validation']['rmse']
rmse_tr = results[eval_names[0]][eval_metric_name]
rmse_va = results[eval_names[1]][eval_metric_name]

plt.figure(figsize=(6, 4))
plt.plot(rmse_tr, label="train")
plt.plot(rmse_va, label="validation")
plt.xlabel("Boosting iteration")
plt.ylabel("RMSE")
plt.title("Learning curves – reference LightGBM")
plt.legend()
plt.tight_layout()
plt.savefig("learning_curves_reference_lgbm.png", dpi=180)
print("  -> plot saved as learning_curves_reference_lgbm.png")

# --- Evaluate on the *held-out* test set ---------------------
y_pred_ref = lgbm_ref.predict(X_test)
rmse_ref = root_mean_squared_error(y_test, y_pred_ref)
r2_ref = r2_score(y_test, y_pred_ref)

print(f"  Test RMSE (reference): {rmse_ref:8.5f}")
print(f"  Test R²   (reference): {r2_ref:8.5f}")

# ------------------------------------------------------------
# Step 3.  Hyper-parameter grid search (3-fold CV) plus KNN = 5
# ------------------------------------------------------------
print("\nStep 3 – Grid-search for better hyper-parameters")

param_grid = {
    "n_estimators": [300, 350],
    "max_depth": [
        10,
        11,
        12,
    ],
    "learning_rate": [
        0.001,
        0.01,
        0.02,
    ],
    "feature_fraction": [0.7, 0.9],  # XGBoost: colsample_bytree
    "lambda_l2": [1.5, 1.75],  # XGBoost: reg_lambda
}

lgbm_base = LGBMRegressor(
    objective="regression",
    random_state=RANDOM_STATE,
    verbose=-1,
    metric=eval_metric_name,
)

grid = GridSearchCV(
    estimator=lgbm_base,
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

# ---- Train final LightGBM with the discovered configuration --
print("\n  -> retraining best configuration on the *full* training data")
lgbm_best = LGBMRegressor(
    objective="regression",
    random_state=RANDOM_STATE,
    verbose=-1,
    metric=eval_metric_name,
    **best_params,
)
lgbm_best.fit(X_train, y_train)

y_pred_best = lgbm_best.predict(X_test)
rmse_best = root_mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"\n  Test RMSE (best LightGBM) : {rmse_best:8.5f}")
print(f"  Test R²   (best LightGBM) : {r2_best:8.5f}")

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
#  Final comparison table
# ------------------------------------------------------------
print("\n----------------------------------------------------------")
print("   Model comparison on the *unseen* test set")
print("   (lower RMSE & higher R² are better)")
print("----------------------------------------------------------")
print(f"   Reference LightGBM :  RMSE {rmse_ref:8.5f} | R² {r2_ref:8.5f}")
print(f"   Tuned    LightGBM :  RMSE {rmse_best:8.5f} | R² {r2_best:8.5f}")
print(f"   k-NN (k=5)       :  RMSE {rmse_knn:8.5f} | R² {r2_knn:8.5f}")
print("----------------------------------------------------------\n")

print("Script finished successfully.")
