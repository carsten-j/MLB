# xgboost_improvements_example.py
# Advanced XGBoost improvements for better RMSE performance

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def load_and_prepare_data(data_file="quasars.csv"):
    """Load and prepare the data with optional feature engineering."""
    df = pd.read_csv(data_file, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)


def create_enhanced_features(X_train, X_test):
    """Create enhanced features using scaling and polynomial terms."""
    # 1. Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Add polynomial features for top 3 most important features
    # (In practice, you'd identify these from feature importance)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

    # Use only first 3 features for polynomial to avoid dimensionality explosion
    X_train_top = X_train_scaled[:, :3]
    X_test_top = X_test_scaled[:, :3]

    X_train_poly = poly.fit_transform(X_train_top)
    X_test_poly = poly.transform(X_test_top)

    # Combine original scaled features with polynomial features
    X_train_enhanced = np.concatenate([X_train_scaled, X_train_poly], axis=1)
    X_test_enhanced = np.concatenate([X_test_scaled, X_test_poly], axis=1)

    return X_train_enhanced, X_test_enhanced


def advanced_xgboost_tuning(X_train, y_train):
    """Perform advanced hyperparameter tuning using RandomizedSearchCV."""

    # Expanded parameter space
    param_distributions = {
        "n_estimators": [200, 300, 500, 700, 1000],
        "max_depth": [4, 6, 8, 10, 12],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8],
        "colsample_bylevel": [0.5, 0.7, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1, 2],
        "reg_lambda": [0.5, 1, 1.5, 2, 3],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3],
        # "booster": ["gbtree", "dart"],  # Try different boosters
    }

    xgb_base = XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_STATE, verbosity=1
    )

    # Use RandomizedSearchCV for efficiency
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=100,  # Try 100 random combinations
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=2,
    )

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_


def ensemble_predictions(X_train, y_train, X_test, best_xgb_params):
    """Create ensemble of different models for better predictions."""

    # XGBoost with best parameters
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        verbosity=0,
        **best_xgb_params,
    )

    # Random Forest as complementary model
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    )

    # Fit models
    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Get predictions
    xgb_pred = xgb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Ensemble (weighted average - you can tune these weights)
    ensemble_pred = 0.7 * xgb_pred + 0.3 * rf_pred

    return xgb_pred, rf_pred, ensemble_pred


def main():
    """Main function demonstrating advanced XGBoost improvements."""
    print("Advanced XGBoost Improvements Demo")
    print("=" * 50)

    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Original features: {X_train.shape[1]}")

    # Create enhanced features
    X_train_enhanced, X_test_enhanced = create_enhanced_features(X_train, X_test)
    print(f"Enhanced features: {X_train_enhanced.shape[1]}")

    # Advanced hyperparameter tuning
    print("\nPerforming advanced hyperparameter tuning...")
    best_model, best_params = advanced_xgboost_tuning(X_train_enhanced, y_train)
    print(f"Best parameters: {best_params}")

    # Create ensemble predictions
    print("\nCreating ensemble predictions...")
    xgb_pred, rf_pred, ensemble_pred = ensemble_predictions(
        X_train_enhanced, y_train, X_test_enhanced, best_params
    )

    # Evaluate all approaches
    print("\nResults Comparison:")
    print("-" * 30)

    # XGBoost only
    rmse_xgb = root_mean_squared_error(y_test, xgb_pred)
    r2_xgb = r2_score(y_test, xgb_pred)
    print(f"XGBoost:  RMSE {rmse_xgb:.5f} | R² {r2_xgb:.5f}")

    # Random Forest only
    rmse_rf = root_mean_squared_error(y_test, rf_pred)
    r2_rf = r2_score(y_test, rf_pred)
    print(f"RF:       RMSE {rmse_rf:.5f} | R² {r2_rf:.5f}")

    # Ensemble
    rmse_ensemble = root_mean_squared_error(y_test, ensemble_pred)
    r2_ensemble = r2_score(y_test, ensemble_pred)
    print(f"Ensemble: RMSE {rmse_ensemble:.5f} | R² {r2_ensemble:.5f}")


if __name__ == "__main__":
    main()
