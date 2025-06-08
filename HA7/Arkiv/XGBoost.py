"""
Machine Learning B - Home Assignment 7
XGBoost Regression for Quasar Redshift Prediction

This script implements XGBoost regression models to predict redshift values
for quasars based on their photometric features.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


def load_and_prepare_data(file_path="quasars.csv"):
    """
    Load the quasar dataset and prepare features and targets

    Args:
        file_path (str): Path to the quasars.csv file

    Returns:
        X (np.array): Feature matrix (10 features)
        y (np.array): Target vector (redshift values)
    """
    print("Loading dataset...")

    # Load the dataset
    data = pd.read_csv(file_path)

    print(f"Dataset shape: {data.shape}")
    print(f"Features: {data.columns[:-1].tolist()}")
    print(f"Target: {data.columns[-1]}")

    # Separate features and target
    X = data.iloc[:, :-1].values  # First 10 columns (features)
    y = data.iloc[:, -1].values  # Last column (target - redshift)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(
        f"Target statistics: mean={y.mean():.3f}, std={y.std():.3f}, min={y.min():.3f}, max={y.max():.3f}"
    )

    return X, y


def part1_data_splitting(X, y):
    """
    Part 1: Split dataset into 80% training and 20% test sets

    Args:
        X (np.array): Feature matrix
        y (np.array): Target vector

    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    print("\n" + "=" * 50)
    print("PART 1: Data Splitting")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(
        f"Training set size: {X_train.shape[0]} ({X_train.shape[0] / X.shape[0] * 100:.1f}%)"
    )
    print(
        f"Test set size: {X_test.shape[0]} ({X_test.shape[0] / X.shape[0] * 100:.1f}%)"
    )

    return X_train, X_test, y_train, y_test


def part2_baseline_xgboost(X_train, y_train, X_test, y_test):
    """
    Part 2: Train XGBoost with baseline parameters and monitoring

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        model: Trained XGBoost model
        results: Dictionary with evaluation results
    """
    print("\n" + "=" * 50)
    print("PART 2: Baseline XGBoost Model")
    print("=" * 50)

    # Further split training data: 90% for training, 10% for validation
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(f"Training subset size: {X_train_fit.shape[0]}")
    print(f"Validation subset size: {X_val.shape[0]}")

    # Initialize XGBoost model with baseline parameters
    model = xgb.XGBRegressor(
        colsample_bytree=0.5,
        learning_rate=0.1,
        max_depth=4,
        reg_lambda=1,  # lambda parameter is reg_lambda in sklearn API
        n_estimators=500,
        random_state=42,
        eval_metric="rmse",
    )

    # Fit model with validation monitoring
    print("\nTraining XGBoost model with validation monitoring...")
    model.fit(
        X_train_fit,
        y_train_fit,
        eval_set=[(X_train_fit, y_train_fit), (X_val, y_val)],
        verbose=False,
    )

    # Plot training and validation RMSE
    results = model.evals_result()
    train_rmse = results["validation_0"]["rmse"]
    val_rmse = results["validation_1"]["rmse"]

    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse, label="Training RMSE", linewidth=2)
    plt.plot(val_rmse, label="Validation RMSE", linewidth=2)
    plt.xlabel("Boosting Iterations")
    plt.ylabel("RMSE")
    plt.title("XGBoost Training Progress - Baseline Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    print("\nBaseline Model Results:")
    print(f"Final Training RMSE: {train_rmse[-1]:.4f}")
    print(f"Final Validation RMSE: {val_rmse[-1]:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")

    return model, {
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "train_rmse": train_rmse[-1],
        "val_rmse": val_rmse[-1],
    }


def part3_grid_search_optimization(X_train, y_train, X_test, y_test):
    """
    Part 3: Grid search for optimal parameters and comparison with KNN

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        best_model: Best XGBoost model from grid search
        results: Dictionary with evaluation results
    """
    print("\n" + "=" * 50)
    print("PART 3: Grid Search Optimization")
    print("=" * 50)

    # Define parameter grid extending around baseline values
    param_grid = {
        "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6],
        "reg_lambda": [0.5, 1, 2, 5],
        "n_estimators": [200, 500, 800],
    }

    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    total_combinations = np.prod([len(values) for values in param_grid.values()])
    print(f"\nTotal parameter combinations: {total_combinations}")

    # Initialize XGBoost model
    xgb_model = xgb.XGBRegressor(random_state=42, eval_metric="rmse")

    # Perform grid search with 3-fold cross-validation
    print("\nPerforming 3-fold cross-validation grid search...")
    print("This may take several minutes...")

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    # Get best parameters and model
    best_params = grid_search.best_params_
    best_cv_score = np.sqrt(-grid_search.best_score_)  # Convert to RMSE

    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best CV RMSE: {best_cv_score:.4f}")

    # Refit best model on all training data
    print("\nRefitting best model on all training data...")
    best_model = xgb.XGBRegressor(**best_params, random_state=42, eval_metric="rmse")
    best_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_best = best_model.predict(X_test)
    best_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
    best_test_r2 = r2_score(y_test, y_pred_best)

    print("\nOptimized Model Results:")
    print(f"Test RMSE: {best_test_rmse:.4f}")
    print(f"Test R² Score: {best_test_r2:.4f}")

    # Compare with k=5 Nearest Neighbors
    print("\n" + "-" * 30)
    print("Comparison with k=5 Nearest Neighbors")
    print("-" * 30)

    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    knn_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
    knn_test_r2 = r2_score(y_test, y_pred_knn)

    print("KNN (k=5) Results:")
    print(f"Test RMSE: {knn_test_rmse:.4f}")
    print(f"Test R² Score: {knn_test_r2:.4f}")

    # Comparison summary
    print("\n" + "=" * 50)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 50)

    print(f"{'Model':<20} {'Test RMSE':<12} {'Test R²':<12} {'Improvement'}")
    print("-" * 60)
    print(f"{'KNN (k=5)':<20} {knn_test_rmse:<12.4f} {knn_test_r2:<12.4f} {'Baseline'}")
    print(
        f"{'XGBoost (optimized)':<20} {best_test_rmse:<12.4f} {best_test_r2:<12.4f} {((knn_test_rmse - best_test_rmse) / knn_test_rmse * 100):+.1f}% RMSE"
    )

    if best_test_rmse < knn_test_rmse:
        print(
            f"\n✅ XGBoost beats KNN baseline by {((knn_test_rmse - best_test_rmse) / knn_test_rmse * 100):.1f}% in RMSE!"
        )
    else:
        print(
            f"\n❌ XGBoost does not beat KNN baseline (worse by {((best_test_rmse - knn_test_rmse) / knn_test_rmse * 100):.1f}% in RMSE)"
        )

    return best_model, {
        "best_params": best_params,
        "best_cv_rmse": best_cv_score,
        "test_rmse": best_test_rmse,
        "test_r2": best_test_r2,
        "knn_rmse": knn_test_rmse,
        "knn_r2": knn_test_r2,
    }


def create_prediction_plot(y_true, y_pred, title, model_name):
    """Create scatter plot of predictions vs actual values"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )

    plt.xlabel("Actual Redshift")
    plt.ylabel("Predicted Redshift")
    plt.title(f"{title}\n{model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calculate and display R²
    r2 = r2_score(y_true, y_pred)
    plt.text(
        0.05,
        0.95,
        f"R² = {r2:.4f}",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.pdf", dpi=600, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function"""
    print("XGBoost Regression for Quasar Redshift Prediction")
    print("=" * 55)

    try:
        # Load and prepare data
        X, y = load_and_prepare_data("quasars.csv")

        # Part 1: Data splitting
        X_train, X_test, y_train, y_test = part1_data_splitting(X, y)

        # Part 2: Baseline XGBoost model
        baseline_model, baseline_results = part2_baseline_xgboost(
            X_train, y_train, X_test, y_test
        )

        # Part 3: Grid search optimization
        best_model, optimization_results = part3_grid_search_optimization(
            X_train, y_train, X_test, y_test
        )

        # Create prediction plots for final models
        y_pred_baseline = baseline_model.predict(X_test)
        y_pred_optimized = best_model.predict(X_test)

        create_prediction_plot(
            y_test,
            y_pred_baseline,
            "Baseline XGBoost Predictions vs Actual",
            f"RMSE: {baseline_results['test_rmse']:.4f}",
        )

        create_prediction_plot(
            y_test,
            y_pred_optimized,
            "Optimized XGBoost Predictions vs Actual",
            f"RMSE: {optimization_results['test_rmse']:.4f}",
        )

        # Final summary
        print("\n" + "=" * 60)
        print("EXERCISE COMPLETION SUMMARY")
        print("=" * 60)
        print("✅ Part 1: Dataset loaded and split (80% train, 20% test)")
        print("✅ Part 2: Baseline XGBoost trained with validation monitoring")
        print("✅ Part 3: Grid search completed and compared with KNN baseline")
        print(
            f"\nBest model improvement over baseline: {((baseline_results['test_rmse'] - optimization_results['test_rmse']) / baseline_results['test_rmse'] * 100):+.1f}% RMSE"
        )

    except FileNotFoundError:
        print("Error: quasars.csv file not found!")
        print("Please ensure the dataset file is in the current directory.")
        print(
            "You can download it from: https://sid.erda.dk/cgi-sid/ls.py?share_id=c9SgMJSGik"
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
