import numpy as np
from typing import List, Tuple


class SimpleDecisionTree:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.depth = 0
    
    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, depth: int = 0):
        self.depth = depth
        n_samples, n_features = X.shape
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(gradients)) == 1):
            self.value = -np.sum(gradients) / (np.sum(hessians) + 1e-8)
            return
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._calculate_gain(X, gradients, hessians, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_gain > 0:
            self.feature_idx = best_feature
            self.threshold = best_threshold
            
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            
            self.left = SimpleDecisionTree(self.max_depth, self.min_samples_split)
            self.right = SimpleDecisionTree(self.max_depth, self.min_samples_split)
            
            self.left.fit(X[left_mask], gradients[left_mask], hessians[left_mask], depth + 1)
            self.right.fit(X[right_mask], gradients[right_mask], hessians[right_mask], depth + 1)
        else:
            self.value = -np.sum(gradients) / (np.sum(hessians) + 1e-8)
    
    def _calculate_gain(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, 
                      feature_idx: int, threshold: float) -> float:
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        G_L = np.sum(gradients[left_mask])
        H_L = np.sum(hessians[left_mask])
        G_R = np.sum(gradients[right_mask])
        H_R = np.sum(hessians[right_mask])
        G = np.sum(gradients)
        H = np.sum(hessians)
        
        gain = 0.5 * ((G_L**2 / (H_L + 1e-8)) + (G_R**2 / (H_R + 1e-8)) - (G**2 / (H + 1e-8)))
        return gain
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.value is not None:
            return np.full(X.shape[0], self.value)
        
        predictions = np.zeros(X.shape[0])
        left_mask = X[:, self.feature_idx] <= self.threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) > 0:
            predictions[left_mask] = self.left.predict(X[left_mask])
        if np.sum(right_mask) > 0:
            predictions[right_mask] = self.right.predict(X[right_mask])
        
        return predictions


class SimpleXGBoost:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3, min_samples_split: int = 2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees: List[SimpleDecisionTree] = []
        self.base_prediction = 0.0
    
    def _calculate_gradients_hessians(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gradients = 2 * (y_pred - y_true)
        hessians = np.full_like(y_true, 2.0)
        return gradients, hessians
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_prediction = np.mean(y)
        y_pred = np.full_like(y, self.base_prediction)
        
        for i in range(self.n_estimators):
            gradients, hessians = self._calculate_gradients_hessians(y, y_pred)
            
            tree = SimpleDecisionTree(self.max_depth, self.min_samples_split)
            tree.fit(X, gradients, hessians)
            
            tree_predictions = tree.predict(X)
            y_pred += self.learning_rate * tree_predictions
            
            self.trees.append(tree)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full(X.shape[0], self.base_prediction)
        
        for tree in self.trees:
            tree_pred = tree.predict(X)
            predictions += self.learning_rate * tree_pred
        
        return predictions


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    
    model = SimpleXGBoost(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    
    predictions = model.predict(X)
    mse = np.mean((y - predictions) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")
    
    X_test = np.array([[1.0, 2.0], [-1.0, 1.0]])
    test_predictions = model.predict(X_test)
    print(f"Test predictions: {test_predictions}")