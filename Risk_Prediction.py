"""
Risk_Prediction.py
==================
Logistic Regression model (built from scratch with NumPy) for illness risk prediction.
Features: age, sleep_duration, water_intake, stress, screentime
"""

import numpy as np
import pandas as pd

# ── Global model state ────────────────────────────────────────────────────────
_theta = None
_mean_X_train = None
_std_X_train = None
_is_trained = False

FEATURE_COLS = ['age', 'sleep_duration', 'water_intake', 'stress', 'screentime']

SLEEP_MAP = {
    '<3': 2.5, '3-4': 3.5, '4-5': 4.5,
    '5-6': 5.5, '6-7': 6.5, '7-8': 7.5, '>8': 8.5
}

# ── Core math helpers ─────────────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hypothesis(theta, X):
    return np.matmul(X, theta)

def cost(X, y, theta):
    z = hypothesis(theta, X)
    h = sigmoid(z)
    m = len(y)
    return (1 / m) * ((-y * np.log(h + 1e-15) - (1 - y) * np.log(1 - h + 1e-15)).sum())

def gradient(X, y, theta):
    z = hypothesis(theta, X)
    h = sigmoid(z)
    m = len(y)
    return (1 / m) * np.dot(X.T, (h - y))

# ── Training ──────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, lr: float = 0.1, iterations: int = 1000):
    """
    Train the logistic regression model on a cleaned DataFrame.
    Returns (theta, mean_X_train, std_X_train, cost_history).
    """
    global _theta, _mean_X_train, _std_X_train, _is_trained

    X = df[FEATURE_COLS].to_numpy()
    y = df['illness'].to_numpy()

    # Shuffle
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split 80/20
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale
    _mean_X_train = np.mean(X_train, axis=0)
    _std_X_train = np.std(X_train, axis=0)
    X_scaled = (X_train - _mean_X_train) / _std_X_train
    X_new = np.concatenate((np.ones((X_scaled.shape[0], 1)), X_scaled), axis=1)

    X_test_scaled = (X_test - _mean_X_train) / _std_X_train
    X_test_new = np.concatenate((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled), axis=1)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    _theta = np.zeros((X_new.shape[1], 1))
    cost_history = []

    for _ in range(iterations):
        c = cost(X_new, y_train, _theta)
        grad = gradient(X_new, y_train, _theta)
        _theta -= lr * grad
        cost_history.append(c)

    _is_trained = True

    # Evaluate
    h_test = sigmoid(hypothesis(_theta, X_test_new))
    y_pred_test = (h_test >= 0.5).astype(float)
    accuracy = float(np.mean(y_pred_test == y_test))

    TP = float(np.sum((y_pred_test == 1) & (y_test == 1)))
    FP = float(np.sum((y_pred_test == 1) & (y_test == 0)))
    FN = float(np.sum((y_pred_test == 0) & (y_test == 1)))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return _theta, _mean_X_train, _std_X_train, cost_history, metrics


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(X: np.ndarray, theta=None, mean_X=None, std_X=None):
    """
    Predict illness risk for input array X of shape (n_samples, 5).
    Returns (label_array, probability_array).
    """
    theta = theta if theta is not None else _theta
    mean_X = mean_X if mean_X is not None else _mean_X_train
    std_X = std_X if std_X is not None else _std_X_train

    if theta is None:
        raise RuntimeError("Model has not been trained yet. Call train() first.")

    X_scaled = (X - mean_X) / std_X
    X_new = np.concatenate((np.ones((X_scaled.shape[0], 1)), X_scaled), axis=1)
    z = hypothesis(theta, X_new)
    h = sigmoid(z)
    labels = (h >= 0.5).astype(float)
    return labels, h


def predict_single(age, sleep_duration, water_intake, stress, screentime):
    """
    Convenience wrapper: predict for one individual.
    Returns (label: int, probability: float).
    """
    X = np.array([[age, sleep_duration, water_intake, stress, screentime]])
    labels, probs = predict(X)
    return int(labels[0][0]), float(probs[0][0])


# ── Data loading & cleaning ───────────────────────────────────────────────────
def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load raw CSV, clean and return DataFrame ready for training."""
    df = pd.read_csv(filepath)
    if 'Timestamp' in df.columns:
        df.drop(['Timestamp'], axis=1, inplace=True)
    df['illness'] = df['illness'].map({'Yes': 1, 'No': 0})
    df['sleep_duration'] = df['sleep_duration'].map(SLEEP_MAP)
    df.dropna(inplace=True)

    # IQR outlier removal
    cols = ['sleep_duration', 'water_intake', 'screentime']
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[mask].reset_index(drop=True)