import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def gradient_descent_from_scratch(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000) -> np.ndarray:
    """
    Implement gradient descent for linear regression from scratch.
    Math: theta = theta - alpha * (1/m) * X.T @ (X @ theta - y)
    Explanation: This optimizes the MSE loss J(theta) = (1/(2m)) * ||X theta - y||^2.
    The gradient ∂J/∂theta = (1/m) * X.T (X theta - y).
    We update iteratively until convergence.
    """
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        predictions = X @ theta
        errors = predictions - y
        grad = (1 / m) * X.T @ errors
        theta -= lr * grad
    return theta

def apply_svd(df: pd.DataFrame, k: int = 3):
    """
    SVD for dimensionality reduction.
    Math: A = U @ Sigma @ V.T, where U and V are orthogonal, Sigma diagonal.
    Explanation: Singular values in Sigma indicate importance; we retain top k for compression.
    Variance explained: sum(Sigma[:k]^2) / sum(Sigma^2).
    """
    scaler = StandardScaler()
    X = df.select_dtypes(include=['float64']).drop(columns=['Year', 'Happiness Score'], errors='ignore')
    X_scaled = scaler.fit_transform(X)
    U, Sigma, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    print("Singular values:", Sigma)
    X_reduced = U[:, :k] @ np.diag(Sigma[:k]) @ Vt[:k, :]
    variance_explained = np.sum(Sigma[:k]**2) / np.sum(Sigma**2)
    print(f"Variance explained by top {k} components: {variance_explained:.2f}")
    return X_reduced

def bayesian_inference(df: pd.DataFrame):
    """
    Bayesian linear regression approximation using OLS (for simplicity; full Bayesian via PyMC possible).
    Math: y ~ N(X beta, sigma^2), with priors on beta, sigma.
    Explanation: Here, we use frequentist OLS as proxy; coefficients with p-values indicate significance.
    For true Bayesian, assume normal priors; posterior via sampling.
    """
    X = sm.add_constant(df['GDP per capita'])
    y = df['Happiness Score']
    model = sm.OLS(y, X).fit()
    print(model.summary())
    # Interpretation: Coefficient for GDP shows expected increase in happiness per unit GDP.