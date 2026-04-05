def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    XtX = X.T @ X
    I = np.eye(d)
    regularized = XtX + lam * I
    inv = np.linalg.inv(regularized)
    Xty = X.T @ y
    w = inv @Xty
    return w.tolist()