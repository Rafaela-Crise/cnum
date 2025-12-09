import numpy as np
from scipy.optimize import approx_fprime

# --- Métodos de raiz ---

def pontofixo(a, g, TOL=1e-8, max_iter=100000):
    x = g(a)
    it = 0
    while abs(x - a) > TOL and it < max_iter:
        a = x
        x = g(a)
        it += 1
    return x


def newton_raphson(a, f, TOL=1e-8, df=None, max_iter=100000):
    if df is None:
        def dfn(x):
            return approx_fprime(np.array([x]), lambda v: f(v[0]))[0]
    else:
        dfn = df
    g = lambda x: x - f(x) / dfn(x)
    return pontofixo(a, g, TOL, max_iter)


def secante(a, b, f, TOL=1e-8, max_iter=100000):
    g = lambda a, b: (a * f(b) - b * f(a)) / (f(b) - f(a))
    x = g(a, b)
    it = 0
    while abs(x - b) > TOL and it < max_iter:
        a, b = b, x
        x = g(a, b)
        it += 1
    return x

# --- Decomposição LU com pivotamento ---

def lu_pivot(A: np.ndarray):
    A = A.astype(float).copy()
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if np.isclose(U[pivot, k], 0.0):
            raise np.linalg.LinAlgError("Matriz singular ou quase singular.")
        if pivot != k:
            U[[k, pivot], k:] = U[[pivot, k], k:]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    np.fill_diagonal(L, 1.0)
    return P, L, U


def lb(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])
    return Y


def uy(U: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    X = np.zeros(n)
    for i in reversed(range(n)):
        if np.isclose(U[i, i], 0.0):
            raise np.linalg.LinAlgError(
                "U possui pivô nulo; sistema sem solução única."
            )
        X[i] = (Y[i] - np.dot(U[i, i + 1 :], X[i + 1 :])) / U[i, i]
    return X


def lu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    P, L, U = lu_pivot(A)
    Pb = P @ B
    Y = lb(L, Pb)
    X = uy(U, Y)
    return X

# --- Métodos iterativos (Jacobi e Gauss-Seidel) ---

def jacobi(A: np.ndarray, B: np.ndarray, max_iter: int = 1000, TOL: float = 1e-8) -> np.ndarray:
    A = A.astype(float)
    B = B.astype(float)
    n = B.shape[0]
    X = np.zeros(n)
    Xk = np.zeros(n)

    D = np.diag(A)
    if np.any(np.isclose(D, 0)):
        raise ValueError("A possui elementos diagonais nulos; Jacobi pode falhar.")

    R = A - np.diagflat(D)

    for _ in range(max_iter):
        Xk = (B - R @ X) / D
        if np.linalg.norm(Xk - X, ord=2) < TOL:
            return Xk
        X = Xk.copy()
    return X


def seidel(A: np.ndarray, B: np.ndarray, max_iter: int = 1000, TOL: float = 1e-8) -> np.ndarray:
    A = A.astype(float)
    B = B.astype(float)
    n = B.shape[0]
    X = np.zeros(n)

    for _ in range(max_iter):
        Xk = X.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], X[:i])
            s2 = np.dot(A[i, i + 1 :], Xk[i + 1 :])
            X[i] = (B[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(X - Xk, ord=2) < TOL:
            return X
    return X

# --- Utilitários ---

def sig_fig(x, sig=4):
    """Formatar número com 'sig' algarismos significativos (retorna string)."""
    if x == 0:
        return f"{0:.{sig-1}f}"
    else:
        from math import log10, floor
        d = floor(log10(abs(x)))
        decimals = sig - 1 - d
        if decimals < 0:
            fmt = f"{.0}"
        else:
            fmt = f"{{:.{decimals}f}}"
        return fmt.format(round(x, decimals))

