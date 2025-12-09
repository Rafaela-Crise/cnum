import numpy as np
from scipy.optimize import approx_fprime

def JN(x, F, eps=1e-8):
    """
    Jacobiana numérica de F em x usando approx_fprime.
    F : função R^n -> R^n
    Retorna matriz n x n onde J[i,j] = dF_i/dx_j
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    Jnum = np.zeros((n, n), dtype=float)

    for i in range(n):
        fi = (lambda v, ii=i: F(v)[ii])  # captura ii por default para evitar problema de closure
        Jnum[i, :] = approx_fprime(x, fi, epsilon=eps)

    return Jnum

def G(x, F, J):
    """
    Iteração de ponto fixo baseada em Newton com Jacobiana analítica J(x).
    G(x) = x - J(x)^{-1} F(x)
    """
    x = np.asarray(x, dtype=float)
    return x - np.linalg.inv(J(x)) @ F(x)

def GN(x, F, eps=1e-8):
    """
    Versão que usa Jacobiana numérica aprox (internamente chama JN).
    Retorna x - JN(x)^{-1} F(x)
    """
    x = np.asarray(x, dtype=float)
    Jnum = JN(x, F, eps=eps)
    return x - np.linalg.inv(Jnum) @ F(x)

def fixed_point(a, g, TOL=1e-10, iter=100):
    """
    Iteração de ponto fixo: x_{k+1} = g(x_k)
    a : chute inicial (vetor)
    g : função de ponto-fixo (R^n -> R^n)
    retorna o vetor solução aproximado.
    """
    a = np.asarray(a, dtype=float)
    x = g(a)
    i = 1
    while np.linalg.norm(x - a) > TOL and i < iter:
        a = x
        x = g(a)
        i += 1
    return x
