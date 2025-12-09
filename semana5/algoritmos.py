# algoritmos.py
import numpy as np
from scipy.optimize import approx_fprime

def pontofixo(a, g, TOL=1e-8, maxiter=100000):
    """
    Iteração de ponto fixo:
    x_{n+1} = g(x_n)
    Retorna a aproximação do ponto fixo iniciando em a.
    Critério de parada: |x_{n+1} - x_n| <= TOL ou número máximo de iterações atingido.
    """
    x_prev = a
    x = g(x_prev)
    it = 1
    while abs(x - x_prev) > TOL and it < maxiter:
        x_prev, x = x, g(x)
        it += 1
    return x

def newton_raphson(a, f, TOL=1e-8, df=None, maxiter=100000):
    """
    Newton-Raphson via reformulação como ponto fixo:
    g(x) = x - f(x)/f'(x)
    Se df None, calcula numericamente a derivada usando approx_fprime.
    """
    if df is None:
        def dfn(x):
            return approx_fprime(np.array([x]), lambda v: f(v[0]))[0]
    else:
        dfn = df
    def g(x): 
        return x - f(x) / dfn(x)
    return pontofixo(a, g, TOL=TOL, maxiter=maxiter)

def secante(a, b, f, TOL=1e-8, maxiter=100000):
    """
    Método da secante.
    g(a,b) = (a*f(b) - b*f(a)) / (f(b) - f(a))
    Retorna aproximação da raiz começando com a e b.
    """
    def g2(a, b):
        denom = (f(b) - f(a))
        if denom == 0:
            raise ZeroDivisionError("Denominador zero na iteração da secante.")
        return (a * f(b) - b * f(a)) / denom

    x_prev = a
    x = g2(a, b)
    it = 1
    while abs(x - b) > TOL and it < maxiter:
        x_prev, b = b, x
        x = g2(x_prev, b)
        it += 1
    return x
