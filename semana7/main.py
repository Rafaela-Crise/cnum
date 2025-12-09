# main.py
import numpy as np
from algoritmos import G, GN, fixed_point

def atividade_1():
    print("-- Atividade 1 --")
    def F(x):
        x1, x2, x3 = x
        return np.array([
            2.0 * x1 - x2 - np.cos(x1),
            -x1 + 2.0 * x2 - x3 - np.cos(x2),
            -x2 + x3 - np.cos(x3),
        ], dtype=float)

    def J(x):
        x1, x2, x3 = x
        return np.array([
            [2.0 + np.sin(x1), -1.0, 0.0],
            [-1.0, 2.0 + np.sin(x2), -1.0],
            [0.0, -1.0, 1.0 + np.sin(x3)],
        ], dtype=float)

    x0 = np.array([1.0, 1.0, 1.0], dtype=float)

    # Newton com J analítica via ponto-fixo
    r1 = fixed_point(x0, lambda xx: G(xx, F, J), TOL=1e-12, iter=200)
    print("Solução (Newton com J analítica):", r1)

    # Newton com Jacobiana numérica
    r2 = fixed_point(x0, lambda xx: GN(xx, F), TOL=1e-12, iter=200)
    print("Solução (Newton com J numérica):", r2)
    print()

def atividade_2():
    print("-- Atividade 2 --")
    def F(x):
        X, Y, Z = x
        return np.array([
            6.0*X - 2.0*Y + np.exp(Z) - 2.0,
            np.sin(X) - Y + Z,
            np.sin(X) + 2.0*Y + 3.0*Z - 1.0
        ], dtype=float)

    def J(x):
        X, Y, Z = x
        return np.array([
            [6.0, -2.0, np.exp(Z)],
            [np.cos(X), -1.0, 1.0],
            [np.cos(X), 2.0, 3.0]
        ], dtype=float)

    x0 = np.array([0.0, 0.0, 0.0], dtype=float)
    r = fixed_point(x0, lambda xx: G(xx, F, J), TOL=1e-12, iter=200)
    print("Solução aproximada (x, y, z):", r)
    print("Valores esperados: x≈0.259751, y≈0.302736, z≈0.045896")
    print()

def atividade_3():
    print("-- Atividade 3 --")
    def F(v):
        x, y = v
        return np.array([
            x**2/8.0 + (y - 1.0)**2/5.0 - 1.0,
            np.arctan(x) + x - y - y**3
        ], dtype=float)

    def J(v):
        x, y = v
        return np.array([
            [x/4.0, 2.0*(y - 1.0)/5.0],
            [1.0/(1.0 + x**2) + 1.0, -1.0 - 3.0*y**2]
        ], dtype=float)

    guesses = [np.array([-1.2, -1.0], dtype=float), np.array([2.8, 1.38], dtype=float)]
    results = []
    for g0 in guesses:
        r = fixed_point(g0, lambda xx: G(xx, F, J), TOL=1e-12, iter=200)
        results.append(r)
        print(f"Chute inicial {g0} -> raiz: {r}")
    print("Valores esperados: (-1.2085435, -1.0216674) e (2.7871115, 1.3807962)")
    print()

def atividade_4():
    print("-- Atividade 4 --")

    def dC1(x):
        # derivada de C1
        return 0.3 + 2.0*1e-4*x + 4.0*3.4e-9*(x**3)

    def dC2(x):
        return 0.25 + 2.0*2e-4*x + 3.0*4.3e-7*(x**2)

    def dC3(x):
        return 0.19 + 2.0*5e-4*x + 4.0*1.1e-7*(x**3)

    def F(vars):
        x1, x2, x3, lam = vars
        return np.array([
            dC1(x1) - lam,
            dC2(x2) - lam,
            dC3(x3) - lam,
            x1 + x2 + x3 - 1500.0
        ], dtype=float)

    def J(vars):
        x1, x2, x3, lam = vars
        # Jacobiana 4x4
        return np.array([
            [ 2.0*1e-4 + 12.0*3.4e-9*(x1**2), 0.0, 0.0, -1.0],
            [ 0.0, 2.0*2e-4 + 6.0*4.3e-7*(x2), 0.0, -1.0],
            [ 0.0, 0.0, 2.0*5e-4 + 16.0*1.1e-7*(x3**3)/4.0, -1.0],
            [1.0, 1.0, 1.0, 0.0]
        ], dtype=float)

    def J_explicit(vars):
        x1, x2, x3, lam = vars
       
        ddC1 = 2.0*1e-4 + 12.0*3.4e-9*(x1**2)
       
        ddC2 = 2.0*2e-4 + 6.0*4.3e-7*(x2)
       
        ddC3 = 2.0*5e-4 + 12.0*1.1e-7*(x3**2)
        return np.array([
            [ddC1, 0.0, 0.0, -1.0],
            [0.0, ddC2, 0.0, -1.0],
            [0.0, 0.0, ddC3, -1.0],
            [1.0, 1.0, 1.0, 0.0]
        ], dtype=float)

    x0 = np.array([500.0, 800.0, 200.0, 0.25], dtype=float)
    r = fixed_point(x0, lambda xx: G(xx, F, J_explicit), TOL=1e-12, iter=300)
    x1, x2, x3, lam = r
    print("Solução (x1, x2, x3):", np.array([x1, x2, x3]))
    print("Lambda:", lam)
    print("Valor esperado aproximado: (453.62, 901.94, 144.43)")
    print()

def main():
    atividade_1()
    atividade_2()
    atividade_3()
    atividade_4()

if __name__ == "__main__":
    main()
