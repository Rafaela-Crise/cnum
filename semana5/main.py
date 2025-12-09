# main.py
import numpy as np
from algoritmos import pontofixo, newton_raphson, secante

# Atividade 1 
f1 = lambda x: np.e**x - x - 2
g1 = lambda x: np.e**x - 2

# Atividade 2 
f2 = lambda x: np.cos(x) - x**2
# Newton-Raphson reorganizado como ponto-fixo:
g2 = lambda x: x + (np.cos(x) - x**2) / (np.sin(x) + 2*x)

# Atividade 3 
# e^{-x^2} = 2x  ->  x = (1/2) e^{-x^2}
g3 = lambda x: 0.5 * np.exp(-x**2)
f3 = lambda x: np.exp(-x**2) - 2*x

def main():
    print("-- Atividade 1 --")
    r = pontofixo(-1.8, g1, TOL=1e-8)
    print(f"raiz ponto fixo = {r:.10f}")    # mostra precisão
    r = newton_raphson(-1.8, f1, df=lambda x: np.e**x - 1, TOL=1e-12)
    print(f"raiz newton-raphson df = {r:.10f}")
    r = newton_raphson(-1.8, f1, TOL=1e-12)
    print(f"raiz newton-raphson (num df) = {r:.10f}")
    r = secante(-1.8, -1.7, f1, TOL=1e-12)
    print(f"raiz secante = {r:.10f}")

    print("\n-- Atividade 2 --")
 
    def pontofixo_criterio(a, g, tol=5e-6, maxiter=100000):
        x_prev = a
        x = g(x_prev)
        it = 1
        while abs(x - x_prev) > tol and it < maxiter:
            x_prev, x = x, g(x)
            it += 1
        return x, it
    r2, it2 = pontofixo_criterio(1.0, g2, tol=5e-6)
    print(f"raiz (atividade 2) = {r2:.5f}  (it {it2})")  # 5 casas conforme pedido
    # também mostramos mais dígitos
    print(f"raiz (atividade 2, mais dígitos) = {r2:.10f}")

    print("\n-- Atividade 3 --")
    # ponto fixo com chute inicial 0.4 (próximo da solução)
    r3 = pontofixo(0.4, g3, TOL=1e-8)
    print(f"raiz (atividade 3) = {r3:.7f}")

if __name__ == "__main__":
    main()
