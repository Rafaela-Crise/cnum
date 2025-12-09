from algoritmos import (
    pontofixo,
    newton_raphson,
    secante,
    lu,
    jacobi,
    seidel,
    sig_fig,
)
import numpy as np


def atividade1_roots():
    print("-- Atividade 1: ponto fixo / Newton / Secante --")
    f1 = lambda x: np.e**x - x - 2
    g1 = lambda x: np.e**x - 2
    r = pontofixo(-1.8, g1, TOL=1e-8)
    print(f"raiz ponto fixo = {r}")
    r = newton_raphson(-1.8, f1, df=lambda x: np.e**x - 1, TOL=1e-12)
    print(f"raiz newton-raphson df = {r}")
    r = newton_raphson(-1.8, f1, TOL=1e-12)
    print(f"raiz newton-raphson (aprox grad) = {r}")
    r = secante(-1.8, -1.7, f1, TOL=1e-12)
    print(f"raiz secante = {r}")


def atividade1_linear():
    print("\n-- Atividade 1 (sistema linear) --")
    A = np.array([[1, 1,  1],[4,4,2],[2,1,-1]], dtype=float)
    B = np.array([1,2,0], dtype=float)
    print("Matriz A:\n", A)
    print("Vetor B:", B)
    X_np = np.linalg.solve(A,B)
    X_lu = lu(A,B)
    print("Solução NumPy x:", X_np)
    print("Solução LU x:   ", X_lu)


def atividade2_jacobi_seidel():
    print("\n-- Atividade 2 (Jacobi / Gauss-Seidel) --")
    A = np.array([[5,1,1],[-1,3,-1],[1,2,10]], dtype=float)
    B = np.array([50,10,-30], dtype=float)
    xj = jacobi(A,B, max_iter=10000, TOL=1e-3)
    xs = seidel(A,B, max_iter=10000, TOL=1e-3)
    print("Jacobi (TOL=1e-3):", xj)
    print("Seidel (TOL=1e-3):", xs)


def atividade3_permutacao():
    print("\n-- Atividade 3 (permutação de linhas) --")
    A = np.array([[1,10,3],[4,0,1],[2,1,4]], dtype=float)
    B = np.array([27,6,12], dtype=float)
    # permutar linhas para melhorar diagonal dominante
    perm = [1,0,2]
    A_p = A[perm,:]
    B_p = B[perm]
    xj = jacobi(A_p,B_p, max_iter=10000, TOL=1e-8)
    xs = seidel(A_p,B_p, max_iter=10000, TOL=1e-8)
    print("Jacobi (perm):", xj)
    print("Seidel (perm):", xs)


def atividade4_circuito():
    print("\n-- Atividade 4 (circuito) --")
    V = 127.0
    def build_system(Rs, V1):
        R1,R2,R3,R4,R5,R6,R7,R8 = Rs
        A = np.zeros((4,4), dtype=float)
        b = np.zeros(4, dtype=float)
        A[0,0] = (-1.0/R1) + (-1.0/R2) + (-1.0/R5)
        A[0,3] = 1.0/R2
        b[0] = -V1 / R1
        A[1,0] = 1.0/R2
        A[1,1] = (-1.0/R2) + (-1.0/R3) + (-1.0/R6)
        A[1,2] = 1.0/R3
        A[1,3] = 0.0
        b[1] = 0.0
        A[2,1] = 1.0/R3
        A[2,2] = (-1.0/R3) + (-1.0/R4) + (-1.0/R7)
        A[2,3] = 1.0/R4
        b[2] = 0.0
        A[3,2] = 1.0/R4
        A[3,3] = (-1.0/R4) + (-1.0/R8)
        b[3] = 0.0
        A = -A
        b = -b
        return A,b

    Rs_a = (2,2,2,2,100,100,100,50)
    A_a,b_a = build_system(Rs_a, V)
    x_a = np.linalg.solve(A_a,b_a)
    Rs_b = (2,2,2,2,50,100,100,100)
    A_b,b_b = build_system(Rs_b, V)
    x_b = np.linalg.solve(A_b,b_b)
    print("Caso a (V1..V5) formatado:")
    vals_a = [V, *x_a]
    print([sig_fig(v,4) for v in vals_a])
    print("Caso b (V1..V5) formatado:")
    vals_b = [V, *x_b]
    print([sig_fig(v,4) for v in vals_b])


def main():
    atividade1_roots()
    atividade1_linear()
    atividade2_jacobi_seidel()
    atividade3_permutacao()
    atividade4_circuito()

if __name__ == "__main__":
    main()
