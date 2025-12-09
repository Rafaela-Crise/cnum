import math
import numpy as np

def atividade1_bisseccao(E=500.125, K=272.975, a=273.0, b=400.0, tol=1e-14, maxiter=200):
    def f(T):
        return 5.67e-8 * T**4 + 0.4*(T - K) - E
    fa, fb = f(a), f(b)
    if fa*fb > 0:
        raise ValueError("Escolha de intervalo não muda de sinal: f(a)*f(b)>0")
    for i in range(maxiter):
        m = 0.5*(a+b)
        fm = f(m)
        if abs(fm) < tol or 0.5*(b-a) < tol:
            return m, i+1
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b), maxiter

def gauss_seidel(A, b, x0=None, tol=1e-12, maxiter=10000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    for k in range(maxiter):
        x_old = x.copy()
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i,j] * x[j]
            x[i] = (b[i] - s) / A[i,i]
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k+1
    return x, maxiter

def newton_system(F, J, x0, tol=1e-14, maxiter=200):
    x = x0.astype(float).copy()
    for k in range(maxiter):
        Fx = F(x)
        Jx = J(x)
        try:
            delta = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            raise RuntimeError("Jacobian singular")
        x = x + delta
        if np.linalg.norm(delta, ord=np.inf) < tol:
            return x, k+1
    return x, maxiter

# Atividade 1
T_min, iters1 = atividade1_bisseccao()

# Atividade 2
A2 = np.array([[17.0, -2.0, -3.0],
               [-5.0, 21.0, -2.0],
               [-5.0, -5.0, 22.0]])
b2 = np.array([500.0, 200.0, 300.0])
R, iters2 = gauss_seidel(A2, b2, tol=1e-12)

# Atividade 3
A3 = np.array([[20.0, 10.0],
               [10.0, 20.0]])
b3 = np.array([100.0, 100.0])
Ivec, iters3 = gauss_seidel(A3, b3, tol=1e-12)
I1, I2 = Ivec[0], Ivec[1]
I_R3 = I1 + I2 

# Atividade 4
E1 = 0.01753
E2 = 0.00254
def F(vec):
    T1, T2 = vec
    f1 = (T1**4 + 0.06823*T1) - (T2**4 + 0.05848*T2) - E1
    f2 = (T1**4 + 0.05848*T1) - (2*(T2**4) + 0.11696*T2) - E2
    return np.array([f1, f2])

def J(vec):
    T1, T2 = vec
    df1dT1 = 4*T1**3 + 0.06823
    df1dT2 = - (4*T2**3 + 0.05848)
    df2dT1 = 4*T1**3 + 0.05848
    df2dT2 = - (8*T2**3 + 0.11696)
    return np.array([[df1dT1, df1dT2],
                     [df2dT1, df2dT2]])

x0 = np.array([0.3, 0.18])
Tsol, iters4 = newton_system(F, J, x0, tol=1e-14, maxiter=500)

print("Resultados:")
print()
print("Atividade 1 (bissecção):")
print(f"Temperatura mínima da placa T = {T_min:.17f} K (iter={iters1})")
print("Resposta aproximada esperada: 304.56801011987010952")
print()
print("Atividade 2 (Gauss-Seidel):")
print(f"R1 = {R[0]:.14f}, R2 = {R[1]:.14f}, R3 = {R[2]:.14f} (iter={iters2})")
print("Respostas aproximadas esperadas: R1=36.56081655798128, R2=20.768358378225123, R3=26.66572157641055")
print()
print("Atividade 3 (Gauss-Seidel):")
print(f"I1 = {I1:.6f} A, I2 = {I2:.6f} A (iter={iters3})")
print(f"I_R3 (I1 + I2) = {I_R3:.6f} A")
print("Resposta aproximada esperada: I_R3 = 6.6667 A")
print()
print("Atividade 4 (Newton-Raphson):")
print(f"T1 = {Tsol[0]:.5f}, T2 = {Tsol[1]:.6f} (iter={iters4})")
print("Respostas aproximadas esperadas: T1=0.30543, T2=0.185261")
