import os
import numpy as np
import matplotlib.pyplot as plt

from algoritmos import regressao

OUT_DIR = "10"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_with_basis(x, y, v, A, x_label, y_label, title, filename):
    # prepara x denso para suavizar a curva
    x_vals = np.linspace(min(x) - 0.25, max(x) + 0.25, 400)
    # monta a matriz V para x_vals
    V_dense = v(x_vals)
    y_vals = V_dense @ A

    plt.figure(figsize=(7, 4))
    plt.scatter(x, y, label="Pontos dados")
    plt.plot(x_vals, y_vals, label="Ajuste", linewidth=2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def atividade_1():
    print("-- Atividade 1 --")
    x = np.array([-0.35, 0.15, 0.23, 0.35], dtype=float)
    y = np.array([0.20, -0.50, 0.54, 0.70], dtype=float)
    v = lambda x_arr: np.column_stack((np.ones(len(x_arr)), x_arr))

    A = regressao(x, y, v)
    print("Coeficientes A (a0 intercepto, a1 coef x):", A)
    # formatação similar ao enunciado: f(x) = a2 x + a1 ???
    # Observação: aqui A[0] = a1 (intercepto), A[1] = a2 (coef de x)
    print(f"f(x) = {A[1]:.2f}x + {A[0]:.2f}")

    img = plot_with_basis(x, y, v, A, "x", "y", "Atividade 1: Ajuste linear", "regressao_1.png")
    print("Gráfico salvo em:", img)
    print()


def atividade_2():
    print("-- Atividade 2 --")
    x = np.array([-1.94, -1.44, 0.93, 1.39], dtype=float)
    y = np.array([1.02, 0.59, -0.28, -1.04], dtype=float)
    v = lambda x_arr: np.column_stack((np.ones(len(x_arr)), x_arr))

    A = regressao(x, y, v)
    print("Coeficientes A:", A)
    print(f"f(x) = {A[1]:.8f}x + {A[0]:.8f}")
    f1 = A[0] + A[1] * 1.0
    print(f"f(1) = {f1:.7f}")

    img = plot_with_basis(x, y, v, A, "x", "y", "Atividade 2: Ajuste linear", "regressao_2.png")
    print("Gráfico salvo em:", img)
    print()


def atividade_3():
    print("-- Atividade 3 --")
    x = np.array([0.01, 1.02, 2.04, 2.95, 3.55], dtype=float)
    y = np.array([1.99, 4.55, 7.20, 9.51, 10.82], dtype=float)
    # base para parábola: 1, x, x^2  (a + b x + c x^2) mas enunciado pede y = a x^2 + b x + c
    # nós retornamos A = [c, b, a] se usarmos potência crescente (p=0 -> 1)
    v = lambda x_arr: np.column_stack((np.ones(len(x_arr)), x_arr, x_arr**2))

    A = regressao(x, y, v)
    # A[0] = c, A[1] = b, A[2] = a
    a = A[2]
    b = A[1]
    c = A[0]
    print(f"y = {a:.7f}x^2 + {b:.7f}x + {c:.7f}")

    img = plot_with_basis(x, y, v, A, "x", "y", "Atividade 3: Ajuste quadrático", "regressao_3.png")
    print("Gráfico salvo em:", img)
    print()


def atividade_4():
    print("-- Atividade 4 --")
    x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    y = np.array([31, 35, 37, 33, 28, 20, 16, 15, 18, 23, 31], dtype=float)

    # (a) base: [1, sin(2πx), cos(2πx)]
    v_a = lambda x_arr: np.column_stack((np.ones(len(x_arr)), np.sin(2 * np.pi * x_arr), np.cos(2 * np.pi * x_arr)))
    A_a = regressao(x, y, v_a)
    print("(a) Coeficientes a, b, c:", A_a)
    print(f"a = {A_a[0]:.6f}, b = {A_a[1]:.7f}, c = {A_a[2]:.7f}")

    img_a = plot_with_basis(x, y, v_a, A_a, "x", "y", "Atividade 4a: Seno/Cosseno", "regressao_4a.png")
    print("Gráfico (a) salvo em:", img_a)

    # (b) base: [1, x, x^2, x^3]
    v_b = lambda x_arr: np.column_stack((np.ones(len(x_arr)), x_arr, x_arr**2, x_arr**3))
    A_b = regressao(x, y, v_b)
    print("(b) Coeficientes a, b, c, d:", A_b)
    print(f"a = {A_b[0]:.6f}, b = {A_b[1]:.6f}, c = {A_b[2]:.6f}, d = {A_b[3]:.6f}")

    img_b = plot_with_basis(x, y, v_b, A_b, "x", "y", "Atividade 4b: Polinômio cúbico", "regressao_4b.png")
    print("Gráfico (b) salvo em:", img_b)
    print()

def main():
    atividade_1()
    atividade_2()
    atividade_3()
    atividade_4()

if __name__ == "__main__":
    main()