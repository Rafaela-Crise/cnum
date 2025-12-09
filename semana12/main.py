import math
from algoritmos import medio, trapezio, simpson, integral, composite_medio, composite_trapezio, composite_simpson

def fmt(v):
    return f"{v:.8f}"

def atividade_1():
    print("-- Atividade 1: integrais em [0,1] --\n")
    a = 0.0
    b = 1.0
    h_step = 1e-3 

    f_a = lambda x: math.exp(-x)
    f_b = lambda x: x**2
    f_c = lambda x: x**3
    f_d = lambda x: x * math.exp(-x**2)
    f_e = lambda x: 1.0 / (x**2 + 1.0)
    f_f = lambda x: x / (x**2 + 1.0)

    functions = [
        ("a) exp(-x)", f_a, lambda: 1.0 - math.exp(-1.0)),
        ("b) x^2", f_b, lambda: 1.0/3.0),
        ("c) x^3", f_c, lambda: 1.0/4.0),
        ("d) x*exp(-x^2)", f_d, lambda: 0.5 * (1.0 - math.exp(-1.0))),
        ("e) 1/(x^2+1)", f_e, lambda: math.pi / 4.0),
        ("f) x/(x^2+1)", f_f, lambda: 0.5 * math.log(2.0)),
    ]

    print("{:<6} {:<14} {:<14} {:<14} {:<14}".format("f", "Ponto médio", "Trapézio", "Simpson", "Exata"))
    for name, func, exact_fun in functions:
        r_m = integral(medio, func, a, b, n=h_step)
        r_t = integral(trapezio, func, a, b, n=h_step)
        r_s = integral(simpson, func, a, b, n=h_step)
        exact = exact_fun()
        print("{:<6} {:<14} {:<14} {:<14} {:<14}".format(
            name,
            fmt(r_m),
            fmt(r_t),
            fmt(r_s),
            fmt(exact)
        ))
    print("\nObs: usei passo h = 1e-3 para integrais compostas (versão por passo).")

def atividade_2():
    print("\n-- Atividade 2: integral de 2 a 5 de exp(4 - x^2) --\n")
    a = 2.0
    b = 5.0
    f = lambda x: math.exp(4.0 - x**2)

    Ns = [3, 5, 7, 9]
    print("{:<6} {:<14} {:<14} {:<14}".format("N", "P. médio", "Trapézio", "Simpson"))
    for N in Ns:
        rm = composite_medio(f, a, b, N)
        rt = composite_trapezio(f, a, b, N)
        rs = composite_simpson(f, a, b, N)  # se N for ímpar, função ajusta para par internamente
        print("{:<6} {:<14} {:<14} {:<14}".format(
            N, fmt(rm), fmt(rt), fmt(rs)
        ))

if __name__ == "__main__":
    atividade_1()
    atividade_2()
