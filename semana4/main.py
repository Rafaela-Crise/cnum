import math
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
import os, sys

# Atividade 1 
def f1(x):
    return x**3 - x - 2

def atividade1():
    # Plota para ajudar a visualizar (salva imagem)
    x_vals = np.arange(-2, 3, 0.01)
    y_vals = f1(x_vals)
    plt.figure(figsize=(6,4))
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.plot(x_vals, y_vals, label='f(x)=x^3-x-2')
    plt.grid(True)
    plt.title("Atividade 1: visualização f(x)=x^3-x-2")
    plt.legend()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/bissecao_atividade1.png", dpi=150, bbox_inches="tight")
    plt.close()

    raiz, its = bissecao(f1, 1.0, 2.0, TOL=1e-15, max_iter=100)
    return raiz, its

# Atividade 2 
def atividade2():
    # resolver x = cos(x) -> f(x) = x - cos(x)
    f = lambda x: x - math.cos(x)
    TOL = 1e-4
    max_it = 4
    raiz, its = bissecao(f, 0.0, 1.0, TOL=TOL, max_iter=max_it)
    # arredonda conforme solicitado (resultado aproximado)
    return raiz, its

# Atividade 3 
def atividade3():
    # f(x) = 5*sin(x^2) - exp(x/10)
    f = lambda x: 5.0*math.sin(x**2) - math.exp(x/10.0)
    intervals = []
    a = 0.0
    bmax = 10.0
    dx = 0.1
    x = a
    while x < bmax:
        fa = f(x)
        fb = f(x+dx)
        if fa == 0.0:
            intervals.append((x, x))
        elif fa * fb < 0:
            intervals.append((x, x+dx))
        x += dx
        if len(intervals) >= 10:
            break

    # pegar as três primeiras raízes positivas
    roots = []
    for (a,b) in intervals:
        if len(roots) >= 3:
            break
        try:
            r, it = bissecao(f, a, b, TOL=1e-5, max_iter=200)
            if r > 0:
                roots.append((r, it, (a,b)))
        except ValueError:
            pass

    return roots

# Atividade 4 
def atividade4():
    # dados físicos
    k = 1.380649e-23
    q = 1.602176634e-19
    T = 300.0
    vt = k * T / q
    Ir = 1e-12

    def solve_case(V, R):
        f = lambda vd: R*Ir*(math.exp(vd/vt)-1.0) + vd - V
        a = -100.0
        b = 100.0
        fa = f(a)
        fb = f(b)
        if fa*fb > 0:
            # tenta aumentar o intervalo
            for factor in [200,500,1000]:
                a = -factor
                b = factor
                fa = f(a)
                fb = f(b)
                if fa*fb <= 0:
                    break
        vd, its = bissecao(f, a, b, TOL=1e-6, max_iter=200)
        def sig(x, p=3):
            if x == 0:
                return 0.0
            else:
                return float(f"{x:.{p}g}")
        return sig(vd, 3), its, vd

    cases = [
        (30.0, 1e3),
        (3.0, 1e3),
        (3.0, 10e3),
        (0.300, 1e3),
        (-0.300, 1e3),
        (-30.0, 1e3),
        (-30.0, 10e3),
    ]
    results = []
    for V,R in cases:
        try:
            aprox3, its, vd_full = solve_case(V,R)
        except Exception as e:
            aprox3, its, vd_full = None, None, None
        results.append(((V,R), aprox3, its, vd_full))
    return results, vt

# Atividade 5 
def atividade5():
    d = 500.0
    fmax = 50.0
    f = lambda C: C*(math.cosh(d/(2.0*C)) - 1.0) - fmax
    a = 1e-6
    b = 1e5
    fa = f(a)
    fb = f(b)
    if fa*fb > 0:
        raise RuntimeError("Não encontrou mudança de sinal para a equação da catenária.")
    C, its = bissecao(f, a, b, TOL=1e-9, max_iter=200)
    return C, its

# Atividade 6 
def atividade6():
    # f = 1 kHz, L=100 mH, R=1kΩ
    f0 = 1e3
    L = 100e-3
    R = 1e3
    omega = 2*math.pi*f0
    tan_phi = omega * L / R
    phi = math.atan(tan_phi)
    # função Id(β) = sin(β - φ) + sin(φ)*exp(-β * tan(φ))  (β em radianos)
    def Id(beta):
        return math.sin(beta - phi) + math.sin(phi) * math.exp(-beta * tan_phi)
    a = math.pi
    b = 2*math.pi
    # garantir mudança de sinal, se não expandir
    fa = Id(a)
    fb = Id(b)
    if fa*fb > 0:
        a = math.pi/2
        for _ in range(6):
            fa = Id(a)
            fb = Id(b)
            if fa*fb <= 0:
                break
            a -= 0.5
    beta, its = bissecao(Id, a, b, TOL=1e-9, max_iter=200)
    beta_deg = math.degrees(beta)
    return beta_deg, its, phi, tan_phi

def main():
    os.makedirs("outputs", exist_ok=True)
    print("--- Executando todas as atividades (1..6) ---\n")
    # Atividade 1
    r1, i1 = atividade1()
    print(f"Atividade 1: raiz f(x)=x^3-x-2 -> {r1:.12g} (iterações={i1})")
    print("  Gráfico salvo em outputs/bissecao_atividade1.png\n")

    # Atividade 2
    r2, i2 = atividade2()
    print(f"Atividade 2: solução x = cos(x) aproximada -> {r2:.6g} (iterações={i2})\n")

    # Atividade 3
    roots = atividade3()
    print("Atividade 3: primeiras 3 raízes positivas encontradas e aproximadas com bissecção:")
    for idx, item in enumerate(roots):
        r, it, inter = item
        print(f"  zero {idx+1}: x ≈ {r:.6g}, intervalo inicial={inter}, it={it}")
    print("")

    # Atividade 4
    res4, vt = atividade4()
    print("Atividade 4: diodo (valores com ~3 algarismos significativos): vt= {:.3e} V".format(vt))
    for case in res4:
        (V,R), aprox3, its, vd_full = case
        print(f"  V={V} V, R={R} Ω -> vd (3 sig figs) ≈ {aprox3}, it={its}, vd_full={vd_full:.6g}")
    print("")

    # Atividade 5
    C, its5 = atividade5()
    print(f"Atividade 5: comprimento do cabo (C) ≈ {C:.6g} m (iterações={its5})")
    print("")

    # Atividade 6
    beta_deg, its6, phi, tan_phi = atividade6()
    print(f"Atividade 6: beta ≈ {beta_deg:.6f} deg (iterações={its6}), phi={math.degrees(phi):.6f} deg, tan(phi)={tan_phi:.6g}")

if __name__ == '__main__':
    main()
