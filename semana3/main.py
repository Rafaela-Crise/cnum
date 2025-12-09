# semana3_exercicios.py
import math, sys
from decimal import Decimal, getcontext

def exp_series(x, atol=0.0):
    eps = sys.float_info.epsilon
    s = 1.0
    term = 1.0
    n = 0
    tol_abs = max(atol, eps)

    while True:
        n += 1
        term *= x / n
        s += term
        if abs(term) < eps * abs(s) or abs(term) < tol_abs:
            break
        if n > 100_000:
            break
    return s, n, term

# Atividade 1: 
def demo_atividade1():
    print("=== Atividade 1: série de Maclaurin para e^x ===")
    for val in [1.0, 5.0, -2.0]:
        approx, nterms, last = exp_series(val)
        print(f"x={val:+g} -> e^x ≈ {approx:.16g} (math.exp={math.exp(val):.16g}, termos={nterms})")
    print()

# Atividade 2: 
def exp_limit_approx(x, max_n=100_000, tol_rel=1e-14, tol_abs=0.0):
    prev = None
    last_diff = None
    for n in range(1, max_n+1):
        base = 1.0 + x / n
        if base <= 0.0:
            # Evita bases não-positivas que causam oscilações / valores complexos para potenciação inteira
            return None, n, False, "base_nonpositive"
        val = base**n
        if prev is not None:
            diff = abs(val - prev)
            last_diff = diff
            rel = diff / max(abs(val), sys.float_info.min)
            if diff < tol_abs or rel < tol_rel:
                return val, n, True, diff
        prev = val
    return prev, max_n, False, last_diff

def demo_atividade2():
    print("=== Atividade 2: (1 + x/n)^n com n crescente ===")
    for x in [1.0, -1.0, -5.0, 5.0]:
        approx, n_used, conv, info = exp_limit_approx(x, max_n=20000, tol_rel=1e-14)
        print(f"x={x:+g}: aprox={approx}, n_used={n_used}, converged={conv}, info={info}   (math.exp={math.exp(x) if abs(x)<800 else 'overflow'})")
    print()


# Atividade 3: 
def exp_series_scaling(x, theta=1.0):
    if x == 0.0:
        return 1.0, 0, 0
    k = max(0, math.ceil(math.log2(abs(x)/theta))) if abs(x) > theta else 0
    m = x / (2**k)

    em, n_terms, _ = exp_series(m)
    y = em
    for _ in range(k):
        # cada iteração eleva ao quadrado -> y <- y^2 ; no texto original "elevar ao quadrado k vezes"
        y *= y
    return y, k, n_terms

def demo_atividade3():
    print("=== Atividade 3: scaling para |x| grande ===")
    for val in [10.0, -20.0, 100.0]:
        y, k, n = exp_series_scaling(val, theta=1.0)
        true = math.exp(val) if abs(val)<800 else float('inf')
        print(f"x={val:+g} -> e^x ≈ {y:.6e} (math.exp={true})  [k={k}, termos série(m)={n}]")
    print()

#  Atividade 4: 
def cos_series(x, atol=0.0):
    """Aproxima cos(x) pela série com recursão de termos.
       Retorna (aprox, n_terms, last_term)."""
    eps = sys.float_info.epsilon
    tol_abs = max(atol, eps)
    t = 1.0  # t0
    s = t
    n = 0
    while True:
        n += 1
        # t_{n} = t_{n-1} * ( - x^2 / ((2n-1)(2n)) )
        t *= - (x*x) / ((2*n-1)*(2*n))
        s += t
        if abs(t) < eps * max(1.0, abs(s)) or abs(t) < tol_abs:
            break
        if n > 100_000:
            break
    return s, n, t

def demo_atividade4():
    print("=== Atividade 4: cos(x) série de Maclaurin (recursiva) ===")
    try:
        import numpy as np
        xs = np.linspace(-20, 20, 200)
    except Exception:
        # fallback sem numpy
        xs = [ -20 + i*(40/199) for i in range(200) ]
    max_rel_error = 0.0
    worst_x = None
    for x in xs:
        s, n, _ = cos_series(x)
        true = math.cos(x)
        denom = max(abs(true), sys.float_info.min)
        rel = abs((s - true) / denom)
        if rel > max_rel_error:
            max_rel_error = rel
            worst_x = x
    print()

# Atividade 5: 
def remainder_bound_min_n(x, tau=1e-12, max_n=100_000):
    term = 1.0
    n = 0
    while True:
        n += 1
        term *= abs(x) / n
        if term < tau:
            return n-1
        if n > max_n:
            return None

def demo_atividade5():
    print("=== Atividade 5: menor N tal que resto < tau ===")
    for x in [1, 3, 10]:
        N = remainder_bound_min_n(x, tau=1e-12)
        print(f"x={x}: N (número de termos necessários) ≈ {N}")
    print()

# Atividade 6: 
def decimal_exp(x, prec=80):
    getcontext().prec = prec
    dx = Decimal(x)
    try:
        res = dx.exp()
    except Exception:
        term = Decimal(1)
        s = Decimal(1)
        n = 0
        lim = Decimal(10) ** (-(prec-2))
        while True:
            n += 1
            term = term * dx / Decimal(n)
            s += term
            if abs(term) < lim or n > 100000:
                break
        res = s
    return res

def demo_atividade6():
    print("=== Atividade 6: precisão alta com Decimal ===")
    for x in [20, 40, 50]:
        try:
            f64 = math.exp(x)
        except OverflowError:
            f64 = float('inf')
        dec = decimal_exp(x, prec=100)
        try:
            dec_to_float = float(dec)
        except OverflowError:
            dec_to_float = float('inf')
        rel_err = None
        if math.isfinite(f64):
            try:
                rel_err = abs((Decimal(f64) - dec) / dec)
            except Exception:
                rel_err = None
        print(f"x={x}: math.exp (float64) = {f64}, Decimal (prec=100) = {dec}")
        if rel_err is not None:
            print(f"  relative error (float64 vs decimal) ≈ {rel_err:.3e}")
        else:
            print("  relative error: não disponível (float64 overflowed ou comparação não aplicável)")
    print()

if __name__ == "__main__":
    demo_atividade1()
    demo_atividade2()
    print(atividade2_explicacao)
    demo_atividade3()
    demo_atividade4()
    demo_atividade5()
    demo_atividade6()
