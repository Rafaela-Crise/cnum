import math
from typing import Callable

def medio(f: Callable[[float], float], a: float, b: float) -> float:
    """Regra do ponto médio em [a,b]."""
    h = b - a
    return h * f((a + b) / 2.0)

def trapezio(f: Callable[[float], float], a: float, b: float) -> float:
    """Regra do trapézio em [a,b]."""
    h = b - a
    return h * (0.5 * f(a) + 0.5 * f(b))

def simpson(f: Callable[[float], float], a: float, b: float) -> float:
    """Regra de Simpson em [a,b] (uma aplicação com 2 subintervalos)."""
    h = (b - a) / 2.0
    # fórmula: (h/3) * [f(a) + 4 f(mid) + f(b)], mas aqui h é (b-a)/2, então coeficiente final é h*(1/3) etc.
    return h * ((1.0 / 3.0) * f(a) + (4.0 / 3.0) * f((a + b) / 2.0) + (1.0 / 3.0) * f(b))

def composite_medio(f: Callable[[float], float], a: float, b: float, N: int) -> float:
    if N < 1:
        raise ValueError("N deve ser >= 1")
    h = (b - a) / N
    s = 0.0
    for i in range(N):
        ai = a + i * h
        bi = ai + h
        s += medio(f, ai, bi)
    return s

def composite_trapezio(f: Callable[[float], float], a: float, b: float, N: int) -> float:
    if N < 1:
        raise ValueError("N deve ser >= 1")
    h = (b - a) / N
    s = 0.5 * f(a) + 0.5 * f(b)
    for i in range(1, N):
        x = a + i * h
        s += f(x)
    return h * s

def composite_simpson(f: Callable[[float], float], a: float, b: float, N: int) -> float:
    if N < 1:
        raise ValueError("N deve ser >= 1")
    if N % 2 == 1:
        # ajusta para o próximo par
        N = N + 1
    h = (b - a) / N
    s = f(a) + f(b)
    # soma pares e ímpares com os fatores 4 e 2
    for i in range(1, N):
        x = a + i * h
        coef = 4 if (i % 2 == 1) else 2
        s += coef * f(x)
    return (h / 3.0) * s

def integral(metodo, f: Callable[[float], float], a: float, b: float, n=1e-3):
    # caso n seja inteiro (número de subintervalos)
    if isinstance(n, int):
        N = n
        # escolhe a função composta correspondente
        if metodo is medio:
            return composite_medio(f, a, b, N)
        elif metodo is trapezio:
            return composite_trapezio(f, a, b, N)
        elif metodo is simpson:
            return composite_simpson(f, a, b, N)
        else:
            # método genérico: aplicar metodo em N subintervalos
            h = (b - a) / N
            s = 0.0
            for i in range(N):
                ai = a + i * h
                bi = ai + h
                s += metodo(f, ai, bi)
            return s
    else:
        # n é interpretado como passo h (float)
        h = float(n)
        if h <= 0:
            raise ValueError("Passo h deve ser positivo")
        s = 0.0
        c = a
        d = a + h
        # percorre [a,b] em blocos de tamanho h; último bloco usa o que restar
        while d < b - 1e-15:
            s += metodo(f, c, d)
            c = d
            d = c + h
        # último subintervalo [c, b]
        if c < b:
            s += metodo(f, c, b)
        return s
