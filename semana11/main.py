import numpy as np
from algoritmos import dp, dr, dc

# valores de h
h1 = 1e-2
h2 = 1e-3

# funções a serem derivadas
f1 = lambda x: np.sin(x)
df1 = lambda x: np.cos(x)  # derivada exata

f2 = lambda x: np.exp(-x)
df2 = lambda x: -np.exp(-x)  # derivada exata

x1 = 2
x2 = 1

def testar(f, df, x, nome):
    print(f"\n===== {nome} =====")
    exata = df(x)
    print(f"Derivada exata: {exata}")

    for h in [h1, h2]:
        print(f"\n--- h = {h} ---")
        print("Progressiva :", dp(f, x, h))
        print("Regressiva  :", dr(f, x, h))
        print("Central     :", dc(f, x, h))

if __name__ == "__main__":
    testar(f1, df1, x1, "f(x)=sin(x) em x=2")
    testar(f2, df2, x2, "f(x)=exp(-x) em x=1")
