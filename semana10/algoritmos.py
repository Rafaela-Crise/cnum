import numpy as np

def regressao(x, y, v):
    #Regressão por mínimos quadrados.
    V = v(x)
    Vt = V.T
    try:
        A = np.linalg.inv(Vt @ V) @ (Vt @ y)
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(V) @ y
    return A


