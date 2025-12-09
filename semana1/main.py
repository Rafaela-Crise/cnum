import math

def calcular_raizes(a, b, c):
    # Equação do 1º grau se a = 0
    if a == 0:
        if b == 0:
            raise ValueError("Não é possível resolver: b não pode ser zero.")
        return (-c / b,)

    # Se b = 0, gerar exceção
    if b == 0:
        raise ValueError("Coeficiente b não pode ser zero.")

    # Fórmula de Bhaskara
    delta = b**2 - 4*a*c

    if delta < 0:
        raise ValueError("A equação não possui raízes reais.")

    x1 = (-b + math.sqrt(delta)) / (2*a)
    x2 = (-b - math.sqrt(delta)) / (2*a)

    return x1, x2


# Exemplo de teste
print(calcular_raizes(1, -3, 2))  # Deve retornar (2, 1)



