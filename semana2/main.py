import numpy as np
import matplotlib.pyplot as plt

#  ATIVIDADE 1 
def is_perfect(n: int) -> bool:
    if n < 1:
        return False

    sum_divisors = 0
    for i in range(1, n):
        if n % i == 0:
            sum_divisors += i

    return sum_divisors == n

#  ATIVIDADE 2 
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("O número deve ser não negativo.")

    result = 1
    i = n
    while i > 1:
        result *= i
        i -= 1

    return result

#  ATIVIDADE 3 
def is_prime(n: int) -> bool:
    if n <= 1:
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True

#  ATIVIDADE 4  
def sum_of_digits(n: int) -> int:
    if n < 0:
        raise ValueError("O número deve ser não negativo.")

    total = 0
    for digit in str(n):
        total += int(digit)

    return total

#  ATIVIDADE 5  
def matrix_product():
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    B = np.array([
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ])

    C = A @ B

    print(C)
    print(C.shape)
    print(C.size)
    print(len(C))

#   ATIVIDADE 6  
def plot(n: int) -> None:
    x = np.linspace(-np.pi, np.pi, n)
    y_sen = np.sin(x)
    y_cos = np.cos(x)

    plt.plot(x, y_sen, label="seno")
    plt.plot(x, y_cos, label="cosseno")

    plt.xlim(-np.pi, np.pi)
    plt.xlabel("Ângulo [rad]")
    plt.ylabel("Função trigonométrica(x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("plot.png")

    print(f'x =\n{x}')
    print(f'y_sen =\n{y_sen}')
    print(f'y_cos =\n{y_cos}')
    
def main():
    # Atividade 1
    assert is_perfect(6) is True
    assert is_perfect(7) is False
    assert is_perfect(-1) is False

    # Atividade 2
    assert factorial(5) == 120
    assert factorial(0) == 1
    try:
        factorial(-1)
    except ValueError as error:
        assert str(error) == "O número deve ser não negativo."

    # Atividade 3
    assert is_prime(7) is True
    assert is_prime(10) is False

    # Atividade 4
    assert sum_of_digits(123) == 6
    try:
        sum_of_digits(-1)
    except ValueError as error:
        assert str(error) == "O número deve ser não negativo."

    # Atividade 5
    matrix_product()

    # Atividade 6
    plot(35)


if __name__ == "__main__":
    main()
