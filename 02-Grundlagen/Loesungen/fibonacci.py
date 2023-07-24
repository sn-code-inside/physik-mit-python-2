"""Berechnung der Fibonacci-Folge."""


def fibonacci(n):
    """Berechne ersten n Elemente der Fibonacci-Folge.

    Args:
        n (int): Anzahl der Elemente.

    Returns:
        list: Liste der Elemente der Folge.
    """
    # Setze die ersten beiden Folgenglieder auf 1.
    fib = [1, 1]

    # Führe die iterationsvorschrift der Folge durch und gib
    # das Ergebnis zurück.
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib


# Gib die ersten 20 Glieder der Fibonacci-Folge aus.
print(fibonacci(20))
