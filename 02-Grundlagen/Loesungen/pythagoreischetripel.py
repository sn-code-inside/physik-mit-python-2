"""Suche spezielle pythagoreische Zahlentripel.

Das Programm sucht alle natürliche Zahlen a <= b <= c mit
    a² + b² = c²
und
    a + b + c = n
für eine fest vorgegebene Zahl n.
"""

# Lege die Zahl n fest.
n = 1000

# Durchlaufe alle infrage kommenden Zahlen für a. Da b und c
# größer oder gleich a sind, kommen nur die Zahlen infrage,
# die kleiner oder gleich n/3 sind.
for a in range(1, n // 3 + 1):
    # Durchlaufe für b nur die Zahlen bis zur Hälfte der
    # Differenz von n und a.
    for b in range(a, (n - a) // 2 + 1):
        # Die Zahl c ist nun eindeutig durch die Bedingung
        # a + b + c = n festgelegt.
        c = n - a - b
        # Überprüfe, ob es sich um ein pythagoreisches Tripel
        # handelt und gib in diesem Fall das Ergebnis aus.
        if a**2 + b**2 == c**2:
            print(f'{a}² + {b}² = {c}² und '
                  f'{a} + {b} + {c} = {a+b+c}')
