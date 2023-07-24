"""Generator für Quadratzahlen."""


def quadratzahlen(n):
    """Erzeuge alle Quadratzahlen kleiner n."""
    i = 1
    while i**2 < n:
        yield i**2
        i += 1


# Gib alle Quadratzahlen kleiner als 100 aus.
for k in quadratzahlen(100):
    print(k)
