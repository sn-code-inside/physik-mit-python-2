"""Berechne die Schaltjahre im gregorianischen Kalender."""


def ist_schaltjahr(n):
    """Gib zurück, ob das Jahr n ein Schaltjahr ist.

    Args:
        n (int): Jahreszahl
    """
    return ((n % 4 == 0) and (n % 100 != 0)) or (n % 400 == 0)


# Erzeuge eine Liste der Schaltjahre zwischen 1900 und 2200.
schaltjahre = []
for jahr in range(1900, 2201):
    if ist_schaltjahr(jahr):
        schaltjahre.append(jahr)

# Gib die Liste der Schaltjahre aus.
print(schaltjahre)
