"""Berechnung des größten gemeinsamen Teilers nach Euklid."""

import math
import random


def ggt(a, b):
    """Berechne den größten gemeinsamen Teiler von a und b.

    Die Funktion berechnet den größten gemeinsamen Teiler
    mithilfe des klassischen Algorithmus von Euklid.
    """
    while a > 0:
        a, b = (max(a, b) - min(a, b), min(a, b))
    return b


# Überprüfe an einer gegebenen Anzahl zufällig gewählter
# natürlicher Zahlen im Bereich von 1 bis `maximale_zahl`, ob die
# Funktion ggt für das gleiche Ergebnis liefert, wie die Funktion
# `math.gcd`.
maximale_zahl = 10000
anzahl = 500
for i in range(anzahl):
    # Wähle zwei zufällige natürliche Zahlen.
    x = random.randint(1, maximale_zahl)
    y = random.randint(1, maximale_zahl)

    # Berechne den größten gemeinsamen Teiler mit beiden
    # Funktionen.
    teiler_euklid = ggt(x, y)
    teiler_math_ggt = math.gcd(x, y)

    # Gib das Ergebnis aus, wenn beide Berechnungen
    # übereinstimmen. Andernfalls gibt eine Fehlermeldung aus und
    # beende die Schleife.
    if teiler_euklid == teiler_math_ggt:
        print(f'ggt({x}, {y}) = {teiler_euklid}')
    else:
        print(f'Fehler ggt({x}, {y}) = {teiler_euklid} aber '
              f'math(gcd({x},{y}) = {teiler_math_ggt}')
        break
