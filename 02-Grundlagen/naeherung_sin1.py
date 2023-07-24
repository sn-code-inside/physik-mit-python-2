"""Berechnung des prozentualen Fehlers der Näherung sin(x) = x.

In diesem Beispiel wird der prozentuale Fehler der Näherung
sin(x) = x im Bereich von 5 Grad bis 90 Grad in 5
Grad-Schritten bestimmt. Das Ergebnis wird als Text
ausgegeben.
"""

import math

for winkel in range(5, 95, 5):
    x = math.radians(winkel)
    fehler = 100 * (x - math.sin(x)) / math.sin(x)
    print(f'Winkel: {winkel:2} Grad, Fehler: {fehler:4.1f} %')
