"""Berechnung des prozentualen Fehlers der Näherung sin(x) = x.

In diesem Beispiel wird der prozentuale Fehler der Näherung
sin(x) = x im Bereich von 5 Grad bis 90 Grad in 5
Grad-Schritten bestimmt. Das Ergebnis wird in zwei Listen
gespeichert.
"""

import math

# Lege eine Liste der Winkel in Grad an.
liste_winkel = list(range(5, 95, 5))

# Erzeuge eine leere Liste für die relativen Fehler in %.
liste_fehler_prozentual = []

# Berechne für jeden Winkel den relativen Fehler.
for winkel in liste_winkel:
    x = math.radians(winkel)
    fehler = 100 * (x - math.sin(x)) / math.sin(x)
    liste_fehler_prozentual.append(fehler)

# Gib das Ergebnis aus.
print(liste_winkel)
print(liste_fehler_prozentual)
