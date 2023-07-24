"""Berechnung des prozentualen Fehlers der Näherung sin(x) = x.

In diesem Beispiel wird der prozentuale Fehler der Näherung
sin(x) = x im Bereich von 5 Grad bis 90 Grad in 5
Grad-Schritten bestimmt. Das Ergebnis wird in NumPy-Arrays
gespeichert.
"""

import numpy as np

# Lege ein Array der Winkel in Grad an.
winkel = np.arange(5, 95, 5)

# Wandle die Winkel in das Bogenmaß um.
x = np.radians(winkel)

# Berechne die relativen Fehler.
fehler = 100 * (x - np.sin(x)) / np.sin(x)

# Gib das Ergebnis aus.
print(winkel)
print(fehler)
