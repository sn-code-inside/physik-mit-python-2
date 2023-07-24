"""Berechnung von Fehlergrößen.

Das Programm berechnet den Mittelwert, die Standardabweichung
und den Fehler des Mittelwertes, indem die entsprechenden
Definitionen mithilfe von for-Schleifen implementiert werden.

Das Programm wurde so modifiziert, dass eine größere Anzahl
von Dezimalstellen ausgegeben wird.
"""

import math
import numpy as np

# Gemessene Schwingungsdauern [s].
messwerte = np.array([2.05, 1.99, 2.06, 1.97, 2.01,
                      2.00, 2.03, 1.97, 2.02, 1.96])

# Anzahl der Messwerte.
n = messwerte.size

# Berechne den Mittelwert.
mittelwert = 0
for x in messwerte:
    mittelwert += x
mittelwert /= n

# Berechne die Standardabweichung.
standardabw = 0
for x in messwerte:
    standardabw += (x - mittelwert) ** 2
standardabw = math.sqrt(standardabw / (n - 1))

# Berechne den mittleren Fehler des Mittelwertes.
fehler = standardabw / math.sqrt(n)

print(f'Mittelwert:            <T> = {mittelwert:.6f} s')
print(f'Standardabweichung:  sigma = {standardabw:.6f} s')
print(f'Mittlerer Fehler:  Delta T = {fehler:.6f} s')
