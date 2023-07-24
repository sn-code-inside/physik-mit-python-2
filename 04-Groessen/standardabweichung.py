"""Berechnung von Fehlergrößen mithilfe von for-Schleifen."""

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

print(f'Mittelwert:            <T> = {mittelwert:.2f} s')
print(f'Standardabweichung:  sigma = {standardabw:.2f} s')
print(f'Mittlerer Fehler:  Delta T = {fehler:.2f} s')
