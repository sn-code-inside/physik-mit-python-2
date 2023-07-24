"""Berechnung von Fehlergrößen mithilfe von NumPy."""

import math
import numpy as np

# Gemessene Schwingungsdauern [s].
messwerte = np.array([2.05, 1.99, 2.06, 1.97, 2.01,
                      2.00, 2.03, 1.97, 2.02, 1.96])

# Berechne die drei gesuchten Kenngrößen.
mittelwert = np.mean(messwerte)
standardabw = np.std(messwerte)
fehler = standardabw / math.sqrt(messwerte.size)

print(f'Mittelwert:            <T> = {mittelwert:.2f} s')
print(f'Standardabweichung:  sigma = {standardabw:.2f} s')
print(f'Mittlerer Fehler:  Delta T = {fehler:.2f} s')
