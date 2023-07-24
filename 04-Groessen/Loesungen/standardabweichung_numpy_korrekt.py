"""Berechnung von Fehlergrößen mithilfe von NumPy.

Das Programm berechnet den Mittelwert, die Standardabweichung
und den Fehler des Mittelwertes mithilfe der entsprechenden
NumPy-Funktionen für eine vorgegebene Reihe von Messdaten.

Das Programm wurde so modifiziert, dass eine größere Anzahl
von Dezimalstellen ausgegeben wird. Mithilfe des zusätzlichen
Arguments ddof=1 wird das korrekte Ergebnis der
Standardabweichung berechnet.
"""

import math
import numpy as np

# Gemessene Schwingungsdauern [s].
messwerte = np.array([2.05, 1.99, 2.06, 1.97, 2.01,
                      2.00, 2.03, 1.97, 2.02, 1.96])

# Berechne die drei gesuchten Kenngrößen.
mittelwert = np.mean(messwerte)
standardabw = np.std(messwerte, ddof=1)
fehler = standardabw / math.sqrt(messwerte.size)

print(f'Mittelwert:            <T> = {mittelwert:.6f} s')
print(f'Standardabweichung:  sigma = {standardabw:.6f} s')
print(f'Mittlerer Fehler:  Delta T = {fehler:.6f} s')
