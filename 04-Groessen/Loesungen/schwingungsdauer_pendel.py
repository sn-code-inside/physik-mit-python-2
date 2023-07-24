"""Wahrscheinlichkeit von Messwerten in einem Intervall.

Die Wahrscheinlichkeit, dass die Messwerte der
Schwingungsdauer eines Pendels innerhalb eines bestimmten
Intervalls T_min bis T_max liegen, wird berechnet.
"""

import math
import numpy as np
import scipy.integrate

# Gemessene Schwingungsdauern [s].
messwerte = np.array([2.05, 1.99, 2.06, 1.97, 2.01,
                      2.00, 2.03, 1.97, 2.02, 1.96])

# Vorgegebene Grenzen des Intervalls [s].
untere_grenze = 1.95
obere_grenze = 2.05

# Berechnung den Mittelwert und die Standardabweichung.
mittelwert = np.mean(messwerte)
standardabw = np.std(messwerte, ddof=1)


def gauss(x, sigma, x0=0):
    """Berechne die normierte Gauß-Verteilung.

    Args:
        x (np.ndarray):
            Werte, für die die Funktion berechnet wird.
        sigma (float):
            Standardabweichung.
        x0 (float):
            Mittelwert.

    Returns:
        np.ndarray: Array der Funktionswerte.
    """
    a = 1 / (math.sqrt(2 * math.pi) * sigma)
    return a * np.exp(- (x - x0) ** 2 / (2 * sigma ** 2))


# Integriere die Gauß-Verteilung über das gegebene Intervall.
integral, fehler = scipy.integrate.quad(gauss,
                                        untere_grenze, obere_grenze,
                                        args=(standardabw,
                                              mittelwert))

# Gib das Ergebnis aus.
print(f'Im Intervall von {untere_grenze:.2f} s '
      f'bis {obere_grenze:.2f} s liegen '
      f'{100 * integral:.1f}% der Messwerte.')
