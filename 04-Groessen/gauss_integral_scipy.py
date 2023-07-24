"""Numerische Integration der Gauß-Verteilung mit SciPy."""

import math
import numpy as np
import scipy.integrate

# Standardabweichung.
standardabw = 0.5
# Integrationsbereich von -x_max bis +x_max.
x_max = 3


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


integral, fehler = scipy.integrate.quad(gauss, -x_max, x_max,
                                        args=(standardabw,))

print(f'Ergebnis der Integration: {integral}')
print(f'Fehler der Integration:   {fehler}')
