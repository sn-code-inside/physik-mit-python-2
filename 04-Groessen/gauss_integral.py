"""Numerische Integration der Gauß-Verteilung."""

import math
import numpy as np

# Standardabweichung.
standardabw = 0.5
# Integrationsbereich von -x_max bis +x_max.
x_max = 3
# Schrittweite für die Integration.
dx = 0.01


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
        np.ndarray: Berechnete Funktionswerte.
    """
    a = 1 / (math.sqrt(2 * math.pi) * sigma)
    return a * np.exp(- (x - x0) ** 2 / (2 * sigma ** 2))


# Führe die numerische Integration aus.
xi = -x_max
integral = 0
while xi < x_max:
    integral += gauss(xi + dx / 2, standardabw) * dx
    xi += dx

print(f'Ergebnis der Integration: {integral}')
