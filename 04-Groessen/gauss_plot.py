"""Grafische Darstellung der Gauß-Verteilung."""

import math
import numpy as np
import matplotlib.pyplot as plt


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


# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x - <x>')
ax.set_ylabel('f(x)')
ax.grid()

# Plotte die Gauß-Verteilung für verschiedene Werte von sigma.
x_plot = np.linspace(-3, 3, 1000)
for standardabw in [0.2, 0.5, 1.0, 2.0]:
    ax.plot(x_plot, gauss(x_plot, standardabw),
            label=f'$\\sigma$ = {standardabw}')

# Erzeuge die Legende und zeige die Grafik an.
ax.legend()
plt.show()
