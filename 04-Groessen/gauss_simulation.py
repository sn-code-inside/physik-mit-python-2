"""Simulation der Gauß-Verteilung von Messwerten."""

import math
import numpy as np
import matplotlib.pyplot as plt

# Anzahl der Messwerte.
n_messwerte = 50000
# Anzahl der Störgrößen.
n_stoergroessen = 20

# Erzeuge die simulierten Messwerte.
messwerte = np.random.rand(n_messwerte, n_stoergroessen)
messwerte = np.sum(messwerte, axis=1)

# Bestimme den Mittelwert und die Standardabweichung.
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


# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Messwert')
ax.set_ylabel('Wahrscheinlichkeitsdichte')
ax.grid()

# Erzeuge ein Histogramm der simulierten Messwerte.
hist_werte, hist_kanten, patches = ax.hist(messwerte, bins=51,
                                           density=True)

# Werte die Gauß-Verteilung an den Rändern der einzelnen
# Histogrammbalken aus und plotte sie.
ax.plot(hist_kanten, gauss(hist_kanten, standardabw, mittelwert))

# Zeige die Grafik an.
plt.show()
