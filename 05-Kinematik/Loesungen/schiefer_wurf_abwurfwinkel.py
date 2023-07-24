"""Schiefer Wurf für unterschiedliche Abwurfwinkel."""

import math
import numpy as np
import matplotlib.pyplot as plt

# Anfangshöhe [m].
h = 10.0
# Anfangsgeschwindigkeit [m/s].
betrag_v0 = 5.0
# Schwerebeschleunigung [m/s²].
g = 9.81


def wurf(alpha, v0, h0, n=500):
    """Berechne die Bahnkurve eines schiefen Wurfs.

    Args:
        alpha (float): Abwurfwinkel [rad].
        v0 (float): Abwurfgeschwindigkeit [m/s].
        h0 (float): Anfangshöhe [m].
        n (int): Anzahl der Datenpunkte.

    Returns:
        np.ndarray: Bahnkurve (n × 2).
    """
    r0 = np.array([0, h0])
    v0 = np.array([v0 * math.cos(alpha),
                   v0 * math.sin(alpha)])
    a = np.array([0, -g])

    # Berechne den Zeitpunkt, zu dem der Gegenstand den Boden
    # erreicht.
    t_ende = v0[1] / g + math.sqrt((v0[1] / g)**2 + 2 * r0[1] / g)

    # Erstelle ein n × 1 - Array mit Zeitpunkten.
    t = np.linspace(0, t_ende, n)
    t = t.reshape(-1, 1)

    # Berechne den Ortsvektor für alle Zeitpunkte im Array t.
    return r0 + v0 * t + 0.5 * a * t**2


# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.grid()

# Plotte die Bahnkurve für verschiedene Abwurfwinkel.
for winkel in range(0, 70, 10):
    r = wurf(math.radians(winkel), betrag_v0, h)
    ax.plot(r[:, 0], r[:, 1], label=f'$\\alpha$ = {winkel}°')

# Erzeuge die Legende und zeige die Grafik an.
ax.legend()
plt.show()
