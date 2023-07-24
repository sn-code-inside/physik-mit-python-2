"""Simulation eines fliegenden Balls mit Luftreibung.

In diesem Programm wird die Bahnkurve für verschiedene
Abwurfwinkel bei sonst festen Parametern dargestellt. Die
sonstigen Parameter entsprechen der Simulation des
Tischtennisballs.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Masse des Körpers [kg].
m = 2.7e-3
# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.45 * math.pi * 20e-3 ** 2
# Abwurfwinkel [grad].
abwurfwinkel_grad = np.arange(0, 70, 10)
# Abwurfhöhe [m].
h = 1.1
# Betrag der Abwurfgeschwindigkeit [m/s].
betrag_v0 = 20
# Erdbeschleunigung [m/s²].
g = 9.81
# Luftdichte [kg/m³].
rho = 1.225


def bahnkurve(alpha, h0, v0, n=1000):
    """Berechne die Bahnkurve eines schiefen Wurfs mit Luftreibung.

    Args:
        alpha (float):
            Abwurfwinkel [rad].
        h0 (float):
            Abwurfhöhe [m].
        v0 (float):
            Anfangsgeschwindigkeit [m/s].
        n (int):
            Anzahl der Datenpunkte.

    Returns:
        tuple[np.ndarray, np.ndarray]:
           - Die Zeitpunkte der Bahnkurve (n).
           - Die zugehörigen Ortsvektoren (n × 2).
    """
    # Lege den Anfangsort und die Anfangsgeschwindigkeit fest.
    r0 = np.array([0, h0])
    v0 = np.array([v0 * math.cos(alpha), v0 * math.sin(alpha)])

    def F(v):
        """Berechne die Kraft als Funktion der Geschwindigkeit."""
        Fr = -0.5 * rho * cwA * np.linalg.norm(v) * v
        Fg = m * g * np.array([0, -1])
        return Fg + Fr

    def dgl(t, u):
        """Berechne die rechte Seite der Differentialgleichung."""
        r, v = np.split(u, 2)
        return np.concatenate([v, F(v) / m])

    def aufprall(t, u):
        """Ereignisfunktion: Detektiere den Aufprall."""
        r, v = np.split(u, 2)
        return r[1]

    # Beende die Simulation beim Auftreten des Ereignisses.
    aufprall.terminal = True

    # Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
    u0 = np.concatenate((r0, v0))

    # Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde
    result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                       events=aufprall,
                                       dense_output=True)

    # Berechne die Interpolation auf einem feinen Raster.
    t = np.linspace(0, np.max(result.t), n)
    r, v = np.split(result.sol(t), 2)

    return t, r


# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')
ax.grid()

# Plotte die Bahnkurve für verschiedene Winkel.
for winkel in abwurfwinkel_grad:
    t, r = bahnkurve(math.radians(winkel), h, betrag_v0)
    ax.plot(r[0], r[1], label=f'$\\alpha$ = {winkel}°')
ax.legend()

# Zeige die Grafik an.
plt.show()
