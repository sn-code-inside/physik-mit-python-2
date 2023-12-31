﻿"""Simulation eines ebenen Pendels."""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeit und Zeitschrittweite [s].
t_max = 20
dt = 0.02
# Masse des Körpers [kg].
m = 1.0
# Länge des Pendels [m].
L = 0.7
# Anfangsauslenkung [rad].
phi0 = math.radians(20.0)
# Erdbeschleunigung [m/s²].
g = 9.81
# Anfangsort [m] und Anfangsgeschwindigkeit [m/s].
r0 = L * np.array([math.sin(phi0), -math.cos(phi0)])
v0 = np.array([0, 0])

# Vektor der Gewichtskraft.
F_g = m * g * np.array([0, -1])


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)

    # Berechne die Zwangskraft.
    F_zwang = - (m * v @ v + F_g @ r) * r / (r @ r)

    # Berechne den Vektor der Beschleunigung.
    a = (F_zwang + F_g) / m

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1.2 * L * math.sin(phi0), 1.2 * L * math.sin(phi0))
ax.set_ylim(-0.95, 0.05)
ax.set_aspect('equal')
ax.grid()

# Erzeuge eine Punktplot für die Position des Pendelkörpers
# und je einen Linienplot für die Pendelstange und die Bahnkurve.
plot_koerper, = ax.plot([], [], 'o', color='red',
                        markersize=10, zorder=5)
plot_stange, = ax.plot([], [], '-', color='black')
plot_bahn, = ax.plot([], [], '-b')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Bahnkurve.
    plot_bahn.set_data(r[0, :n + 1], r[1, :n + 1])

    # Aktualisiere die Position des Pendelkörpers.
    plot_koerper.set_data(r[:, n].reshape(-1, 1))

    # Aktualisiere die Pendelstange.
    plot_stange.set_data([0, r[0, n]], [0, r[1, n]])

    return plot_stange, plot_koerper, plot_bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
