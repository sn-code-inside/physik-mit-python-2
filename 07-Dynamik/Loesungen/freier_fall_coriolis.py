"""Ablenkung einer fallenden Masse durch die Corioliskraft.

Das Programm simuliert einen freien Fall mit Berücksichtigung
der Corioliskraft.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

# Lege den Anfangsort fest [m].
r0 = np.array([0, 0, 100.0])
# Lege den Vektor der Anfangsgeschwindigkeit fest [m/s].
v0 = np.array([0.0, 0.0, 0.0])
# Breitengrad [rad].
theta = math.radians(49.4)
# Betrag der Winkelgeschwindigkeit der Erde [rad/s].
betrag_omega = 7.292e-5
# Erdbeschleunigung [m/s²].
g = 9.81

# Vektor der Winkelgeschwindigkeit [rad/s].
omega = betrag_omega * np.array(
    [0, math.cos(theta), math.sin(theta)])


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)

    # Schwerebeschleunigung.
    a_g = g * np.array([0, 0, -1])

    # Coriolisbeschleunigung.
    a_c = - 2 * np.cross(omega, v)

    # Summe aus Schwere- und Coriolisbeschleunigung.
    a = a_g + a_c
    return np.concatenate([v, a])


def aufprall(t, u):
    """Ereignisfunktion: Detektiere das Erreichen des Erdbodens."""
    r, v = np.split(u, 2)
    return r[2]


# Beende die Simulation beim Auftreten des Ereignisses.
aufprall.terminal = True

# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde.
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                   events=aufprall,
                                   dense_output=True)

# Berechne die Interpolation auf einem feinen Raster.
t = np.linspace(0, np.max(result.t), 1000)
r, v = np.split(result.sol(t), 2)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(9, 3))
fig.set_tight_layout(True)

# Plotte den Zeitverlauf der x-Koordinate.
ax_x = fig.add_subplot(1, 3, 1)
ax_x.set_xlabel('$t$ [s]')
ax_x.set_ylabel('$x$ [mm]')
ax_x.grid()
ax_x.plot(t, 1e3 * r[0], '-b')

# Plotte den Zeitverlauf der y-Koordinate.
ax_y = fig.add_subplot(1, 3, 2)
ax_y.set_xlabel('$t$ [s]')
ax_y.set_ylabel('$y$ [µm]')
ax_y.grid()
ax_y.plot(t, 1e6 * r[1], '-b')

# Plotte den Zeitverlauf der z-Koordinate.
ax_z = fig.add_subplot(1, 3, 3)
ax_z.set_xlabel('$t$ [s]')
ax_z.set_ylabel('$z$ [m]')
ax_z.grid()
ax_z.plot(t, r[2], '-b')

# Zeige die Grafik an.
plt.show()
