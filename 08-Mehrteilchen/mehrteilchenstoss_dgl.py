"""Simulation von Stößen über Differentialgleichungen."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Anzahl der Raumdimensionen und Anzahl der Teilchen.
n_dim = 2
n_teilchen = 10

# Simulationszeit und Zeitschrittweite [s].
t_max = 10
dt = 0.02

# Federkonstante beim Aufprall [N/m].
D = 5e3

# Positioniere die Massen zufällig im Bereich
# x=0,5 ... 1,5 und y = 0,5 ... 1,5 [m].
r0 = 0.5 + np.random.rand(n_teilchen, n_dim)

# Wähle zufällige Geschwindigkeiten im Bereich
# vx = -0,5 ... 0,5 und vy = -0,5 ... 0,5 [m/s].
v0 = -0.5 + np.random.rand(n_teilchen, n_dim)

# Wähle zufällige Radien im Bereich von 0,02 bis 0,04 [m].
radien = 0.02 + 0.02 * np.random.rand(n_teilchen)

# Wähle zufällige Massen im Bereich von 0,2 bis 2,0 [kg].
m = 0.2 + 1.8 * np.random.rand(n_teilchen)


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)
    r = r.reshape(n_teilchen, n_dim)
    a = np.zeros((n_teilchen, n_dim))
    for i in range(n_teilchen):
        for j in range(i):
            # Berechne den Abstand der Mittelpunkte.
            dr = np.linalg.norm(r[i] - r[j])
            # Berechne die Eindringtiefe.
            federweg = max(radien[i] + radien[j] - dr, 0)
            # Die Kraft soll proportional zum Federweg sein.
            F = D * federweg
            richtungsvektor = (r[i] - r[j]) / dr
            a[i] += F / m[i] * richtungsvektor
            a[j] -= F / m[j] * richtungsvektor
    return np.concatenate([v, a.reshape(-1)])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))

# Löse die Bewegungsgleichung bis zum Zeitpunkt t_max.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                   max_step=dt,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Wandle r und v in ein 3-dimensionales Array um:
#    1. Index - Teilchen
#    2. Index - Koordinatenrichtung
#    3. Index - Zeitpunkt
r = r.reshape(n_teilchen, n_dim, -1)
v = v.reshape(n_teilchen, n_dim, -1)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_aspect('equal')
ax.grid()

# Erzeuge für jedes Teilchen einen Kreis.
kreise = []
for radius in radien:
    kreis = mpl.patches.Circle([0, 0], radius, visible=False)
    ax.add_patch(kreis)
    kreise.append(kreis)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    for kreis, ort in zip(kreise, r):
        kreis.set_center(ort[:, n])
        kreis.set_visible(True)
    return kreise


# Erstelle die Animation und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
