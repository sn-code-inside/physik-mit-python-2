"""Simulation der brownschen Bewegung."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
from stossprozess import Mehrteilchenstoss

# Anzahl der Teilchen.
n_teilchen = 150

# Simulationszeit und Zeitschrittweite [s].
t_max = 200
dt = 0.025

# Für jede Wand wird der Abstand vom Koordinatenursprung
# wand_d und ein nach außen zeigender Normalenvektor angegeben.
wandabstaende = np.array([2.0, 2.0, 2.0, 2.0])
wandnormalen = np.array([[0, -1.0], [0, 1.0], [-1.0, 0], [1.0, 0]])

# Anzahl der Raumdimensionen.
n_dim = wandnormalen.shape[1]

# Positioniere die Massen zufällig auf einem Kreisring mit
# Innenradius 0,5 m und Außenradius 1,9 m.
r = np.empty((n_teilchen, n_dim))
rho = 0.5 + 1.4 * np.random.rand(n_teilchen)
phi = 2 * np.pi * np.random.rand(n_teilchen)
r[:, 0] = rho * np.cos(phi)
r[:, 1] = rho * np.sin(phi)

# Wähle zufällige Geschwindigkeiten mit Komponenten zwischen -1
# und +1 [m/s].
v = 2 * (-0.5 + np.random.rand(n_teilchen, n_dim))

# Erzeuge ein Objekt der Klasse `Mehrteilchenstoß`.
stoss = Mehrteilchenstoss(r, v, massen=0.1, radien=0.02,
                          waende=(wandabstaende, wandnormalen))

# Das Teilchen mit dem Index 0 stellt den Pollenkörper dar.
stoss.r[0] = (0, 0)
stoss.v[0] = (0, 0)
stoss.radien[0] = 0.3
stoss.massen[0] = 1.0

# Führe die Simulation durch, indem die Methode `zeitschritt`
# mehrfach aufgerufen wird.
t = np.arange(0, t_max, dt)
r = []
for i in range(t.size):
    stoss.zeitschritt(dt)
    r.append(stoss.r.copy())
    # Gib eine Information zum Fortschritt der Simulation aus.
    print(f'Zeitschritt {i + 1} von {t.size}')
r = np.array(r)

# Erzeuge eine Figure und eine Axes mit entsprechenden
# Beschriftungen für die Animation der Bewegung der Teilchen.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-2.1, 2.1)
ax.grid()

# Erzeuge einen Plot für die Bahnkurve des Teilchens.
plot_bahn, = ax.plot([], [], color='red')

# Erstelle die Farbtabelle und erzeuge ein Objekt, das jeder
# Masse eine Farbe zuordnet.
mapper = mpl.cm.ScalarMappable(cmap=mpl.cm.jet)
mapper.set_array(stoss.massen)
mapper.autoscale()

# Erzeuge für jedes Teilchen einen Kreis mit passendem Radius.
kreise = []
for radius, masse in zip(stoss.radien, stoss.massen):
    kreis = mpl.patches.Circle([0, 0], radius, visible=False,
                               color=mapper.to_rgba(masse))
    ax.add_patch(kreis)
    kreise.append(kreis)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Positionen der Teilchen.
    for kreis, ort in zip(kreise, r[n]):
        kreis.set_center(ort)
        kreis.set_visible(True)

    # Aktualisiere die Trajektorie des Pollenkörpers (Index 0).
    plot_bahn.set_data(r[:n + 1, 0, 0], r[:n + 1, 0, 1])
    return kreise + [plot_bahn]


# Erstelle die Animation und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
