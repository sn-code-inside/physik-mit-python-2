"""Simulation der elastischen Stöße mehrerer Teilchen."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
from stossprozess import Mehrteilchenstoss

# Simulationszeit und Zeitschrittweite [s].
t_max = 10
dt = 0.005

# Anfangspositionen [m].
r0 = np.array([[-1.0, 0.0],  [0.5, 0.0], [0.45, -0.05],
               [0.45, 0.05], [0.55, -0.05], [0.55, 0.05]])

# Für jede Wand wird der Abstand vom Koordinatenursprung und ein
# nach außen zeigender Normalenvektor angegeben.
wandabstaende = np.array([1.2, 1.2, 0.6, 0.6])
wandnormalen = np.array([[-1.0, 0], [1.0, 0], [0, -1.0], [0, 1.0]])

# Erzeuge ein Objekt der Klasse `Mehrteilchenstoß`.
stoss = Mehrteilchenstoss(r0, massen=0.2, radien=0.03,
                          waende=(wandabstaende, wandnormalen))

# Setze die Geschwindigkeit des ersten Teilchens
# auf 3 m/s in x-Richtung.
stoss.v[0, :] = (3.0, 0.0)

# Führe die Simulation durch, indem die Methode `zeitschritt`
# mehrfach aufgerufen wird.
t = np.arange(0, t_max, dt)
r = []
for i in range(t.size):
    stoss.zeitschritt(dt)
    r.append(stoss.r.copy())
r = np.array(r)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.6, 0.6)
ax.set_aspect('equal')
ax.grid()

# Erzeuge für jedes Teilchen einen Kreis.
kreise = []
for radius in stoss.radien:
    kreis = mpl.patches.Circle([0, 0], radius, visible=False)
    ax.add_patch(kreis)
    kreise.append(kreis)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Positionen der Teilchen.
    for kreis, ort in zip(kreise, r[n]):
        kreis.set_center(ort)
        kreis.set_visible(True)
    return kreise


# Erstelle die Animation und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
