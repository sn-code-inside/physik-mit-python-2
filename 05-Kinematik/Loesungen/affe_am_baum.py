"""Der Schuss auf einen fallenden Affen.

Darstellung der Bahnkurve eines Schusses, der in direkter
Linie auf einen Affen abgegeben wird. Zeitgleich mit dem
Schuss lässt sich der Affe fallen.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Anfangsort des Pfeils [m].
r0_pfeil = np.array([0.0, 0.0])
# Anfangsort des Affen [m].
r0_affe = np.array([3.0, 2.0])
# Anfangsgeschwindigkeit [m/s].
betrag_v0_pfeil = 9.0
# Erdbeschleunigung [m/s²].
g = 9.81
# Zeitschrittweite [s].
dt = 0.001

# Berechne den Vektor der Abschussgeschwindigkeit.
r0_pfeil_affe = r0_affe - r0_pfeil
v0 = betrag_v0_pfeil * r0_pfeil_affe / np.linalg.norm(r0_pfeil_affe)

# Lege den Vektor der Erdbeschleunigung fest.
a = np.array([0, -g])

# Berechne den Zeitpunkt, zu dem der Pfeil den Affen trifft.
t_ende = (r0_affe[0] - r0_pfeil[0]) / v0[0]

# Erzeuge ein Array von Zeitpunkten und berechne die Position
# von Pfeil und Affe zu jedem Zeitpunkt.
t = np.arange(0, t_ende, dt).reshape(-1, 1)
r_pfeil = r0_pfeil + v0 * t + 0.5 * a * t**2
r_affe = r0_affe + 0.5 * a * t**2

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(0, 1.1 * max(r0_pfeil[0], r0_affe[0]))
ax.set_ylim(0, 1.1 * max(r0_pfeil[1], r0_affe[1]))
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')
ax.grid()

# Plotte die Ziellinie des Schusses als Gerade.
plot_gerade = ax.plot([r0_pfeil[0], r0_affe[0]],
                      [r0_pfeil[1], r0_affe[1]],
                      '--', color='black', lw='0.5')

# Plotte die Bahnkurve des Pfeils.
plot_bahn_pfeil, = ax.plot(r_pfeil[:, 0], r_pfeil[:, 1],
                           color='red', zorder=3)

# Erzeuge je einen Punktplot für die Position der beiden Körper.
plot_pfeil, = ax.plot([], [], 'o', color='red', zorder=3)
plot_affe, = ax.plot([], [], 'o', color='blue', zorder=2)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position der beiden Punkte.
    plot_pfeil.set_data(r_pfeil[n].reshape(-1, 1))
    plot_affe.set_data(r_affe[n].reshape(-1, 1))

    # Plotte die Bahnkurve des Pfeils bis zur aktuellen Zeit.
    plot_bahn_pfeil.set_data(r_pfeil[:n + 1, 0],
                             r_pfeil[:n + 1, 1])

    return plot_pfeil, plot_affe, plot_bahn_pfeil


# Erzeuge das Animationsobjekt und starte die Animation
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True,
                                  repeat_delay=1000)
plt.show()
