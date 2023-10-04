"""Animation der Bewegung entlang einer Schraubenbahn."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import mpl_toolkits.mplot3d

# Radius der Schraubenbahn [m].
radius = 3.0
# Umlaufdauer [s].
T = 8.0
# Zeitschrittweite [s].
dt = 0.05
# Geschwindigkeit in z-Richtung [m/s].
v_z = 0.5
# Anzahl der Umläufe.
n_umlaeufe = 5

# Berechne die Winkelgeschwindigkeit [1/s].
omega = 2 * np.pi / T

# Erzeuge ein Array von Zeitpunkten für N Umläufe.
t = np.arange(0, n_umlaeufe * T, dt)

# Berechne die Position des Massenpunktes für jeden Zeitpunkt.
r = np.empty((t.size, 3))
r[:, 0] = radius * np.cos(omega * t)
r[:, 1] = radius * np.sin(omega * t)
r[:, 2] = v_z * t

# Berechne den Geschwindigkeits- und Beschleunigungsvektor.
v = (r[1:] - r[:-1]) / dt
a = (v[1:] - v[:-1]) / dt

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d', elev=30, azim=45)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_zlabel('$z$ [m]')
ax.grid()


class Arrow3D(mpl.patches.FancyArrowPatch):
    """Darstellung eines Pfeiles in einer 3D-Grafik.

    Args:
        posA (tuple):
            Koordinaten (x, y, z) des Startpunktes.
        posB (tuple):
            Koordinaten (x, y, z) des Endpunktes.
        *args:
            Weitere Argumente für mpl.patches.FancyArrowPatch
        **kwargs:
            Weitere Schlüsselwortargumente für
            mpl.patches.FancyArrowPatch
    """

    def __init__(self, posA, posB, *args, **kwargs):
        super().__init__(posA[0:2], posB[0:2], *args, **kwargs)
        self._pos = np.array([posA, posB])

    def set_positions(self, posA, posB):
        """Setze den Start- und Endpunkt des Pfeils."""
        self._pos = np.array([posA, posB])

    def do_3d_projection(self, renderer=None):
        """Projiziere die Punkte in die Bildebene."""
        p = mpl_toolkits.mplot3d.proj3d.proj_transform(*self._pos.T, self.axes.M)
        p = np.array(p)
        super().set_positions(p[:, 0], p[:, 1])
        return np.min(p[2, :])


# Plotte die Bahnkurve.
plot_bahn, = ax.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=0.7)

# Erzeuge einen Punktplot, der die Position der Masse darstellt.
plot_punkt, = ax.plot([], [], [], 'o', color='red')

# Erzeuge Pfeile für die Geschwindigkeit und die
# Beschleunigung und füge diese der Axes hinzu.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = Arrow3D((0, 0, 0), (0, 0, 0),
                  color='red', arrowstyle=style)
pfeil_a = Arrow3D((0, 0, 0), (0, 0, 0),
                  color='black', arrowstyle=style)
ax.add_patch(pfeil_v)
ax.add_patch(pfeil_a)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Setze den Start- und Endpunkt des Geschwindigkeitspfeils.
    if n < len(v):
        pfeil_v.set_positions(r[n], r[n] + v[n])

    # Setze den Start- und Endpunkt des Beschleunigungspfeils.
    if n < len(a):
        pfeil_a.set_positions(r[n], r[n] + a[n])

    # Aktualisiere die Position des Punktes.
    plot_punkt.set_data_3d(r[n, :].reshape(-1, 1))

    return plot_punkt, pfeil_v, pfeil_a


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30)
plt.show()
