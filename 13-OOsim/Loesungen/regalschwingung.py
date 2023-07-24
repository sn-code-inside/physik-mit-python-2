"""Schwingung eines elastischen Stabwerks."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import stabwerke
import stabwerksdynamik

# Simulationszeit und Zeitschrittweite [s].
t_max = 5.0
dt = 0.01

# Reibungskoeffizient [kg/s].
reibung = 3.5

# Definiere die Geometrie des Stabwerks und die äußeren Kräfte.
punkte = np.array([[0, 0], [1.2, 0], [1.2, 2.1], [0, 2.1],
                   [0.6, 1.05]])
indizes_stuetz = [0, 1]
staebe = np.array([[1, 2], [2, 3], [3, 0],
                   [0, 4], [1, 4], [3, 4], [2, 4]])
steifigkeiten = np.array([5.6e3, 5.6e3, 5.6e3,
                          7.1e3, 7.1e3, 7.1e3, 7.1e3])
F_ext = np.array([[0, 0], [0, 0], [0, 0], [200.0, 0], [0, 0]])
massen = np.array([0.0, 0.0, 5.0, 5.0, 1.0])

# Erzeuge das dynamische Stabwerksmodell.
stabw = stabwerksdynamik.StabwerkElastischDynamisch(
    punkte, indizes_stuetz, staebe, steifigkeiten=steifigkeiten,
    punktmassen=massen, kraefte_ext=F_ext, reibung=reibung)
stabw.suche_gleichgewichtsposition()
stabw.kraefte_ext[:, :] = 0

# Löse die zugehörigen Differentialgleichungen.
t, punkte, v = stabw.solve(t_max, t_eval=np.arange(0, t_max, dt))

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-0.6, 1.8)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')
ax.grid()

# Erzeuge einen Stabwerksplot.
plot_stabwerk = stabwerke.PlotStabwerk(ax, stabw,
                                       cmap='jet', linewidth_stab=3,
                                       arrows=False, annot=False)
for artist in plot_stabwerk.artists:
    artist.set_visible(False)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    stabw.punkte = punkte[n]
    plot_stabwerk.update_stabwerk()
    for artist in plot_stabwerk.artists:
        artist.set_visible(True)
    return plot_stabwerk.artists


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
