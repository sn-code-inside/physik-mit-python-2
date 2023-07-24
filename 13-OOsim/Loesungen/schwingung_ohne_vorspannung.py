"""Nichtlineare Schwingung eines einfachen Stabwerks."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import stabwerke

# Simulationszeit und Zeitschrittweite [s].
t_max = 10.0
dt = 0.02

# Erstelle das Stabwerk.
punkte = np.array([[-0.1, 0.0], [0.0, 0.0],  [0.1, 0.0]])
indizes_stuetz = [0, 2]
staebe = np.array([[0, 1], [1, 2]])
steifigkeiten = np.array([1e3, 1e3])
massen = np.array([0.0, 1.0, 0.0])
stabw_lin = stabwerke.StabwerkElastisch(
    punkte, indizes_stuetz, staebe,
    steifigkeiten=steifigkeiten, punktmassen=massen)

# Schalte die Schwerkraft aus.
stabw_lin.g_vector[:] = 0

# Lege die Anfangsauslenkung des mittleren Punktes fest.
stabw_lin.punkte[1, 0] = 0
stabw_lin.punkte[1, 1] = 0.01


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)
    # Verschiebe den mittleren Punkt an die angegebene Position.
    stabw_lin.punkte[1] = r
    # Berechne die Kräfte.
    kraefte = stabw_lin.gesamtkraefte()
    # Berechne die Beschleunigung des mittleren Punkts.
    a = kraefte[1] / stabw_lin.punktmassen[1]
    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((stabw_lin.punkte[1],
                     np.zeros(stabw_lin.n_dim)))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(15, 4))
fig.set_tight_layout(True)

# Erzeuge eine Axes und plotte den Zeitverlauf der Auslenkung.
ax_plot = fig.add_subplot(1, 2, 1)
ax_plot.set_xlabel('$t$ [s]')
ax_plot.set_ylabel('Auslenkung [m]')
ax_plot.grid()
ax_plot.plot(t, r[0], label='$x$')
ax_plot.plot(t, r[1], label='$y$')
ax_plot.legend(loc='upper right')

# Erzeuge eine Axes für die animierte Darstellung des Stabwerks.
ax_anim = fig.add_subplot(1, 2, 2)
ax_anim.set_xlim(-0.12, 0.12)
ax_anim.set_ylim(-0.05, 0.05)
ax_anim.set_xlabel('$x$ [m]')
ax_anim.set_ylabel('$y$ [m]')
ax_anim.set_aspect('equal')
ax_anim.grid()

# Plotte das Stabwerk.
plot_stabwerk = stabwerke.PlotStabwerk(ax_anim, stabw_lin,
                                       cmap='jet', linewidth_stab=3,
                                       annot=False, arrows=False)
for artist in plot_stabwerk.artists:
    artist.set_visible(False)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    stabw_lin.punkte[1] = r[:, n]
    plot_stabwerk.update_stabwerk()
    for artist in plot_stabwerk.artists:
        artist.set_visible(True)

    return plot_stabwerk.artists


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update,
                                  frames=t.size,
                                  interval=30, blit=True)
plt.show()
