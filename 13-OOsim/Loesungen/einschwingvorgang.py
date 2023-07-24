"""Simulation eines Masse-Feder-Pendels mit Anregung."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import schraubenfeder

# Simulationszeit und Zeitschrittweite [s].
t_max = 20.0
dt = 0.01
# Masse des Körpers [kg].
m = 0.1
# Federkonstante [N/m].
D = 2.5
# Reibungskoeffizient [kg / s].
b = 0.05
# Anfangslänge und -radius der Feder [m].
l0 = 0.3
r0 = 0.03
# Erdbeschleunigung [m/s²].
g = 9.81
# Anregungskreisfrequenz [1/s].
omega = 6.0
# Anregungsamplitude [m].
amplitude = 0.1
# Anfangsposition der Masse = Gleichgewichtsposition [m].
y0 = -l0 - m * g / D
# Anfangsgeschwindigkeit [m/s].
v0 = 0.0


def y_a(t):
    """Auslenkung des Aufhängepunktes als Funktion der Zeit."""
    return amplitude * np.sin(omega * t)


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    y, v = np.split(u, 2)
    F = D * (y_a(t) - l0 - y) - m * g - b * v
    return np.concatenate([v, F / m])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.array([y0, v0])

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
y, v = result.y

# Erzeuge eine Figure und ein GridSpec-Objekt.
fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)
gridspec = fig.add_gridspec(1, 2, width_ratios=[1, 5])

# Erzeuge eine Axes für die animierte Darstellung.
ax_anim = fig.add_subplot(gridspec[0, 0])
ax_anim.set_ylabel('$y$ [m]')
ax_anim.set_aspect('equal')
ax_anim.set_xlim(-2 * r0, 2 * r0)
ax_anim.set_xticks([])

# Erzeuge eine Axes für den Plot des Zeitverlaufs.
ax_zeitverlauf = fig.add_subplot(gridspec[0, 1], sharey=ax_anim)
ax_zeitverlauf.grid()
ax_zeitverlauf.set_xlabel('$t$ [s]')
ax_zeitverlauf.tick_params(labelleft=False)
ax_zeitverlauf.set_xlim(0, t_max)
ax_zeitverlauf.set_ylim(np.min(y) - 0.2 * amplitude, 
                        np.max(y_a(t) + 0.2 * amplitude))

# Erzeuge die Grafikelemente für die Aufhängung, die Feder
# und die Masse sowie zwei Linienplots für die Auslenkung der
# Masse und der Aufhängung als Funktion der Zeit.
plot_aufhaengung, = ax_anim.plot([], [], 'bo', zorder=5)
plot_masse, = ax_anim.plot([], [], 'ro', zorder=5)
plot_auslenkung_masse, = ax_zeitverlauf.plot([], [], 'r-')
plot_auslenkung_aufhg, = ax_zeitverlauf.plot([], [], '-b')
plot_feder = schraubenfeder.Schraubenfeder([0, 0], [0, -1],
                                           color='black',
                                           n_wdg=10, r0=r0, a=r0,
                                           l0=l0, zorder=4)
ax_anim.add_artist(plot_feder)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die beiden Linienplots.
    plot_auslenkung_masse.set_data(t[:n + 1], y[:n + 1])
    plot_auslenkung_aufhg.set_data(t[:n + 1], y_a(t[:n + 1]))

    # Aktualisiere die Position der Aufhängung und der Masse.
    plot_aufhaengung.set_data([0], [y_a(t[n])])
    plot_masse.set_data([0], [y[n]])

    # Aktualisiere die Darstellung der Schraubenfeder.
    plot_feder.startpunkt = np.array([0, y_a(t[n])])
    plot_feder.endpunkt = np.array([0, y[n]])

    return (plot_masse, plot_feder, plot_aufhaengung,
            plot_auslenkung_aufhg, plot_auslenkung_masse)


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
