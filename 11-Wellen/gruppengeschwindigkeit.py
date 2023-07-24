"""Gruppengeschwindigkeit.

Der Begriff der Gruppengeschwindigkeit wird am Beispiel der
Überlagerung zweier sinusförmiger Wellen demonstriert.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.05
# Dargestellter Bereich von x=0 bis x=x_max [m].
x_max = 30.0
# Wellenlängen der beiden Wellen [m].
wellenlaenge1 = 0.95
wellenlaenge2 = 1.05
# Phasengeschwindigkeit der beiden Wellen [m/s].
c1 = 0.975
c2 = 1.025

# Berechne die Wellenzahlen und die Kreisfrequenzen.
k1 = 2 * np.pi / wellenlaenge1
k2 = 2 * np.pi / wellenlaenge2
omega1 = c1 * k1
omega2 = c2 * k2

# Berechne die Gruppengeschwindigkeit.
c_gr = (omega1 - omega2) / (k1 - k2)

# Berechne die mittlere Phasengeschwindigkeit.
c_ph = (c1 + c2) / 2

# Lege ein Array von x-Werten an.
x = np.linspace(0, x_max, 1000)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('Auslenkung [a.u.]')
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(-2.2, 2.2)

# Erzeuge drei Linienplots für die Wellen.
plot_welle1, = ax.plot([], [], '-r', zorder=5, linewidth=1)
plot_welle2, = ax.plot([], [], '-b', zorder=5, linewidth=1)
plot_summe, = ax.plot([], [], '-k', zorder=2, linewidth=2)

# Erzeuge zwei Linienplots zur Darstellung der Geschwindigkeit.
linie_gruppengeschw, = ax.plot([], [], '-', color='gray',
                               zorder=1, linewidth=4)
linie_phasengeschw, = ax.plot([], [], '-m', zorder=1, linewidth=4)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Bestimme die aktuelle Zeit.
    t = dt * n

    # Werte die Wellenfunktionen aus.
    u1 = np.cos(k1 * x - omega1 * t)
    u2 = np.cos(k2 * x - omega2 * t)

    # Stelle die beiden Wellen und ihre Überlagerung dar.
    plot_welle1.set_data(x, u1)
    plot_welle2.set_data(x, u2)
    plot_summe.set_data(x, u1 + u2)

    # Bewege die Linie mit der Gruppengeschwindigkeit.
    position = (t * c_gr) % x_max
    linie_gruppengeschw.set_data(2 * [position], ax.get_ylim())

    # Bewege die Linie mit der Phasengeschwindigkeit.
    position = (t * c_ph) % x_max
    linie_phasengeschw.set_data(2 * [position], ax.get_ylim())

    return (plot_welle1, plot_welle2, plot_summe,
            linie_gruppengeschw, linie_phasengeschw)


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
