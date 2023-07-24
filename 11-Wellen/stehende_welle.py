"""Animation einer stehenden Welle (1d)."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.01
# Dargestellter Bereich von x=-x_max bis x=x_max [m].
x_max = 20.0

# Amplitude [a.u.] und Frequenz [Hz] der Welle von links.
A_links = 1.0
f_links = 1.0
# Amplitude [a.u.] und Frequenz [Hz] der Welle von rechts.
A_rechts = 1.0
f_rechts = 1.0
# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 10.0

# Berechne Kreisfrequenz und Kreiswellenzahl.
omega_links = 2 * np.pi * f_links
omega_rechts = 2 * np.pi * f_rechts
k_links = omega_links / c
k_rechts = omega_rechts / c

# Erzeuge eine x-Achse.
x = np.linspace(-x_max, x_max, 500)

# Erzeuge eine Figure und eines Axes.
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-1.2 * (A_links + A_rechts), 1.2 * (A_links + A_rechts))
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$u$')
ax.grid()

# Erzeuge die Plots für die beiden Wellen und deren Summe.
plot_welle_links, = ax.plot([], [], '-r', zorder=5, linewidth=2,
                            label='von links')
plot_welle_rechts, = ax.plot([], [], '-b', zorder=5, linewidth=2,
                             label='von rechts')
plot_summe, = ax.plot([], [], '-k', zorder=2, linewidth=2.5,
                      label='Überlagerung')

# Füge die entsprechenden Legendeneinträge hinzu.
ax.legend(loc='upper right', ncol=2)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Bestimme die aktuelle Zeit.
    t = dt * n

    # Berechne die Auslenkung der von links einlaufenden Welle.
    phi_links = omega_links * t - k_links * (x + x_max)
    u_links = A_links * np.sin(phi_links)
    u_links[phi_links < 0] = 0

    # Berechne die Auslenkung der von rechts einlaufenden Welle.
    phi_rechts = omega_rechts * t + k_rechts * (x - x_max)
    u_rechts = A_rechts * np.sin(phi_rechts)
    u_rechts[phi_rechts < 0] = 0

    # Aktualisiere die beiden Darstellungen.
    plot_welle_links.set_data(x, u_links)
    plot_welle_rechts.set_data(x, u_rechts)

    # Aktualisiere die Überlagerung.
    plot_summe.set_data(x, u_links + u_rechts)

    return plot_welle_links, plot_welle_rechts, plot_summe


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
