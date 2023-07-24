"""Animation einer Transversal- und einer Longitudinalwelle."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.005
# Anzahl der dargestellten Teilchen.
n_teilchen = 51
# Kettenlänge [m].
kettenlaenge = 20
# Amplitude A [m] und Frequenz f [Hz] der Welle.
amplitude = 0.8
frequenz = 1.0
# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 10.0

# Berechne Kreisfrequenz und Kreiswellenzahl.
omega = 2 * np.pi * frequenz
k = omega / c

# Wähle einen Punkt aus, der rot markiert dargestellt wird.
index_mark = n_teilchen // 2

# Lege die Ruhepositionen der einzelnen Massenpunkte fest.
x = np.linspace(0, kettenlaenge, n_teilchen)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(12, 4))

# Erzeuge eine Axes für die Transversalwelle.
ax_trans = fig.add_subplot(2, 1, 1)
ax_trans.set_xlim(0, np.max(x))
ax_trans.set_ylim(-1.5 * amplitude, 1.5 * amplitude)
ax_trans.tick_params(labelbottom=False)
ax_trans.set_ylabel('$y$ [m]')
ax_trans.grid()

# Erzeuge eine Axes für die Longitudinalwelle.
ax_long = fig.add_subplot(2, 1, 2, sharex=ax_trans)
ax_long.set_ylim(-1.5 * amplitude, 1.5 * amplitude)
ax_long.set_xlabel('$x$ [m]')
ax_long.set_ylabel('$y$ [m]')
ax_long.grid()

# Erzeuge je einen Plot mit kleinen Punkten für die Darstellung
# der Ruhelage.
ax_trans.plot(x, 0 * x, '.', color='lightblue', zorder=4)
ax_long.plot(x, 0 * x, '.', color='lightblue', zorder=4)
ax_trans.plot([x[index_mark]], [0], '.', color='pink', zorder=5)
ax_long.plot([x[index_mark]], [0], '.', color='pink', zorder=5)

# Erzeuge je einen Punktplot (blau) für die beiden Wellen und
# lege jeweils einen Punktplot (rot) für den markierten
# Massenpunkt darüber.
plot_trans, = ax_trans.plot([], [], 'o', color='blue', zorder=6)
plot_long, = ax_long.plot([], [], 'o', color='blue', zorder=6)
plot_trans_mark, = ax_trans.plot([], [],
                                 'o', color='red', zorder=7)
plot_long_mark, = ax_long.plot([], [],
                               'o', color='red', zorder=7)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Bestimme die aktuelle Zeit und berechne die
    # Momentanauslenkung u der Welle.
    t = dt * n
    u = amplitude * np.cos(k * x - omega * t)

    # Aktualisiere die beiden Darstellungen.
    plot_trans.set_data(x, u)
    plot_long.set_data(x + u, 0 * u)

    # Aktualisiere die Position des hervorgehobenen Punktes.
    plot_trans_mark.set_data([x[index_mark]], [u[index_mark]])
    plot_long_mark.set_data([x[index_mark] + u[index_mark]], [0])

    return plot_long, plot_trans, plot_long_mark, plot_trans_mark


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
