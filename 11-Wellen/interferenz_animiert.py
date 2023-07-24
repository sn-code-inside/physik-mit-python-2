"""Interferenz zweier kreisförmiger Wellen (Animation)."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.02
# Frequenz der Quelle [Hz].
f = 1.0
# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 1.0
# Lege die Amplitude [a.u.] jeder Quelle in einem Array ab.
amplituden = np.array([1.0, 1.0])
# Lege die Phase jeder Quelle in einem Array ab [rad].
phasen = np.radians(np.array([0, 0]))
# Lege die Position jeder Quelle in einem n × 2 - Array ab [m].
positionen = np.array([[-3.0, 0], [3.0, 0]])
# Dargestellter Bereich in x- und y-Richtung: -xy_max bis +xy_max.
xy_max = 10.0
# Anzahl der Punkte in jeder Koordinatenrichtung.
n_punkte = 500

# Wir wollen die Wellenfunktionen auf einem Raster der angegebenen
# Größe auswerten.
x_lin = np.linspace(-xy_max, xy_max, n_punkte)
y_lin = np.linspace(-xy_max, xy_max, n_punkte)
x, y = np.meshgrid(x_lin, y_lin)

# Berechne die Kreisfrequenz und die Wellenzahl.
omega = 2 * np.pi * f
k = omega / c

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')

# Stelle ein 2-dimensionales Array als Bild dar.
image = ax.imshow(0 * x, origin='lower',
                  extent=(np.min(x_lin), np.max(x_lin),
                          np.min(y_lin), np.max(y_lin)),
                  cmap='jet', clim=(-2, 2),
                  interpolation='bicubic')

# Füge einen Farbbalken hinzu.
fig.colorbar(image, label='Auslenkung [a.u.]')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Bestimme die aktuelle Zeit.
    t = dt * n

    # Lege ein mit Nullen gefülltes Array der passenden Größe an.
    u = np.zeros(x.shape)

    # Addiere nacheinander die Wellenfelder jeder Quelle.
    for A, (x0, y0), phi0 in zip(amplituden, positionen, phasen):
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        u_sender = A * np.sin(omega * t - k * r + phi0) / np.sqrt(r)
        u_sender[omega * t - k * r < 0] = 0
        u += u_sender

    # Aktualisiere die Bilddaten.
    image.set_data(u)
    return image,


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
