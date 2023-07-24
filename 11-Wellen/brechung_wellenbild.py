"""Brechungsgesetz mit ebenen Wellen."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.02
# Frequenz der Sender [Hz].
f = 0.25
# Ausbreitungsgeschwindigkeit der Welle [m/s].
c1 = 5.0
c2 = 1.5
# Einfallswinkel [rad].
alpha = np.radians(50)
# Dargestellter Bereich in x- und y-Richtung: -xy_max bis +xy_max.
xy_max = 15.0
# Anzahl der Punkte in jeder Koordinatenrichtung.
n_punkte = 500

# Wir wollen die Wellenfunktionen auf einem Raster der angegebenen
# Größe auswerten.
x_lin = np.linspace(-xy_max, xy_max, n_punkte)
y_lin = np.linspace(-xy_max, xy_max, n_punkte)
x, y = np.meshgrid(x_lin, y_lin)

# Erzeuge ein n_punkte × n_punkte × 2 - Array, das für jeden
# Punkt den Ortsvektor beinhaltet.
r = np.stack((x, y), axis=2)

# Austrittswinkel der gebrochenen Strahlen nach Snellius [rad].
beta = np.arcsin(np.sin(alpha) * c2 / c1)

# Berechne die Kreisfrequenz und die Wellenzahlvektoren.
omega = 2 * np.pi * f
k1 = omega / c1 * np.array([np.sin(alpha), -np.cos(alpha)])
k2 = omega / c2 * np.array([np.sin(beta), -np.cos(beta)])

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')

# Stelle das Wellenfeld als Bild dar.
image = ax.imshow(0 * x, origin='lower',
                  extent=(np.min(x_lin), np.max(x_lin),
                          np.min(y_lin), np.max(y_lin)),
                  cmap='jet', clim=(-2, 2),
                  interpolation='bicubic')

# Zeichne eine dünne schwarze Linie, die die Grenze der beiden
# Gebiete darstellt.
plot_trennlinie, = ax.plot([np.min(x_lin), np.max(x_lin)], [0, 0],
                           '-k', linewidth=0.5, zorder=5)

# Füge einen Farbbalken hinzu.
fig.colorbar(image, label='Auslenkung [a.u.]')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Bestimme die aktuelle Zeit.
    t = dt * n

    # Werte die beiden Wellenfunktionen aus.
    u1 = np.cos(r @ k1 - omega * t)
    u2 = np.cos(r @ k2 - omega * t)

    # Erzeuge ein Array, das in der oberen Halbebene die Welle
    # u1 darstellt und in der unteren Halbebene die Welle u2.
    u = u1
    u[y < 0] = u2[y < 0]

    # Aktualisiere die Bilddaten.
    image.set_data(u)

    return image, plot_trennlinie


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
