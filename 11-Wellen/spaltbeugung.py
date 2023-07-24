"""Beugungsfeld eines Spaltes nach dem huygensschen Prinzip."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors

# Wellenlänge [m].
wellenlaenge = 1.0
# Anzahl der Elementarwellen.
n_wellen = 100
# Spaltbreite [m].
spaltbreite = 10

# Lege die Startpositionen jeder Elementarwelle fest.
positionen = np.zeros((n_wellen, 2))
positionen[:, 0] = np.linspace(-spaltbreite / 2, spaltbreite / 2,
                               n_wellen)

# Lege die Phase jeder Elementarwelle fest [rad].
phasen = np.zeros(n_wellen)

# Lege die Amplitude jeder Elementarwelle fest.
amplituden = spaltbreite / n_wellen * np.ones(n_wellen)

# Berechne die Wellenzahl.
k = 2 * np.pi / wellenlaenge

# Lege das Gitter für die Auswertung der Wellenfunktion fest.
x_lin = np.linspace(-6 * spaltbreite, 6 * spaltbreite, 500)
y_lin = np.linspace(0.1 * spaltbreite, 12 * spaltbreite, 500)
x, y = np.meshgrid(x_lin, y_lin)

# Lege ein Array komplexer Zahlen der passenden Größe an, das mit
# Nullen gefüllt ist.
u = np.zeros(x.shape, dtype=complex)

# Addiere die komplexe Amplitude jeder Elementarwelle.
for A, (x0, y0), phi0 in zip(amplituden, positionen, phasen):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    cos_theta = (y - y0) / r
    u += A * np.exp(1j * (k * r + phi0)) / np.sqrt(r) * cos_theta

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x / \\lambda$')
ax.set_ylabel('$y / \\lambda$')

# Stelle das Betragsquadrat der Amplitude als Bild dar.
image = ax.imshow(np.abs(u) ** 2,
                  interpolation='bicubic', origin='lower',
                  extent=(np.min(x_lin), np.max(x_lin),
                          np.min(y_lin), np.max(y_lin)),
                  norm=mpl.colors.LogNorm(vmin=0.001, vmax=2),
                  cmap='inferno')

# Füge einen Farbbalken hinzu.
fig.colorbar(image, label='Intensität [a.u.]')

# Zeige das Bild an.
plt.show()
