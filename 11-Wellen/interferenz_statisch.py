"""Interferenz zweier kreisförmiger Wellen (Intensität)."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors

# Wellenlänge der ausgestrahlten Wellen [m].
wellenlaenge = 1.0
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

# Lege ein Raster der angegebenen Größe an, auf dem die
# Wellenfunktionen ausgewertet wird.
x_lin = np.linspace(-xy_max, xy_max, n_punkte)
y_lin = np.linspace(-xy_max, xy_max, n_punkte)
x, y = np.meshgrid(x_lin, y_lin)

# Berechne die Wellenzahl.
k = 2 * np.pi / wellenlaenge

# Lege ein Array komplexer Zahlen der passenden Größe an, das mit
# Nullen gefüllt ist.
u = np.zeros(x.shape, dtype=complex)

# Addiere für jede Quelle die entsprechende komplexe Amplitude.
for A, (x0, y0), phi0 in zip(amplituden, positionen, phasen):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    u += A * np.exp(1j * (k * r + phi0)) / np.sqrt(r)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')

# Stelle das Betragsquadrat der Amplitude als Bild dar.
image = ax.imshow(np.abs(u) ** 2, origin='lower',
                  extent=(np.min(x_lin), np.max(x_lin),
                          np.min(y_lin), np.max(y_lin)),
                  norm=mpl.colors.LogNorm(vmin=0.01, vmax=10),
                  cmap='inferno', interpolation='bicubic')

# Füge einen Farbbalken hinzu.
fig.colorbar(image, label='Intensität [a.u.]')

# Zeige das Bild an.
plt.show()
