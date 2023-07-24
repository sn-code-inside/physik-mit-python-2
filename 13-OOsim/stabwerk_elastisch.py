"""Verformung eines elastischen Stabwerks.

Vergleich der Kraftverteilung sowie die Verformung eines
2-dimensionalen elastischen bei
     2. der Lösung des linearisierten Gleichungssystems und
     2. der wiederholten Anwendung der Linearisierung.
"""

import numpy as np
import matplotlib.pyplot as plt
import stabwerke

# Lege den Skalierungsfaktor für die Kraftvektoren fest [m/N].
scal_kraft = 0.001

# Definiere die Geometrie des Stabwerks und die äußeren Kräfte.
punkte = np.array([[0, 0], [1.2, 0], [1.2, 2.1], [0, 2.1],
                   [0.6, 1.05]])
indizes_stuetz = [0, 1]
staebe = np.array([[1, 2], [2, 3], [3, 0],
                   [0, 4], [1, 4], [3, 4], [2, 4]])
steifigkeiten = np.array([5.6e6, 5.6e6, 5.6e6,
                          7.1e3, 7.1e3, 7.1e3, 7.1e3])
F_ext = np.array([[0, 0], [0, 0], [200.0, 0], [0, 0], [0, 0]])

# Erzeuge eine Figure und zwei Axes-Objekte.
fig = plt.figure(figsize=(9, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
for ax in [ax1, ax2]:
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.set_xlim(-0.3, 1.8)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.grid()

# Erzeuge das Stabwerk in linearer Näherung und plotte es.
stabwerk_linear = stabwerke.StabwerkElastischLin(
    punkte, indizes_stuetz, staebe,
    steifigkeiten=steifigkeiten, kraefte_ext=F_ext)
stabwerk_linear.suche_gleichgewichtsposition()
stabwerke.PlotStabwerk(ax1, stabwerk_linear, scal_kraft=scal_kraft,
                       kopie=True)
ax1.set_title('1. lineare Näherung')

# Wiederhole die lineare Näherung, bis zum Erreichen des
# Gleichgewichtszustandes und stelle das Ergebnis dar.
iteration = 1
while not stabwerk_linear.ist_im_gleichgewicht():
    iteration += 1
    print(f'{iteration}. lineare Näherung.')
    stabwerk_linear.suche_gleichgewichtsposition()
stabwerke.PlotStabwerk(ax2, stabwerk_linear,
                       scal_kraft=scal_kraft)
ax2.set_title(f'{iteration}. lineare Näherung')
plt.show()
