"""Verformung einer Brückenkonstruktion.

Die Kraftverteilung in einem 2-dimensionalen elastischen Stabwerk
sowie die Verformung des Stabwerks werden in der linearen
Näherung für kleine Deformationen berechnet.
"""

import numpy as np
import matplotlib.pyplot as plt
import stabwerke

# Lege einen Skalierungsfaktor für die Kraftvektoren fest.
scal_kraft = 0.0001
# Elastizitätsmodul [N/m²].
E_modul = 210e9
# Querschnittsfläche der Stäbe [m²].
flaeche = 5e-2 ** 2
# Dichte des Stabmaterials [kg / m³].
rho = 7860.0
# Geometrie der Brücke.
punkte = np.array([[0, 0], [4, 0], [8, 0],
                   [12, 0], [16, 0], [20, 0],
                   [2, 2], [6, 2], [10, 2],
                   [14, 2], [18, 2]], dtype=float)
indizes_stuetz = [0, 5]
staebe = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                   [6, 7], [7, 8], [8, 9], [9, 10],
                   [0, 6], [1, 7], [2, 8], [3, 9], [4, 10],
                   [6, 1], [7, 2], [8, 3], [9, 4], [10, 5]])

# Berechne die Steifigkeiten der Stäbe.
S = np.ones(len(staebe)) * E_modul * flaeche

# Erstelle ein linearisiertes elastisches Stabwerk.
stabwerk = stabwerke.StabwerkElastischLin(punkte, indizes_stuetz,
                                          staebe, steifigkeiten=S)

# Lege die Gewichtskraft der Knotenpunkte durch die
# Gewichtskräfte der angrenzenden Stäbe fest:
stabwerk.punktmassen = np.zeros(len(punkte))
for i, stab in enumerate(staebe):
    masse = stabwerk.stablaengen()[i] * flaeche * rho / 2
    stabwerk.punktmassen[stab] += masse

# Berechne die Gleichgewichtsposition.
stabwerk.suche_gleichgewichtsposition()

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure(figsize=(12, 3))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1, 21)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.grid()

# Stelle das Stabwerk dar.
plot = stabwerke.PlotStabwerk(ax, stabwerk, scal_kraft=scal_kraft)
plt.show()
