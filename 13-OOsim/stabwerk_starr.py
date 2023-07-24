"""Kraftverteilung in einem 2-dimensionalen starren Stabwerk."""

import numpy as np
import matplotlib.pyplot as plt
import stabwerke

# Definiere die Geometrie des Stabwerks und die äußeren Kräfte.
punkte = np.array([[0, 0], [0, 1.1], [1.2, 0], [2.5, 1.1]])
indizes_stuetz = [0, 1]
staebe = np.array([[0, 2], [1, 2], [2, 3], [1, 3]])
F_ext = np.array([[0, 0], [0, 0], [0, -147.15], [0, -98.1]])

# Erzeuge das Stabwerk.
stabwerk = stabwerke.StabwerkStarr(punkte, indizes_stuetz, staebe,
                                   kraefte_ext=F_ext)

# Gib die Stabkräfte aus.
print(stabwerk.stabkraefte_scal())

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1.0, 3.0)
ax.set_ylim(-0.5, 2.0)
ax.set_aspect('equal')
ax.grid()

# Plotte das Stabwerk.
plot = stabwerke.PlotStabwerk(ax, stabwerk, scal_kraft=0.002)
plt.show()
