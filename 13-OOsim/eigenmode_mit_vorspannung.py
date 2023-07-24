"""Eigenmoden eines ganz einfachen vorgespannten Stabwerks."""

import numpy as np
import matplotlib.pyplot as plt
import stabwerke

# Erstelle das Stabwerk.
punkte = np.array([[-0.1, 0.0], [0.0, 0.0],  [0.1, 0.0]])
indizes_stuetz = [0, 2]
staebe = np.array([[0, 1], [1, 2]])
steifigkeiten = np.array([1e3, 1e3])
massen = np.array([0.0, 1.0, 0.0])
stabw = stabwerke.StabwerkElastisch(punkte,
                                    indizes_stuetz, staebe,
                                    steifigkeiten=steifigkeiten,
                                    punktmassen=massen)

# Suche die Gleichgewichtsposition.
stabw.suche_gleichgewichtsposition()

# Linearisiere um die gefundene Gleichgewichtsposition.
stabw_lin = stabwerke.StabwerkElastischLin.from_stabwerk_elastisch(
    stabw)

# Erzeuge eine Figure und zwei Axes.
fig = plt.figure(figsize=(5, 5))
fig.set_tight_layout(True)

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlim(-0.12, 0.12)
ax1.set_ylim(-0.05, 0.05)
ax1.tick_params(labelbottom=False)
ax1.set_ylabel('$y$ [m]')
ax1.set_aspect('equal')
ax1.grid()

ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, sharey=ax1)
ax2.set_xlabel('$x$ [m]')
ax2.set_ylabel('$y$ [m]')
ax2.set_aspect('equal')
ax2.grid()

# Stelle die Eigenmoden animiert dar.
animationen = []
for i, ax in enumerate([ax1, ax2]):
    freq = stabw_lin.eigenmoden()[0][i]
    ax.set_title(f'$f$ = {freq:.2f} Hz')
    p = stabwerke.AnimationEigenmode(ax, stabw_lin, eigenmode=i,
                                     amplitude=0.02, cmap='jet',
                                     arrows=False, annot=False)
    animationen.append(p)

plt.show()
