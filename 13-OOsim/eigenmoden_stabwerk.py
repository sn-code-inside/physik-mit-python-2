"""Berechnung der Eigenmoden eines Stabwerks."""

import numpy as np
import matplotlib.pyplot as plt
import stabwerke

# Lege den Skalierungsfaktor für die Kraftvektoren fest [m/N].
scal_kraft = 0.001

# Definiere das Stabwerk.
punkte = np.array([[0, 0], [1.2, 0], [1.2, 2.1], [0, 2.1],
                   [0.6, 1.05]])
indizes_stuetz = [0, 1]
staebe = np.array([[1, 2], [2, 3], [3, 0],
                   [0, 4], [1, 4], [3, 4], [2, 4]])
steifigkeiten = np.array([5.6e6, 5.6e6, 5.6e6,
                          7.1e3, 7.1e3, 7.1e3, 7.1e3])
kraefte = np.array([[0, 0], [0, 0], [200.0, 0], [0, 0], [0, 0]])
massen = np.array([0.0, 0.0, 5.0, 5.0, 1.0])

# Suche die Gleichgewichtspositionen im elastischen Stabwerk.
sw0 = stabwerke.StabwerkElastisch(punkte, indizes_stuetz, staebe,
                                  kraefte_ext=kraefte,
                                  steifigkeiten=steifigkeiten,
                                  punktmassen=massen)
sw0.suche_gleichgewichtsposition()

# Linearisiere das System um die gefundene Gleichgewichtsposition.
sw = stabwerke.StabwerkElastischLin.from_stabwerk_elastisch(sw0)

# Bestimme die Eigenfrequenzen des Stabwerks.
eigenfrequenzen = sw.eigenmoden()[0]

# Anzahl der darzustellenden Eigenmoden.
n_moden = eigenfrequenzen.size

# Erzeuge ein geeignetes n_zeilen × n_spalten - Raster.
n_zeilen = int(np.sqrt(n_moden))
n_spalten = n_moden // n_zeilen
while n_zeilen * n_spalten < n_moden:
    n_spalten += 1

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
fig.set_tight_layout(True)

plotter = []
# Erstelle die Plots für jede Eigenmode in einer eigenen Axes.
for mode in range(n_zeilen * n_spalten):
    # Erzeuge ein neues Axes-Objekt.
    ax = fig.add_subplot(n_zeilen, n_spalten, mode + 1)
    ax.set_title(f'$f_{{{mode+1}}}$={eigenfrequenzen[mode]:.1f} Hz')
    ax.set_xlim(-0.1, 1.7)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    p = stabwerke.AnimationEigenmode(ax, sw,
                                     eigenmode=mode, amplitude=0.2,
                                     scal_kraft=0.005,
                                     arrows=False, annot=False)
    plotter.append(p)

plt.show()
