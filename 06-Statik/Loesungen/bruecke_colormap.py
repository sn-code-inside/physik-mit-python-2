"""Verformung einer Brückenkonstruktion.

Die Kraftverteilung in einem 2-dimensionalen elastischen Stabwerk
sowie die Verformung des Stabwerks werden in der linearen
Näherung für kleine Deformationen berechnet.

Die Darstellung der Kräfte erfolgt hier durch eine Farbtabelle.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm

# Lege die maximale Kraft für die Farbtabelle fest [N].
F_max = 16000

# Lege die Positionen der Punkte fest [m].
punkte = np.array([[0, 0], [4, 0], [8, 0],
                   [12, 0], [16, 0], [20, 0],
                   [2, 2], [6, 2], [10, 2],
                   [14, 2], [18, 2]], dtype=float)

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
indizes_stuetz = [0, 5]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                   [6, 7], [7, 8], [8, 9], [9, 10],
                   [0, 6], [1, 7], [2, 8], [3, 9], [4, 10],
                   [6, 1], [7, 2], [8, 3], [9, 4], [10, 5]])

# Elastizitätsmodul [N/m²].
E_modul = 210e9
# Querschnittsfläche der Stäbe [m²].
flaeche = 5e-2 ** 2
# Dichte des Stabmaterials [kg/m³].
rho = 7860.0
# Erdbeschleunigung [m/s²].
g = 9.81

# Definiere die Dimension sowie die Anzahl der Punkte, Stäbe, etc.
n_punkte, n_dim = punkte.shape
n_staebe = len(staebe)
n_stuetz = len(indizes_stuetz)
n_knoten = n_punkte - n_stuetz

# Lege die Steifigkeit jedes Stabes fest.
steifigkeiten = np.ones(n_staebe) * E_modul * flaeche

# Erzeuge eine Liste mit den Indizes der Knoten.
indizes_knoten = list(set(range(n_punkte)) - set(indizes_stuetz))


def ev(i_pkt, i_stb, koord=punkte):
    """Bestimme den Einheitsvektor in einem Punkt für einen Stab.

    Args:
        i_pkt (int):
            Index des betrachteten Punktes.
        i_stb (int):
            Index des betrachteten Stabes.
        koord (np.ndarray):
            Koordinaten der Punkte (n_punkte × n_dim).

    Returns:
        np.ndarray: Berechneter Einheitsvektor oder der Nullvektor,
                    wenn der Stab den Punkt nicht enthält.
    """
    stb = staebe[i_stb]
    if i_pkt not in stb:
        return np.zeros(n_dim)
    if i_pkt == stb[0]:
        vektor = koord[stb[1]] - koord[i_pkt]
    else:
        vektor = koord[stb[0]] - koord[i_pkt]
    return vektor / np.linalg.norm(vektor)


def laenge(i_stb, koord=punkte):
    """Berechne die Länge eines Stabes.

    Args:
        i_stb (int):
            Index des betrachteten Stabes.
        koord (np.ndarray):
            Koordinaten der Punkte (n_punkte × n_dim).

    Returns:
        float: Länge des Stabes.
    """
    i1, i2 = staebe[i_stb]
    return np.linalg.norm(koord[i2] - koord[i1])


# Lege die äußere Kraft auf jeden Knotenpunkt durch die
# Gewichtskraft der angrenzenden Stäbe fest:
F_ext = np.zeros((n_punkte, n_dim))
for i, stab in enumerate(staebe):
    for k in stab:
        F_ext[k, 1] -= laenge(i) * flaeche * rho * g / 2


# Stelle das Gleichungssystem für die Kräfte auf.
A = np.zeros((n_knoten, n_dim, n_knoten, n_dim))
for n, k in enumerate(indizes_knoten):
    for m, j in enumerate(indizes_knoten):
        for i in range(n_staebe):
            A[n, :, m, :] -= (steifigkeiten[i] / laenge(i)
                              * np.outer(ev(k, i), ev(j, i)))
A = A.reshape((n_knoten * n_dim, n_knoten * n_dim))

# Löse das Gleichungssystem A @ dr = -F_ext.
dr = np.linalg.solve(A, -F_ext[indizes_knoten].reshape(-1))
dr = dr.reshape(n_knoten, n_dim)

# Das Array dr enthält nur die Verschiebungen der
# Knotenpunkte. Für den weiteren Ablauf des Programms ist es
# praktisch, stattdessen ein Array zu haben, das die gleiche
# Größe wie das Array 'punkte' hat und an den Stützstellen
# Nullen enthält.
delta_r = np.zeros((n_punkte, n_dim))
delta_r[indizes_knoten] = dr

# Berechne die Kraft in jedem der Stäbe in linearer Näherung.
F = np.zeros(n_staebe)
for i_stab, (j, k) in enumerate(staebe):
    F[i_stab] = (steifigkeiten[i_stab] / laenge(i_stab)
                 * ev(k, i_stab) @ (delta_r[j] - delta_r[k]))

# Berechne die äußeren Kräfte auf die Stützpunkte.
for i_stuetz in indizes_stuetz:
    for i_stab in range(n_staebe):
        F_ext[i_stuetz] -= F[i_stab] * ev(i_stuetz, i_stab)

# Berechne die neue Position der einzelnen Punkte.
punkte_neu = punkte + delta_r

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure(figsize=(12, 5))
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1, 21)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.grid()

# Erzeuge einen Mapper, der jeder Kraft eine Farbe zuordnet.
mapper = mpl.cm.ScalarMappable(cmap=mpl.cm.jet)
mapper.set_array([-F_max / 1e3, F_max / 1e3])
mapper.autoscale()

# Erzeuge einen Farbbalken am unteren Rand des Bildes.
fig.colorbar(mapper, format='%.1f', label='Kraft [kN]',
             orientation='horizontal', shrink=0.7, ax=ax)

# Plotte die Stäbe und beschrifte diese mit dem Wert der Kraft.
for stab, kraft in zip(staebe, F):
    ax.plot(punkte[stab, 0], punkte[stab, 1],
            linewidth=3, color=mapper.to_rgba(kraft / 1e3))
    position = np.mean(punkte[stab], axis=0)
    annot = ax.annotate(f'{kraft / 1e3:+.1f} kN', position,
                        color='black',
                        horizontalalignment='center')
    annot.draggable(True)

# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
ax.plot(punkte[indizes_knoten, 0], punkte[indizes_knoten, 1], 'bo')
ax.plot(punkte[indizes_stuetz, 0], punkte[indizes_stuetz, 1], 'ro')

# Zeige die Grafik an.
plt.show()
