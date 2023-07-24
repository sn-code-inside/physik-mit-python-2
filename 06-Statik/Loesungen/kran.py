"""Kraftverteilung in einem 2-dimensionalen starren Stabwerk.

Berechnung der Kraftverteilung in einem Kranausleger.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm

# Lege die maximale Kraft für die Farbtabelle fest [N].
F_max = 9000

# Lege die Positionen der Punkte fest [m].
punkte = np.array([[0, 0], [0.7, 0], [2.1, 0], [3.5, 0],
                   [4.9, 0], [6.3, 0], [0, 1], [1.4, 1],
                   [2.8, 1], [4.2, 1], [5.6, 1]])

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
indizes_stuetz = [0, 6]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                   [6, 7], [7, 8], [8, 9], [9, 10],
                   [6, 1], [1, 7], [7, 2], [2, 8], [8, 3],
                   [3, 9], [9, 4], [4, 10], [10, 5]])

# Lege die äußere Kraft fest, die auf jeden Punkt wirkt [N].
# Für die Stützpunkte setzen wir diese Kraft zunächst auf 0.
# Diese wird später berechnet.
F_ext = np.zeros(punkte.shape)
F_ext[:, 1] = -100.0
F_ext[5, 1] = -1000.0
F_ext[indizes_stuetz] = 0

# Definiere die Dimension sowie die Anzahl der Punkte, Stäbe, etc.
n_punkte, n_dim = punkte.shape
n_staebe = len(staebe)
n_stuetz = len(indizes_stuetz)
n_knoten = n_punkte - n_stuetz

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
                    wenn der Stab den Punkt nicht enthält (n_dim).
    """
    stb = staebe[i_stb]
    if i_pkt not in stb:
        return np.zeros(n_dim)
    if i_pkt == stb[0]:
        vektor = koord[stb[1]] - koord[i_pkt]
    else:
        vektor = koord[stb[0]] - koord[i_pkt]
    return vektor / np.linalg.norm(vektor)


# Stelle das Gleichungssystem für die Kräfte auf.
A = np.empty((n_knoten, n_dim, n_staebe))
for n, k in enumerate(indizes_knoten):
    for i in range(n_staebe):
        A[n, :, i] = ev(k, i)
A = A.reshape(n_knoten * n_dim, n_staebe)

# Löse das Gleichungssystem A @ F = -F_ext nach den Kräften F.
b = -F_ext[indizes_knoten].reshape(-1)
F = np.linalg.solve(A, b)

# Berechne die äußeren Kräfte auf die Stützpunkte.
for i_stuetz in indizes_stuetz:
    for i_stab in range(n_staebe):
        F_ext[i_stuetz] -= F[i_stab] * ev(i_stuetz, i_stab)

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1.0, 7.0)
ax.set_ylim(-0.5, 2.0)
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
    annot = ax.annotate(f'{kraft / 1e3:+.2f} kN', position,
                        color='black',
                        horizontalalignment='center')
    annot.draggable(True)

# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
ax.plot(punkte[indizes_knoten, 0], punkte[indizes_knoten, 1], 'bo')
ax.plot(punkte[indizes_stuetz, 0], punkte[indizes_stuetz, 1], 'ro')

# Zeige die Grafik an.
plt.show()
