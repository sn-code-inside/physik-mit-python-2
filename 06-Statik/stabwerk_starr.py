"""Kraftverteilung in einem 2-dimensionalen starren Stabwerk."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Lege den Skalierungsfaktor für die Kraftvektoren fest [m/N].
scal_kraft = 0.002

# Lege die Positionen der Punkte fest [m].
punkte = np.array([[0, 0], [0, 1.1], [1.2, 0], [2.5, 1.1]])

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
indizes_stuetz = [0, 1]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[0, 2], [1, 2], [2, 3], [1, 3]])

# Lege die äußere Kraft fest, die auf jeden Punkt wirkt [N].
# Für die Stützpunkte setzen wir diese Kraft zunächst auf 0.
# Diese wird später berechnet.
F_ext = np.array([[0, 0], [0, 0], [0, -147.15], [0, -98.1]])

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
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1.0, 3.0)
ax.set_ylim(-0.5, 2.0)
ax.set_aspect('equal')
ax.grid()

# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
ax.plot(punkte[indizes_knoten, 0], punkte[indizes_knoten, 1], 'bo')
ax.plot(punkte[indizes_stuetz, 0], punkte[indizes_stuetz, 1], 'ro')

# Plotte die Stäbe und beschrifte diese mit dem Wert der Kraft.
for stab, kraft in zip(staebe, F):
    ax.plot(punkte[stab, 0], punkte[stab, 1], color='black')
    position = np.mean(punkte[stab], axis=0)
    annot = ax.annotate(f'{kraft:+.1f} N', position, color='blue')
    annot.draggable(True)

# Zeichne die äußeren Kräfte mit roten Pfeilen in das Diagramm
# ein und erzeuge Textfelder, die den Betrag der Kraft angeben.
style = mpl.patches.ArrowStyle.Simple(head_length=10, head_width=5)
for p1, kraft in zip(punkte, F_ext):
    p2 = p1 + scal_kraft * kraft
    pfeil = mpl.patches.FancyArrowPatch(p1, p2, color='red',
                                        arrowstyle=style,
                                        zorder=2)
    ax.add_patch(pfeil)
    annot = ax.annotate(f'{np.linalg.norm(kraft):.1f} N',
                        (p1 + p2) / 2, color='red')
    annot.draggable(True)

# Zeichne die inneren Kräfte mit blauen Pfeilen in das Diagramm.
for i_stab, stab in enumerate(staebe):
    for i_punkt in stab:
        p1 = punkte[i_punkt]
        p2 = p1 + scal_kraft * F[i_stab] * ev(i_punkt, i_stab)
        pfeil = mpl.patches.FancyArrowPatch(p1, p2, color='blue',
                                            arrowstyle=style,
                                            zorder=2)
        ax.add_patch(pfeil)

# Zeige die Grafik an.
plt.show()
