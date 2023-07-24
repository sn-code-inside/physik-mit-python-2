"""Kraftverteilung in einem 3-dimensionalen starren Stabwerk."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# Lege den Skalierungsfaktor für die Kraftvektoren fest [m/N].
scal_kraft = 0.005

# Lege die Positionen der Punkte fest [m].
punkte = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0],
                   [0, 0, 2], [1, 0, 2], [1, 1, 2],
                   [0, 1, 2]])

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
indizes_stuetz = [0, 1, 2]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[0, 3], [1, 4], [2, 6],
                   [3, 4], [4, 5], [5, 6], [6, 3],
                   [3, 5], [0, 4],
                   [0, 6], [0, 5], [1, 5]])

# Lege die äußere Kraft fest, die auf jeden Punkt wirkt [N].
# Für die Stützpunkte setzen wir diese Kraft zunächst auf 0.
# Diese wird später berechnet.
F_ext = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                  [0, 0, -98.1], [0, 0, -98.1], [0, 0, -98.1],
                  [0, 0, -98.1]])

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
ax = fig.add_subplot(1, 1, 1, projection='3d',
                     elev=40, azim=45)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_zlabel('$z$ [m]')
ax.set_xlim(-0.5, 2.0)
ax.set_ylim(-0.5, 2.0)
ax.set_zlim(0, 3.0)
ax.grid()


class Arrow3D(mpl.patches.FancyArrowPatch):
    """Darstellung eines Pfeiles in einer 3D-Grafik.

    Args:
        posA (tuple):
            Koordinaten (x, y, z) des Startpunktes.
        posB (tuple):
            Koordinaten (x, y, z) des Endpunktes.
        *args:
            Weitere Argumente für mpl.patches.FancyArrowPatch
        **kwargs:
            Weitere Schlüsselwortargumente für
            mpl.patches.FancyArrowPatch
    """

    def __init__(self, posA, posB, *args, **kwargs):
        super().__init__(posA[0:2], posB[0:2], *args, **kwargs)
        self._pos = np.array([posA, posB])

    def set_positions(self, posA, posB):
        """Setze den Start- und Endpunkt des Pfeils."""
        self._pos = np.array([posA, posB])

    def do_3d_projection(self, renderer=None):
        """Projiziere die Punkte in die Bildebene."""
        p = mpl_toolkits.mplot3d.proj3d.proj_points(self._pos,
                                                    self.axes.M)
        super().set_positions(*p[:, 0:2])
        return np.min(p[:, 2])


class Annotation3D(mpl.text.Annotation):
    """Darstellung einer Annotation in einer 3D-Grafik.

    Args:
        s (str):
            Dazustellender Text.
        pos (tuple):
            Koordinaten (x, y, z) der Annotation.
        *args:
            Weitere Argumente für mpl.text.Annotation
        **kwargs:
            Weitere Schlüsselwortargumente für mpl.text.Annotation
    """

    def __init__(self, s, pos, *args, **kwargs):
        super().__init__(s, xy=(0, 0), *args,
                         xytext=(0, 0),
                         textcoords='offset points',
                         **kwargs)
        self._pos = np.array(pos)

    def draw(self, renderer):
        """Zeichne die Annotation."""
        p = mpl_toolkits.mplot3d.proj3d.proj_transform(
            *self._pos, self.axes.M)
        self.xy = p[0:2]
        super().draw(renderer)


# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
ax.plot(punkte[indizes_knoten, 0],
        punkte[indizes_knoten, 1],
        punkte[indizes_knoten, 2], 'bo')
ax.plot(punkte[indizes_knoten, 0],
        punkte[indizes_knoten, 1],
        punkte[indizes_knoten, 2], 'ro')

# Plotte die Stäbe und beschrifte diese mit dem Wert der Kraft.
for stab, kraft in zip(staebe, F):
    ax.plot(punkte[stab, 0],
            punkte[stab, 1],
            punkte[stab, 2], color='black')
    # Erzeuge ein Textfeld, das den Wert der Kraft angibt. Das
    # Textfeld wird am Mittelpunkt des Stabes platziert.
    pos = np.mean(punkte[stab], axis=0)
    annot = Annotation3D(f'{kraft:+.1f} N', pos, color='blue')
    ax.add_artist(annot)

# Zeichne die äußeren Kräfte mit roten Pfeilen in das Diagramm
# ein und erzeuge Textfelder, die den Betrag der Kraft angeben.
style = mpl.patches.ArrowStyle.Simple(head_length=10,
                                      head_width=5)
for p1, kraft in zip(punkte, F_ext):
    p2 = p1 + scal_kraft * kraft
    pfeil = Arrow3D(p1, p2, color='red',
                    arrowstyle=style, zorder=2)
    ax.add_patch(pfeil)
    annot = Annotation3D(f'{np.linalg.norm(kraft):.1f} N',
                         p1, color='red')
    ax.add_artist(annot)

# Zeichne die inneren Kräfte mit blauen Pfeilen in das Diagramm.
for i_stab, stab in enumerate(staebe):
    for i_punkt in stab:
        p1 = punkte[i_punkt]
        p2 = p1 + scal_kraft * F[i_stab] * ev(i_punkt, i_stab)
        pfeil = Arrow3D(p1, p2, color='blue',
                        arrowstyle=style, zorder=2)
        ax.add_patch(pfeil)

# Zeige die Grafik an.
plt.show()
