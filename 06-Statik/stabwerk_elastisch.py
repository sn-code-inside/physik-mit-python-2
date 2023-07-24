"""Verformung eines elastischen Stabwerks.

Die Kraftverteilung in einem 2-dimensionalen elastischen Stabwerk
sowie die Verformung des Stabwerks werden durch Lösen eines
nichtlinearen Gleichungssystems berechnet.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize

# Lege den Skalierungsfaktor für die Kraftvektoren fest [m/N].
scal_kraft = 0.001

# Lege die Positionen der Punkte fest [m].
punkte = np.array([[0, 0], [1.2, 0], [1.2, 2.1], [0, 2.1],
                   [0.6, 1.05]])

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
indizes_stuetz = [0, 1]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[1, 2], [2, 3], [3, 0],
                   [0, 4], [1, 4], [3, 4], [2, 4]])

# Produkt aus E-Modul und Querschnittsfläche der Stäbe [N].
steifigkeiten = np.array([5.6e6, 5.6e6, 5.6e6,
                          7.1e3, 7.1e3, 7.1e3, 7.1e3])

# Lege die äußere Kraft fest, die auf jeden Punkt wirkt [N].
# Für die Stützpunkte setzen wir diese Kraft zunächst auf 0.
# Diese wird später berechnet.
F_ext = np.array([[0, 0], [0, 0], [200.0, 0], [0, 0], [0, 0]])

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


def stabkraft(i_stb, koord):
    """Berechne die Kraft in einem Stab.

    Args:
        i_stb (int):
            Index des betrachteten Stabes.
        koord (np.ndarray):
            Koordinaten der Punkte (n_punkte × n_dim).

    Returns:
        float: Wert der Kraft in diesem Stab [N].
    """
    l0 = laenge(i_stb)
    return steifigkeiten[i_stb] * (laenge(i_stb, koord) - l0) / l0


def gesamtkraft(koord):
    """Berechne die Gesamtkräfte auf alle Punkte.

    Args:
        koord (np.ndarray):
            Koordinaten der Punkte (n_punkte × n_dim).

    Returns:
        np.ndarray: Kraftvektoren [N] (n_punkte × n_dim).
    """
    # Initialisiere das Array mit den äußeren Kräften.
    F_ges = F_ext.copy()

    # Addiere für jeden Stab die Stabkraft zu der Gesamtkraft
    # der angrenzenden Punkte.
    for i_stb, stb in enumerate(staebe):
        for i_pkt in stb:
            F_ges[i_pkt] += (stabkraft(i_stb, koord)
                             * ev(i_pkt, i_stb, koord))
    return F_ges


def funktion_opti(x):
    """Gib die Kräfte auf die Knotenpunkte als 1D-Array zurück.

    Args:
        x (np.ndarray):
            1D-Array der Koordinaten der Knotenpunkte.

    Returns:
        np.ndarray: 1D-Array der Kräfte auf die Knotenpunkte.
    """
    # Erzeuge ein Array, das die Koordinaten der Stützpunkte und
    # die aktuellen Koordinaten der Knoten enthält.
    p = punkte.copy()
    p[indizes_knoten] = x.reshape(n_knoten, n_dim)

    # Berechne die Gesamtkraft für jeden einzelnen Punkt.
    F_ges = gesamtkraft(p)

    # Wähle die Knotenkräfte aus und gib das Ergebnis als
    # 1-dimensionales Array zurück.
    F_knoten = F_ges[indizes_knoten]
    return F_knoten.reshape(-1)


# Suche eine Lösung der Gleichung func(x) = 0. Als
# Startpositionen geben wir die Anfangspositionen der Knoten vor.
result = scipy.optimize.root(funktion_opti, punkte[indizes_knoten])
print(result.message)
print(f'Die Funktion wurde {result.nfev}-mal ausgewertet.')

# Erzeuge ein Array mit den berechneten Positionen der Punkte.
punkte_neu = punkte.copy()
punkte_neu[indizes_knoten] = result.x.reshape(n_knoten, n_dim)

# Berechne die Kraft in jedem der Stäbe.
F = np.zeros(n_staebe)
for i_stab in range(n_staebe):
    F[i_stab] = stabkraft(i_stab, punkte_neu)

# Berechne die äußeren Kräfte auf die Stützpunkte.
F_ext[indizes_stuetz] = -gesamtkraft(punkte_neu)[indizes_stuetz]

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-0.3, 1.8)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')
ax.grid()

# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
ax.plot(punkte_neu[indizes_knoten, 0],
        punkte_neu[indizes_knoten, 1], 'bo')
ax.plot(punkte_neu[indizes_stuetz, 0],
        punkte_neu[indizes_stuetz, 1], 'ro')

# Plotte die Stäbe und beschrifte diese mit dem Wert der Kraft.
for stab, kraft in zip(staebe, F):
    ax.plot(punkte_neu[stab, 0], punkte_neu[stab, 1],
            color='black')
    # Erzeuge ein Textfeld, das den Wert der Kraft angibt. Das
    # Textfeld wird am Mittelpunkt des Stabes platziert.
    position = np.mean(punkte_neu[stab], axis=0)
    annot = ax.annotate(f'{kraft:+.1f} N', position, color='blue')
    annot.draggable(True)

# Zeichne die äußeren Kräfte mit roten Pfeilen in das Diagramm
# ein und erzeuge Textfelder, die den Betrag der Kraft angeben.
style = mpl.patches.ArrowStyle.Simple(head_length=10, head_width=5)
for p1, kraft in zip(punkte_neu, F_ext):
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
        p1 = punkte_neu[i_punkt]
        p2 = p1 + scal_kraft * F[i_stab] * ev(i_punkt, i_stab,
                                              punkte_neu)
        pfeil = mpl.patches.FancyArrowPatch(p1, p2, color='blue',
                                            arrowstyle=style,
                                            zorder=2)
        ax.add_patch(pfeil)

# Zeige die Grafik an.
plt.show()
