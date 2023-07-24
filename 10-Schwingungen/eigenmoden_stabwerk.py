﻿"""Eigenmoden und Eigenfrequenzen eines elastischen Stabwerks."""

import numpy as np
import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt

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

# Massen der Knotenpunkte [kg].
knotenmassen = np.array([5.0, 5.0, 1.0])

# Amplitude, mit der die Eigenmoden dargestellt werden [m].
amplitude = 0.3

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


# Stelle das Gleichungssystem für die Kräfte auf.
A = np.zeros((n_knoten, n_dim, n_knoten, n_dim))
for n, k in enumerate(indizes_knoten):
    for m, j in enumerate(indizes_knoten):
        for i in range(n_staebe):
            A[n, :, m, :] -= (steifigkeiten[i] / laenge(i)
                              * np.outer(ev(k, i), ev(j, i)))
A = A.reshape((n_knoten * n_dim, n_knoten * n_dim))

# Erzeuge ein Array, das die Masse für jede Koordinate
# der Knotenpunkte enthält.
massen = np.repeat(knotenmassen, n_dim)

# Berechne die Matrix Lambda.
Lambda = -A / massen.reshape(-1, 1)

# Bestimme die Eigenwerte und die Eigenvektoren.
eigenwerte, eigenvektoren = np.linalg.eig(Lambda)

# Eigentlich sollten alle Eigenwerte reell sein.
if np.any(np.iscomplex(eigenwerte)):
    print('Achtung: Einige Eigenwerte sind komplex.')
    print('Der Imaginärteil wird ignoriert')
    eigenwerte = np.real(eigenwerte)
    eigenvektoren = np.real(eigenvektoren)

# Eigentlich sollte es keine negativen Eigenwerte geben.
eigenwerte[eigenwerte < 0] = 0

# Sortiere die Eigenmoden nach aufsteigender Frequenz.
indizes_sortiere_eigenwerte = np.argsort(eigenwerte)
eigenwerte = eigenwerte[indizes_sortiere_eigenwerte]
eigenvektoren = eigenvektoren[:, indizes_sortiere_eigenwerte]

# Berechne die Eigenfrequenzen.
eigenfrequenzen = np.sqrt(eigenwerte) / (2 * np.pi)

# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Anzahl der darzustellenden Eigenmoden.
n_moden = eigenfrequenzen.size

# Erzeuge ein geeignetes n_zeilen × n_spalten - Raster.
n_zeilen = int(np.sqrt(n_moden))
n_spalten = n_moden // n_zeilen
while n_zeilen * n_spalten < n_moden:
    n_spalten += 1

# Erzeuge eine Liste der animierten Grafikobjekte jeder Eigenmode.
plots = []

# Erstelle die Plots für jede Eigenmode in einer eigenen Axes.
for mode in range(n_moden):
    ax = fig.add_subplot(n_zeilen, n_spalten, mode + 1)
    ax.set_title(f'$f_{{{mode+1}}}$={eigenfrequenzen[mode]:.1f} Hz')
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Erzeuge ein Dictionary für die animierten Plot-Objekte
    # dieser Mode und hänge dieses an die Liste plots an.
    plot_objekte = {}
    plots.append(plot_objekte)

    # Lege einen Plot für die Knotenpunkte in Blau an.
    plot_objekte['knoten'], = ax.plot([], [], 'bo', zorder=5)

    # Lege Plots für die Stäbe an.
    plot_objekte['staebe'] = []
    for stab in staebe:
        plot_stab, = ax.plot([], [], color='black', zorder=4)
        plot_objekte['staebe'].append(plot_stab)

    # Plotte die Stützpunkte in Rot.
    ax.plot(punkte[indizes_stuetz, 0], punkte[indizes_stuetz, 1],
            'ro', zorder=5)

    # Plotte die Ausgangslage der Knotenpunkte hellblau.
    ax.plot(punkte[indizes_knoten, 0], punkte[indizes_knoten, 1],
            'o', color='lightblue', zorder=2)

    # Plotte die Ausgangslage der Stäbe Hellgrau.
    for stab in staebe:
        ax.plot(punkte[stab, 0], punkte[stab, 1],
                color='lightgray', zorder=1)

# Zeitachse, die 60 Punkte im Bereich von 0 ... 2 pi enthält.
t = np.radians(np.arange(0, 360, 6))


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    for mode in range(n_moden):

        # Stelle den zu dieser Mode gehörenden Eigenvektor
        # als ein n_knoten × dim - Array dar.
        ev = eigenvektoren[:, mode].reshape(n_knoten, n_dim)

        # Berechne die aktuellen Positionen p aller Punkte.
        p = punkte.copy()
        p[indizes_knoten] += amplitude * np.sin(t[n]) * ev

        # Aktualisiere die Positionen der Knotenpunkte.
        plots[mode]['knoten'].set_data(p[indizes_knoten].T)

        # Aktualisiere die Koordinaten der Stäbe.
        for linie, stab in zip(plots[mode]['staebe'], staebe):
            linie.set_data(p[stab, 0], p[stab, 1])

    # Gib eine Liste aller geänderten Objekte zurück.
    geaendert = []
    for p in plots:
        geaendert.append(p['knoten'])
        geaendert += p['staebe']
    return geaendert


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
