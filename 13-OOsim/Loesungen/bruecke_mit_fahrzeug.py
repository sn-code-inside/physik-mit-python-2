"""Verformung einer Brückenkonstruktion bei Belastung.

Die Kraftverteilung in einem 2-dimensionalen elastischen Stabwerk
sowie die Verformung des Stabwerks werden in der linearen
Näherung für kleine Deformationen berechnet.

Die Darstellung der Kräfte erfolgt hier durch eine Farbtabelle.
Die Brücke wird durch ein fahrendes Fahrzeug zusätzlich
belastet.
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm
import stabwerke

# Elastizitätsmodul [N/m²].
E_modul = 210e9
# Querschnittsfläche der Stäbe [m²].
flaeche = 5e-2 ** 2
# Dichte des Stabmaterials [kg/m³].
rho = 7860.0
# Masse des Fahrzeugs [kg].
masse_fahrzeug = 3000.0
# Geschwindigkeit des Fahrzeugs [m/s].
v_fahrzeug = 0.03
# Zeitdauer zwischen zwei Bildern.
dt = 0.5
# Lege die maximale Kraft für die Farbtabelle fest [N].
F_max = 70000


# Definiere die Geometrie des Stabwerks.
punkte = np.array([[0, 0], [4, 0], [8, 0],
                   [12, 0], [16, 0], [20, 0],
                   [2, 2], [6, 2], [10, 2],
                   [14, 2], [18, 2]], dtype=float)
indizes_stuetz = [0, 5]
staebe = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                   [6, 7], [7, 8], [8, 9], [9, 10],
                   [0, 6], [1, 7], [2, 8], [3, 9], [4, 10],
                   [6, 1], [7, 2], [8, 3], [9, 4], [10, 5]])

# Länge eines Segments der Fahrbahn [m].
segmentlaenge = 4.0

# Gesamtlänge der Brücke [m].
gesaemtlaenge = 20.0

# Berechne die Steifigkeiten der Stäbe.
steifigkeiten = np.ones(len(staebe)) * E_modul * flaeche

# Erzeuge das Stabwerk.
sw = stabwerke.StabwerkElastischLin(punkte, indizes_stuetz, staebe,
                                    steifigkeiten=steifigkeiten)

# Lege die Massen der einzelnen Punkte fest, indem die
# Masse der Stäbe berechnet und diese gleichmäßig auf die jeweils
# benachbarten Punkte verteilt wird.
sw.punktmassen = np.zeros(len(punkte))
for i, stab in enumerate(staebe):
    sw.punktmassen[stab] += sw.stablaengen()[i] * flaeche * rho / 2

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1, 21)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.grid()

# Plotte das Stabwerk und das Fahrzeug.
plot_stabwerk = stabwerke.PlotStabwerk(ax, sw, cmap='jet',
                                       linewidth_stab=3,
                                       annot=False, arrows=False)
for a in plot_stabwerk.artists:
    a.set_visible(False)
plot_fahrzeug, = ax.plot([], [], 'go', zorder=200)

plot_stabwerk.mapper.set_array([-F_max, F_max])
plot_stabwerk.mapper.autoscale()


# Erzeuge einen Farbbalken am Rand des Bildes. Wir wollen die
# Kräfte in kN angeben. Dazu erzeugen wir ein Objekt vom Typ
# mpl.ticker.FuncFormatter. Diesem übergeben wir eine Funktion,
# die die Zahlenwerte in N in kN formatiert ausgibt.
def format_kraft(x, pos=None):
    return f'{x/1e3:.1f}'
fmt = mpl.ticker.FuncFormatter(format_kraft)
fig.colorbar(plot_stabwerk.mapper, format=fmt, label='Kraft [kN]',
             fraction=0.12, pad=0.2, orientation='horizontal')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Lege die aktuelle Position des Autos fest und ermittle
    # einen Index, bei dem Gewichtskraft des Autos
    # mit dem Anteil 'anteil' berücksichtigt werden
    # muss. Der Rest der Gewichtskraft muss beim nächsten
    # Index berücksichtigt werden.
    x_position = v_fahrzeug * dt * n
    anteil, index = math.modf(x_position / segmentlaenge)
    index = int(index)

    # Berücksichtige die Gewichtskraft des Autos
    g = np.linalg.norm(sw.g_vector)
    gewichtskraft = np.zeros((sw.n_punkte, sw.n_dim))
    gewichtskraft[index, 1] -= masse_fahrzeug * g * (1-anteil)
    gewichtskraft[index+1, 1] -= masse_fahrzeug * g * anteil
    sw.kraefte_ext = gewichtskraft

    # Aktualisiere das Stabwerk.
    sw.suche_gleichgewichtsposition()
    plot_stabwerk.update_stabwerk()
    for a in plot_stabwerk.artists:
        a.set_visible(True)

    # Aktualisiere die Position des Fahrzeugs.
    plot_fahrzeug.set_data(x_position, 0)

    return plot_stabwerk.artists + [plot_fahrzeug]


# Berechne die Anzahl von Frames, in denen das Fahrzeug
# die Brücke komplett überquert.
n_frames = int(gesaemtlaenge / (v_fahrzeug * dt))

# Erzeuge das Animationsobjekt.
ani = mpl.animation.FuncAnimation(fig, update,
                                  frames=n_frames,
                                  interval=30, blit=False)

# Starte die Animation.
plt.show()
