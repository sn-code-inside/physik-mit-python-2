"""Animation zur Entstehung des machschen Kegels."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.005
# Frequenz, mit der die Wellenzüge ausgesendet werden [Hz].
f_Q = 5.0
# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 1.0
# Lege die Startposition der Quelle und des Beobachters fest [m].
startort_Q = np.array([-2.0, 0.5])
startort_B = np.array([2.0, -0.5])
# Lege die Geschwindigkeit von Quelle und Beobachter fest [m/s].
v_Q = np.array([1.3, 0])
v_B = np.array([-0.5, 0])
# Dargestellter Ortsbereich [m].
plotbereich_x = (-2.0, 2.0)
plotbereich_y = (-1.0, 1.0)

# Berechne den Öffnungwinkel des machschen Kegels nach der
# gegebenen Gleichung.
phi = 2 * np.arcsin(c / np.linalg.norm(v_Q))

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(plotbereich_x)
ax.set_ylim(plotbereich_y)
ax.set_aspect('equal')

# Beschriftung.
ax.text(0.05, 0.95, f'$v_Q = {np.linalg.norm(v_Q) / c:.2f} c$',
        transform=ax.transAxes, color='black', size=12,
        horizontalalignment='left', verticalalignment='top')
ax.text(0.95, 0.95, f'$v_B = {np.linalg.norm(v_B) / c:.2f} c$',
        transform=ax.transAxes, color='blue', size=12,
        horizontalalignment='right', verticalalignment='top')

# Erzeuge zwei Kreise für Quelle und Beobachter.
kreis_quelle = mpl.patches.Circle((0, 0), radius=0.03,
                                  visible=False, color='black',
                                  fill=True, zorder=4)
ax.add_patch(kreis_quelle)
kreis_beobachter = mpl.patches.Circle((0, 0), radius=0.03,
                                      visible=False, color='blue',
                                      fill=True, zorder=4)
ax.add_patch(kreis_beobachter)

# Erzeuge einen Linienplot für den machschen Kegel. Dieser wird
# später in der update-Funktion durch drei Punkte dargestellt.
plot_kegel, = ax.plot([0], [0], 'k', linewidth=2, zorder=5)

# Lege eine Liste an, die die kreisförmigen Wellenzüge speichert.
kreise = []


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Berechne den aktuellen Zeitpunkt.
    t = dt * n

    # Berechne die aktuelle Position von Quelle und Beobachter.
    kreis_quelle.center = startort_Q + v_Q * t
    kreis_beobachter.center = startort_B + v_B * t
    kreis_quelle.set_visible(True)
    kreis_beobachter.set_visible(True)

    # Zeichne den Kegel. Die Spitze des Kegels soll sich an
    # der aktuellen Position des Senders befinden. Wir müssen
    # noch zwei weitere Punkte r1 und r2 konstruieren,
    # die sich in Bewegungsrichtung unter einem Winkel phi/2
    # befinden. Dazu erzeugen wir zunächst einen
    # Einheitsvektor e_v in Bewegungsrichtung und einen
    # Einheitsvektor e_s der senkrecht dazu steht.
    e_v = v_Q / np.linalg.norm(v_Q)
    e_s = np.array([e_v[1], -e_v[0]])

    # Konstruiere jeweils einen Richtungsvektor der einen Winkel
    # phi/2 mit der Bewegungsrichtung einschließt. Berechne mit
    # diesem Reichtungvektor jeweils einen zusätzlichen Punkt des
    # Kegels über eine Geradengleichung. Dabei wird die Länge der
    # Linien so gewählt, dass diese gerade bis zum ersten
    # ausgesendeten Wellenzug reichen.
    laenge = t * np.sqrt(v_Q @ v_Q - c ** 2)
    richtung1 = -np.cos(phi / 2) * e_v + np.sin(phi / 2) * e_s
    punkt1 = kreis_quelle.center + laenge * richtung1
    richtung2 = -np.cos(phi / 2) * e_v - np.sin(phi / 2) * e_s
    punkt2 = kreis_quelle.center + laenge * richtung2

    # Setze die drei Punkte zusammen und aktualisiere den Plot.
    punkte = np.stack([punkt1, kreis_quelle.center, punkt2])
    plot_kegel.set_data(punkte.T)

    # Erzeuge zum Startzeitpunkt einen neuen Kreis oder wenn
    # seit dem Aussenden des letzten Wellenzuges mehr als eine
    # Periodendauer vergangen ist.
    if not kreise or t >= kreise[-1].startzeit + 1 / f_Q:
        kreis = mpl.patches.Circle(kreis_quelle.center, radius=0,
                                   color='red', linewidth=1.5,
                                   fill=False, zorder=3)
        kreis.startzeit = t
        kreise.append(kreis)
        ax.add_patch(kreis)

    # Aktualisiere die Radien aller dargestellten Kreise.
    for kreis in kreise:
        kreis.radius = (t - kreis.startzeit) * c

    # Färbe den Beobachter rot, wenn ein Wellenzug auftrifft.
    kreis_beobachter.set_color('blue')
    for kreis in kreise:
        d = np.linalg.norm(kreis.center - kreis_beobachter.center)
        if abs(d - kreis.radius) < kreis_beobachter.radius:
            kreis_beobachter.set_color('red')

    # Färbe die Quelle rot, wenn ein Wellenzug ausgesendet wird.
    d = np.linalg.norm(kreise[-1].center - kreis_quelle.center)
    if abs(d - kreise[-1].radius) < kreis_quelle.radius:
        kreis_quelle.set_color('red')
    else:
        kreis_quelle.set_color('black')

    return kreise + [kreis_quelle, kreis_beobachter, plot_kegel]


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
