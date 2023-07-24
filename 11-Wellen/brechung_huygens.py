"""Brechung und Reflexion nach dem huygensschen Prinzip."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Ausbreitungsgeschwindigkeit in den beiden Medien [m/s].
c1 = 1.0
c2 = 0.3
# Einfallswinkel [rad].
alpha = np.radians(50)
# Anzahl der Strahlen.
n_strahlen = 7
# Breite des einfallenden Strahlenbündes [m].
breite = 10.0
# Anfangsabstand des Mittelstrahls von der Grenzfläche [m].
abstand = 21.0
# Dargestellter Bereich in x- und y-Richtung: -xy_max bis +xy_max.
xy_max = 15.0
# Zeitschrittweite [s].
dt = 0.05

# Austrittswinkel der gebrochenen Strahlen nach Snellius [rad].
beta = np.arcsin(np.sin(alpha) * c2 / c1)

# Bestimme die normierten Richtungsvektoren der einfallenden,
# der reflektierten und der gebrochenen Strahlen.
e_ein = np.array([np.sin(alpha), -np.cos(alpha)])
e_ref = np.array([np.sin(alpha), np.cos(alpha)])
e_geb = np.array([np.sin(beta), -np.cos(beta)])

# Bestimme einen Einheitsvektor, der senkrecht auf dem einfallenden
# Strahl steht.
e_ein_senkrecht = np.array([np.cos(alpha), np.sin(alpha)])

# Berechne die Startpunkte der einzelnen Strahlen. Die Strahlen
# sollen parallel einlaufen und gleichmäßig über die Strahlbreite
# verteilt sein.
b = np.linspace(-breite / 2, breite / 2, n_strahlen)
startorte = b.reshape(-1, 1) * e_ein_senkrecht - abstand * e_ein

# Berechne die Auftreffzeitpunkte der einzelnen Strahlen. Diese
# ergeben sich aus der Bedingung y = 0.
auftreffzeiten = -startorte[:, 1] / (c1 * e_ein[1])

# Berechne die Auftrefforte der Strahlen auf der Grenzfläche.
auftrefforte = (startorte +
                c1 * auftreffzeiten.reshape(-1, 1) * e_ein)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-xy_max, xy_max)
ax.set_ylim(-xy_max, xy_max)
ax.set_aspect('equal')

# Erzeuge leere Listen für die einfallenden, reflektierten und
# transmittierten Strahlen sowie für die Kreisbögen der
# Elementarwellen.
strahlen_ein = []
strahlen_ref = []
strahlen_geb = []
kreise_ref = []
kreise_geb = []

# Erzeuge die entsprechenden Grafikelemente und füge sie den
# Listen hinzu.
for i in range(n_strahlen):
    # Erzeuge einen einfallenden Lichtstrahl.
    strahl_ref, = ax.plot([], [])
    strahlen_ein.append(strahl_ref)

    # Erzeuge den reflektierten und transmittierten Lichtstrahl mit
    # der jeweils gleichen Farbe, wie der einfallende Lichtstrahl.
    farbe = strahl_ref.get_color()
    strahlen_ref.extend(ax.plot([], [], ':', color=farbe))
    strahlen_geb.extend(ax.plot([], [], '--', color=farbe))

    # Erzeuge die Kreisbögen für die Elementarwellen der Reflexion.
    kreis_ref = mpl.patches.Arc(auftrefforte[i], width=1, height=1,
                                theta1=0, theta2=180,
                                visible=False,
                                fill=False, color=farbe)
    ax.add_patch(kreis_ref)
    kreise_ref.append(kreis_ref)

    # Erzeuge die Kreisbögen für die Elementarwellen der Brechung.
    kreis_geb = mpl.patches.Arc(auftrefforte[i], width=1, height=1,
                                theta1=180, theta2=360,
                                visible=False,
                                fill=False, color=farbe)
    ax.add_patch(kreis_geb)
    kreise_geb.append(kreis_geb)

# Färbe die obere Hälfte des Koordinatensystems hellgrau ein.
hintergrund_oben = mpl.patches.Rectangle((-xy_max, 0),
                                         2 * xy_max, xy_max,
                                         color='0.9', zorder=0)
ax.add_patch(hintergrund_oben)

# Färbe die untere Hälfte des Koordinatensystems etwas dunkler ein.
hintergrund_unten = mpl.patches.Rectangle((-xy_max, -xy_max),
                                          2 * xy_max, xy_max,
                                          color='0.8', zorder=0)
ax.add_patch(hintergrund_unten)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Berechne den aktuellen Zeitpunkt.
    t = dt * n

    # Aktualisiere die einfallenden Strahlen.
    for strahl, r, t0 in zip(strahlen_ein,
                             startorte, auftreffzeiten):
        punkte = np.array([r, r + c1 * min(t, t0) * e_ein])
        strahl.set_data(punkte.T)

    # Aktualisiere die reflektierten Strahlen.
    for strahl, r, t0 in zip(strahlen_ref,
                             auftrefforte, auftreffzeiten):
        punkte = np.array([r, r + c1 * max(0, (t - t0)) * e_ref])
        strahl.set_data(punkte.T)

    # Aktualisiere die gebrochenen Strahlen.
    for strahl, r, t0 in zip(strahlen_geb,
                             auftrefforte, auftreffzeiten):
        punkte = np.array([r, r + c2 * max(0, (t - t0)) * e_geb])
        strahl.set_data(punkte.T)

    # Aktualisiere die Kreise in der oberen Halbebene.
    for kreis, t0 in zip(kreise_ref, auftreffzeiten):
        if t > t0:
            kreis.width = kreis.height = 2 * (t - t0) * c1
        kreis.set_visible(t > t0)

    # Aktualisiere die Kreise in der unteren Halbebene.
    for kreis, t0 in zip(kreise_geb, auftreffzeiten):
        if t > t0:
            kreis.width = kreis.height = 2 * (t - t0) * c2
        kreis.set_visible(t > t0)

    return (strahlen_ein + strahlen_ref + strahlen_geb +
            kreise_ref + kreise_geb)


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
