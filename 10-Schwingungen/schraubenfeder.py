"""Darstellung einer Schraubenfeder (2-dimensional)."""

import math
import numpy as np
import matplotlib.pyplot as plt


def data(start, ende, n_wdg=5, l0=0, a=0.1, r0=0.2, n_punkte=300):
    """Berechne die Daten für die Darstellung einer Schraubenfeder.

    Args:
        start (np.ndarray):
            Koordinaten des Anfangspunktes.
        ende (np.ndarray):
            Koordinaten des Endpunktes.
        n_wdg (int):
            Anzahl der Windungen.
        l0 (float):
            Ruhelänge der Feder inkl. der geraden Verbindungsstücke.
            Bei positiver Ruhelänge wird der Radius der Feder
            automatisch so angepasst, dass die Drahtlänge der
            Feder konstant bleibt.
        a (float):
            Länge der Geraden am Anfang und Ende der Feder.
        r0 (float):
            Radius der Feder bei der Ruhelänge.
        n_punkte (int):
            Anzahl der zu berechnenden Punkte.

    Returns:
        np.ndarray: Datenpunkte (2 × n_punkte).
    """
    # Gesamtlänge der Feder.
    laenge = np.linalg.norm(ende - start)

    # Passe für l0 > 0 den Radius der Feder so an, dass die
    # Drahtlänge unverändert bleibt.
    if l0 > 0:
        d_quad = (l0 - 2 * a) ** 2 + (2 * np.pi * n_wdg * r0) ** 2
        if d_quad > (laenge - 2 * a) ** 2:
            umfang = math.sqrt(d_quad - (laenge - 2 * a) ** 2)
            radius = umfang / (2 * np.pi * n_wdg)
        else:
            radius = 0
    else:
        radius = r0

    # Array für das Ergebnis.
    punkte = np.empty((n_punkte, 2))

    # Setze den Anfangs- und Endpunkt.
    punkte[0] = start
    punkte[-1] = ende

    # Einheitsvektor in Richtung der Verbindungslinie.
    richtung_feder = (ende - start) / laenge

    # Einheitsvektor senkrecht zur Verbindungslinie.
    richtung_quer = np.array([richtung_feder[1],
                              -richtung_feder[0]])

    # Parameter entlang der Feder.
    s = np.linspace(0, 1, n_punkte - 2)
    s = s.reshape(-1, 1)

    # Berechne die restlichen Punkte als sinusförmige Linie.
    x = a + s * (laenge - 2 * a)
    y = radius * np.sin(n_wdg * 2 * np.pi * s)
    punkte[1:-1] = (start + x * richtung_feder + y * richtung_quer)

    return punkte.T


# Der folgende Code wird nur aufgerufen, wenn dieses Modul
# direkt in Python gestartet wird. Wenn das Modul von einem
# anderen Python-Modul importiert wird, dann wird dieser Code
# nicht ausgeführt.
if __name__ == '__main__':

    # Erzeuge die Daten für die Schraubenfeder.
    startpunkt = np.array([-1.0, 0.0])
    endpunkt = np.array([0.5, 3.0])
    dat = data(startpunkt, endpunkt, l0=2, n_wdg=8)

    # Erzeuge eine Figure und ein Axes-Objekt.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    # Stelle die Schraubenfeder mit einem Linienplot dar.
    plot, = ax.plot(dat[0], dat[1], 'k-', linewidth=2)
    plt.show()
