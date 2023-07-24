"""Darstellung einer Schraubenfeder (2-dimensional)."""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.lines


class Schraubenfeder(mpl.lines.Line2D):
    """Grafische Darstellung einer Schraubenfeder in 2D.

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
        **kwargs:
            Schlüsselwortargumente für `mpl.lines.Line2D`.
    """

    def __init__(self, startpunkt, endpunkt,
                 n_wdg=5, l0=0, a=0.1, r0=0.2, n_punkte=300,
                 **kwargs):
        super().__init__([], [], **kwargs)
        self._start = np.array(startpunkt)
        self._ende = np.array(endpunkt)

        self.n_wdg = n_wdg
        """int: Anzahl der Windungen."""
        self.l0 = l0
        """float: Ruhelänge der Feder."""
        self.a = a
        """float: Länge der Geraden am Anfang und Ende der Feder."""
        self.r0 = r0
        """float: Radius der Feder bei der Ruhelänge."""
        self.n_punkte = n_punkte
        """int: Anzahl der Punkte."""

        self.aktualisiere_plot()

    @property
    def startpunkt(self):
        """np.ndarray: Startpunkt der Feder."""
        return self._start.copy()

    @startpunkt.setter
    def startpunkt(self, p):
        self._start = np.array(p)
        self.aktualisiere_plot()

    @property
    def endpunkt(self):
        """np.ndarray: Endpunkt der Feder."""
        return self._ende.copy()

    @endpunkt.setter
    def endpunkt(self, p):
        self._ende = np.array(p)
        self.aktualisiere_plot()

    def aktualisiere_plot(self):
        """Berechne die Daten für die Darstellung der Feder."""
        # Gesamtlänge der Feder.
        laenge = np.linalg.norm(self._ende - self._start)

        # Passe für l0 > 0 den Radius der Feder so an, dass die
        # Drahtlänge unverändert bleibt.
        if self.l0 > 0:
            d_quadrat = (self.l0 - 2 * self.a) ** 2 + (
                        2 * np.pi * self.n_wdg * self.r0) ** 2
            if d_quadrat > (laenge - 2 * self.a) ** 2:
                umfang = math.sqrt(d_quadrat
                                   - (laenge - 2 * self.a) ** 2)
                radius = umfang / (2 * np.pi * self.n_wdg)
            else:
                radius = 0
        else:
            radius = self.r0

        # Array für das Ergebnis.
        punkte = np.empty((self.n_punkte, 2))

        # Setze den Anfangs- und Endpunkt.
        punkte[0] = self._start
        punkte[-1] = self._ende

        # Einheitsvektor in Richtung der Verbindungslinie.
        richtung_feder = (self._ende - self._start) / laenge

        # Einheitsvektor senkrecht zur Verbindungslinie.
        richtung_quer = np.array([richtung_feder[1],
                                  -richtung_feder[0]])

        # Parameter entlang der Feder.
        s = np.linspace(0, 1, self.n_punkte - 2)
        s = s.reshape(-1, 1)

        # Berechne die restlichen Punkte als sinusförmige Linie.
        x = self.a + s * (laenge - 2 * self.a)
        y = radius * np.sin(self.n_wdg * 2 * np.pi * s)
        punkte[1:-1] = (self._start
                        + x * richtung_feder + y * richtung_quer)

        self.set_data(punkte.T)


# Der folgende Code wird nur aufgerufen, wenn dieses Modul
# direkt in Python gestartet wird. Wenn das Modul von einem
# anderen Python-Modul importiert wird, dann wird dieser Code
# nicht ausgeführt.
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Erzeuge eine Figure und ein Axes-Objekt.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    ax.grid()

    # Stelle die Schraubenfeder dar.
    plot = Schraubenfeder([-1, 0], [0.5, 3.0], l0=2, n_wdg=8)
    ax.add_line(plot)
    ax.autoscale()
    plt.show()
