"""Dynamik eines elastischen Stabwerks."""

import numpy as np
import scipy.integrate
import stabwerke


class StabwerkElastischDynamisch(stabwerke.StabwerkElastisch):
    """Dynamik eines Stabwerks mit elastischen Stäben.

    Die Stäbe können gedehnt und gestaucht werden. Sie verbiegen
    sich allerdings nicht. Die Dehnung der Stäbe wird mit dem
    hookeschen Gesetz berechnet und es wird eine Reibungskraft
    angesetzt, die proportional zur Geschwindigkeit der Punkte
    des Stabwerks ist. Das Anfangswertproblem des Stabwerks
    lässt sich mit der Methode `solve` lösen.

    Args:
        punkte (np.ndarray):
            Ortsvektoren der Punkte (n_punkte × n_dim).
        stuetz (list[int]):
            Indizes der Punkte, bei denen es sich um Stützpunkte
            handelt.
        staebe (np.ndarray):
            Jede Zeile enthält die Indizes der miteinander
            verbundenen Punkte (n_staebe × 2).
        reibung (np.ndarray):
            Reibungskoeffizienten der Punkte [kg/s] (n_punkte).
        **kwargs:
            Weitere Schlüsselwortargumente für `StabwerkElastisch`.
    """

    def __init__(self, punkte, stuetz, staebe, reibung=None,
                 **kwargs):
        super().__init__(punkte, stuetz, staebe, **kwargs)

        self.reibung = reibung
        """np.ndarray: Reibungskoeffizienten [kg / s] (n_punkte)."""

        # Setze die Reibungskoeffizienten auf null, wenn keine
        # Reibung angegeben wurde.
        if self.reibung is None:
            self.reibung = np.zeros(self.n_punkte)

        # Setze die Reibungskoeffizienten für alle Punkte gleich,
        # falls eine skalare Größe angegeben wurde.
        if np.isscalar(reibung):
            self.reibung = reibung * np.ones(self.n_punkte)

    def _dgl(self, t, u):
        """Berechne die rechte Seite der Differentialgleichung."""
        r, v = np.split(u, 2)
        r = r.reshape(self.n_knoten, self.n_dim)
        v = v.reshape(self.n_knoten, self.n_dim)

        # Verschiebe den mittleren Punkt an die angegebene Position.
        self.punkte[self.indizes_knoten] = r

        # Berechne die Kräfte.
        kraefte = self.gesamtkraefte()[self.indizes_knoten]
        kraefte -= (self.reibung[self.indizes_knoten].
                    reshape(-1, 1) * v)

        # Berechne die Beschleunigung des mittleren Punkts.
        m = self.punktmassen[self.indizes_knoten].reshape(-1, 1)
        a = kraefte / m

        return np.concatenate([v.reshape(-1), a.reshape(-1)])

    def solve(self, t_max, r0=None, v0=None, **kwargs):
        """Bestimme die zeitabhängige Dynamik des Stabwerks.

        Die Dynamik des Stabwerks wird in Form eines
        Anfangswertproblems mithilfe der Funktion
        `scipy.integrate.solve_ivp` bestimmt.

        Args:
            t_max (float):
                Simulationsdauer [s].
            r0 (np.ndarray):
                Ortsvektoren der Punkte [m] (n_punkte × n_dim).
                Der Vorgabewert entspricht der aktuellen
                Konfiguration des Stabwerks.
            v0 (np.ndarray):
                Anfangsgeschwindigkeiten [m/s] (n_punkte × n_dim).
                Der Vorgabewert entspricht einer
                Anfangsgeschwindigkeit von null.
            **kwargs:
                Schlüsselwortargumente für die Funktion
                `scipy.integrate.solve_ivp`.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Zeitpunkte, an denen die Lösung berechnet wurde.
                - Ortsvektoren der Punkte des Stabwerks
                  (n_zeitpunkte × n_punkte × n_dim)
                - Geschwindigkeitsvektoren der Punkte des Stabwerks
                  (n_zeitpunkte × n_punkte × n_dim)
        """
        # Setzte die aktuelle Konfiguration des Stabwerks für die
        # Anfangspositionen ein, falls diese nicht angegeben sind.
        if r0 is None:
            r0 = self.punkte.copy()

        # Setze die Anfangsgeschwindigkeiten auf null, wenn diese
        # nicht angegeben sind.
        if v0 is None:
            v0 = np.zeros((self.n_punkte, self.n_dim))

        # Lege den Anfangszustand fest.
        u0 = np.concatenate((r0[self.indizes_knoten].reshape(-1),
                             v0[self.indizes_knoten].reshape(-1)))

        # Sichere die aktuelle Konfiguration.
        punkte = self.punkte
        self.punkte = self.punkte.copy()

        # Löse die Differentialgleichungen numerisch.
        result = scipy.integrate.solve_ivp(self._dgl,
                                           [0, t_max], u0,
                                           **kwargs)
        t = result.t
        r, v = np.split(result.y, 2)

        # Schreibe die ursprüngliche Konfiguration zurück.
        self.punkte = punkte

        # Mache den ersten Index in den Lösungsarrays zum Zeitindex.
        # und wandle dies in ein n_zeitpunkte × n_knoten × n_dim -
        # Array um.
        r = r.T.reshape(-1, self.n_knoten, self.n_dim)
        v = v.T.reshape(-1, self.n_knoten, self.n_dim)

        # Erzeuge Arrays für die Positionen und Geschwindigkeiten
        # aller Punkte (Knoten und Stützpunkte)
        punkte = np.empty((t.size, self.n_punkte, self.n_dim))
        punkte[:] = self.punkte
        punkte[:, self.indizes_knoten, :] = r
        geschw = np.zeros((t.size, self.n_punkte, self.n_dim))
        geschw[:, self.indizes_knoten, :] = v

        return t, punkte, geschw
