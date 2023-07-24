"""Behandlung elastischer Stöße von Teilchen in einem Kasten."""

import numpy as np


class Mehrteilchenstoss:
    """Eine Menge von Teilchen in einem konvexen Polyeder.

    Die Teilchen bestehen aus Kreisen (2D) bzw. Kugeln (3D) mit
    gegebenem Radius und gegebener Masse. Jedes Teilchen hat
    einen vorgegebenen Ort und eine vorgegebene
    Anfangsgeschwindigkeit. Die Teilchen bewegen sich zwischen
    den Stößen geradlinig-gleichförmig. Die Stöße zwischen den
    Teilchen und zwischen Teilchen und Wänden erfolgen vollkommen
    elastisch. Es wird kein Drehimpuls auf die einzelnen Teilchen
    übertragen.

    Die Begrenzung des Simulationsgebiets wird durch nach außen
    zeigende Flächennormalen und den jeweils zugehörigen Abstand
    vom Koordinatenursprung angegeben.

    Mithilfe der Methode `zeitschritt` werden die Teilchen für
    eine angegebene Zeitdauer weiter bewegt und der interne
    Zustand des Mehrteilchenstosses entsprechend aktualisiert.

    Args:
        r (np.ndarray):
            Ortsvektoren der Teilchen [m] (n_teilchen × n_dim).
        v (np.ndarray):
            Teilchengeschwindigkeiten [m/s] (n_teilchen × n_dim).
        radien (np.ndarray):
            Radien der Teilchen [m] (n_teilchen × n_dim).
        massen (np.ndarray):
            Massen der Teilchen [m] (n_teilchen).
            stuetz (list[int]):
            Indizes der Punkte, bei denen es sich um Stützpunkte
            handelt.
        waende (tuple[np.array, np.array]):
            Der erste Eintrag des Tupels enthält ein Array
            der Abstände der Wände vom Koordinatenursprung.
            Der zweite Eintrag enthält ein Array (n_waende × n_dim)
            der nach außen zeigenden Normalenvektoren.
    """

    def __init__(self, r, v=None, radien=None, massen=None,
                 waende=None):
        self.r = np.array(r)
        """np.ndarray: Ortsvektoren (n_teilchen × n_dim)."""
        self.v = v
        """np.ndarray: Geschwindigkeiten (n_teilchen × n_dim)."""
        self.radien = radien
        """np.ndarray: Radien der Teilchen [m] (n_teilchen)."""
        self.massen = massen
        """np.ndarray: Massen der Teilchen [kg] (n_teilchen)."""
        self.delta_t_min = 1e-9
        """float: Zeitdifferenz ab der Stöße gleichzeitig sind."""

        # Setze die Geschwindigkeiten auf null, diese nicht
        # angegeben wurden.
        if self.v is None:
            self.v = np.zeros((self.n_teilchen, self.n_dim))
        else:
            self.v = np.array(self.v)

        # Setze alle Radien auf 1 m, wenn diese nicht angegeben
        # wurden.
        if self.radien is None:
            self.radien = np.ones(self.n_teilchen)
        if np.isscalar(radien):
            self.radien = radien * np.ones(self.n_teilchen)

        # Setze alle Massen auf 1 kg, wenn diese nicht angegeben
        # wurden.
        if self.massen is None:
            self.massen = np.ones(self.n_teilchen)
        if np.isscalar(massen):
            self.massen = massen * np.ones(self.n_teilchen)

        # Erzeuge leere Arrays für die Wände, falls diese nicht
        # angegeben wurden.
        if waende is None:
            self.wandabstaende = np.zeros(0)
            self.wandnormalen = np.zeros((0, self.n_dim))
        else:
            self.wandabstaende = np.array(waende[0])
            self.wandnormalen = np.array(waende[1])

        # Da die Berechnung der Stoßzeitpunkte und -partner sehr
        # aufwendig ist, speichern wir das Ergebnis in den
        # folgenden privaten Attributen. Ein Wert von
        # `_t_naechster_stoss=None` zeigt an, dass eine
        # Neuberechnung notwendig ist.
        self._t_naechster_stoss = None
        """float: Zeit bis zum nächsten Stoßereignis oder None."""

        self._stosspartner_teilchen = []
        """list: Partner des nächsten Teilchen-Teilchen-Stoßes.

        Eine Liste der zum Zeitpunkt `self._t_naechster_stoss`
        kollidierenden Teilchen-Teilchen-Kollisionspartner. Jeder
        Listeneintrag enthält zwei Teilchenindizes.
        """

        self._stosspartner_wand = []
        """list: Partner des nächsten Teilchen-Wand-Stoßes.

        Eine Liste der zum Zeitpunkt `self._t_naechster_stoss`
        kollidierenden Teilchen-Wand-Kollisionspartner. Jeder
        Listeneintrag enthält den Teilchenindex und den
        entsprechenden Wandindex.
        """

    @property
    def n_teilchen(self):
        """int: Anzahl der Teilchen."""
        return self.r.shape[0]

    @property
    def n_dim(self):
        """int: Anzahl der Raumdimensionen (2 oder 3)."""
        return self.r.shape[1]

    def _bestimme_naechsten_stoss(self):
        """Bestimme die nächste stattfindende Teilchenkollision.

        Das Ergebnis wird in privaten Attributen
        zwischengespeichert und erst nach einem Aufruf der
        Methode `stoss_teilchen` oder `stoss_wand` neu berechnet.
        """
        if self._t_naechster_stoss is not None:
            return

        # Lösche die vorhandenen Listen mit Kollisionspartnern.
        self._stosspartner_wand.clear()
        self._stosspartner_teilchen.clear()

        # Berechne die nächsten Stöße.
        dt_teil, stosspartner_teilchen = self._koll_teilchen()
        dt_wand, stosspartner_wand = self._koll_wand()
        dt = min(dt_teil, dt_wand)

        # Wähle aus, welche Stöße als nächstes ausgeführt werden.
        if abs(dt_teil - dt) < self.delta_t_min:
            self._stosspartner_teilchen = stosspartner_teilchen
        if abs(dt_wand - dt) < self.delta_t_min:
            self._stosspartner_wand = stosspartner_wand
        self._t_naechster_stoss = dt

    def _koll_teilchen(self):
        """Bestimme die nächste stattfindende Teilchenkollision.

        Returns:
            tuple[float, list[tuple[int, int]]]:
                - Die Zeitdauer bis zur nächsten Kollision oder inf,
                  falls keine Teilchen mehr kollidieren.
                - Eine Liste der zugehörigen Kollisionspartner.
                  Jeder Listeneintrag enthält zwei Teilchenindizes.
        """
        # Erstelle n_teilchen × n_teilchen × n_dim-Arrays, die die
        # paarweisen Orts- und Geschwindigkeitsdifferenzen
        # enthalten:
        #            dr[i, j] ist der Vektor r[i] - r[j]
        #            dv[i, j] ist der Vektor v[i] - v[j]
        dr = self.r.reshape(self.n_teilchen, 1, self.n_dim) - self.r
        dv = self.v.reshape(self.n_teilchen, 1, self.n_dim) - self.v

        # Erstelle ein n_teilchen × n_teilchen-Array, das das
        # Betragsquadrat der Vektoren aus dem Array dv enthält.
        dv_quadrat = np.sum(dv * dv, axis=2)

        # Erstelle ein n_teilchen × n_teilchen-Array, das die
        # paarweisen Summen der Radien der Teilchen enthält.
        radiensummen = (self.radien
                        + self.radien.reshape(self.n_teilchen, 1))

        # Um den Zeitpunkt der Kollision zu bestimmen, muss eine
        # quadratische Gleichung der Form
        #          t² + 2 a t + b = 0
        # gelöst werden. Nur die kleinere Lösung ist relevant.
        a = np.sum(dv * dr, axis=2) / dv_quadrat
        b = (np.sum(dr * dr,
                    axis=2) - radiensummen ** 2) / dv_quadrat
        D = a ** 2 - b
        t = -a - np.sqrt(D)

        # Suche den kleinsten positiven Zeitpunkt einer Kollision.
        t[t <= 0] = np.nan
        t_min = np.nanmin(t)

        # Suche die entsprechenden Teilchenindizes heraus.
        teilchen1, teilchen2 = np.where(
            np.abs(t - t_min) < self.delta_t_min)

        # Bilde eine Liste mit Tupeln der Kollisionspartner.
        partner = list(zip(teilchen1, teilchen2))

        # Entferne die doppelten Teilchenpaarungen.
        partner = partner[:len(partner) // 2]

        # Setze den Zeitpunkt auf inf, wenn keine Kollision
        # stattfindet.
        if np.isnan(t_min):
            t_min = np.inf

        # Gib den Zeitpunkt und die Teilchenindizes zurück.
        return t_min, partner

    def _koll_wand(self):
        """Bestimme die nächste stattfindende Wandkollision.

        Returns:
            tuple[float, list[tuple[int, int]]]:
                - Die Zeitdauer bis zur nächsten Kollision oder inf,
                  falls keine Kollisionen mehr stattfinden.
                - Eine Liste der zugehörigen Kollisionspartner.
                  Jeder Listeneintrag enthält den Teilchenindex und
                  den entsprechenden Wandindex.
        """
        # Berechne die Zeitpunkte der Kollisionen der Teilchen mit
        # einer der Wände.
        # Das Ergebnis ist ein n_teilchen × Wandanzahl-Array.
        z = (self.wandabstaende - self.radien.reshape(-1, 1)
             - self.r @ self.wandnormalen.T)
        t = z / (self.v @ self.wandnormalen.T)

        # Ignoriere alle nichtpositiven Zeiten.
        t[t <= 0] = np.nan

        # Ignoriere alle Zeitpunkte, bei denen sich das Teilchen
        # entgegen dem Normalenvektor bewegt. Eigentlich dürfte
        # so etwas gar nicht vorkommen, aber aufgrund von
        # Rundungsfehlern kann es passieren, dass ein Teilchen
        # sich leicht außerhalb einer Wand befindet.
        t[(self.v @ self.wandnormalen.T) < 0] = np.nan

        # Suche den kleinsten Zeitpunkt, der noch übrig bleibt.
        t_min = np.nanmin(t)

        # Setze den Zeitpunkt auf inf, wenn keine Kollision
        # stattfindet.
        if np.isnan(t_min):
            t_min = np.inf

        # Bilde eine Liste mit Tupeln der Kollisionspartner.
        teilchen, wand = np.where(
            np.abs(t - t_min) < self.delta_t_min)
        partner = list(zip(teilchen, wand))

        # Gib den Zeitpunkt und die Indizes der Partner zurück.
        return t_min, partner

    def _bewege_teilchen_ohne_stoss(self, dt):
        """Bewege die Teilchen mit der aktuellen Geschwindigkeit.

        Die Teilchen werden für die angegebene Zeit weiter bewegt,
        maximal jedoch bis zum nächsten Stoßereignis.

        Args:
            dt (float):
                Angefragte Simulationszeit.

        Returns:
            float: Tatsächliche Simulationszeit.
        """
        dt = min(dt, self._t_naechster_stoss)
        self.r += self.v * dt
        self._t_naechster_stoss -= dt
        return dt

    def _stoss_teilchen(self, i, j):
        """Lasse Teilchen i mit Teilchen j kollidieren.

        Achtung: Diese Methode überprüft nicht, ob sich die
        Teilchen zum betrachteten Zeitpunkt überhaupt berühren.
        """
        m1 = self.massen[i]
        m2 = self.massen[j]
        r1 = self.r[i]
        r2 = self.r[j]
        v1 = self.v[i]
        v2 = self.v[j]

        # Berechne die Schwerpunktsgeschwindigkeit.
        v_schwerpunkt = (m1 * v1 + m2 * v2) / (m1 + m2)

        # Berechne die Richtung, in der der Stoß stattfindet.
        richtung = (r1 - r2) / np.linalg.norm(r1 - r2)

        # Berechne die neuen Geschwindigkeiten nach dem Stoß.
        v1_neu = v1 + 2 * (v_schwerpunkt - v1) @ richtung * richtung
        v2_neu = v2 + 2 * (v_schwerpunkt - v2) @ richtung * richtung
        self.v[i] = v1_neu
        self.v[j] = v2_neu

        # Erzwinge eine Neuberechnung des nächsten Stoßvorgangs.
        self._t_naechster_stoss = None

    def _stoss_wand(self, i, j):
        """Lasse Teilchen i mit Wand j kollidieren.

        Achtung: Diese Methode überprüft nicht, ob sich die
        Teilchen zum betrachteten Zeitpunkt überhaupt berühren.
        """
        # Berechne die neue Geschwindigkeit.
        normale = self.wandnormalen[j]
        self.v[i] = self.v[i] - 2 * self.v[i] @ normale * normale

        # Erzwinge eine Neuberechnung des nächsten Stoßvorgangs.
        self._t_naechster_stoss = None

    def zeitschritt(self, dt=None):
        """Bewege alle Teilchen über die angegebene Zeit weiter.

        Falls keine Zeit angegeben wurde, dann werden die Teilchen
        bis zum nächsten Kollisionsereignis weiter bewegt.

        Args:
            dt (float):
                Simulationszeit.

        Returns:
            float: Zeitdauer, die tatsächlich simuliert wurde.
        """
        # Berechne ggf. die Zeitdauer bis zur ersten Kollision
        # und die beteiligten Partner.
        self._bestimme_naechsten_stoss()

        # Wenn keine Zeitdauer angegeben ist, dann rechnen wir bis
        # zur nächsten Kollision.
        if dt is None:
            dt = self._t_naechster_stoss

        # Zeit, die bisher bereits simuliert wurde.
        t = 0

        # Behandle nacheinander alle Kollisionen innerhalb des
        # angegebenen Zeitintervalls.
        while t < dt:
            # Bewege die Teilchen bis zum nächsten Stoß oder bis
            # zum Ende des Zeitintervalls.
            t += self._bewege_teilchen_ohne_stoss(dt - t)

            # Führe die Stöße gegebenenfalls aus.
            if abs(self._t_naechster_stoss) < self.delta_t_min:

                # Lass die Teilchen untereinander kollidieren.
                for teilch1, teilch2 in self._stosspartner_teilchen:
                    self._stoss_teilchen(teilch1, teilch2)

                # Lass die Teilchen mit Wänden kollidieren.
                for teilchen, wand in self._stosspartner_wand:
                    self._stoss_wand(teilchen, wand)

                # Berechne die nächsten Stöße.
                self._bestimme_naechsten_stoss()

        # Gib die simulierte Zeitdauer zurück.
        return t
