"""Berechnung der Kräfte und/oder Verformungen in Stabwerken."""

import copy
import numpy as np
import scipy.optimize


class Stabwerk:
    """Ein allgemeines Stabwerk.

    Ein Stabwerk besteht aus Punkten, die durch Stäbe miteinander
    verbunden sind. Jeder Stab verbindet genau zwei Punkte.

    Die Punkte unterteilen sich in Knotenpunkte, deren Position
    allein durch die Stabverbindungen festgelegt sind, und
    Stützpunkte, die durch äußere Zwangsbedingungen fixiert sind.

    Auf jeden Punkt können externe Kräfte und die Gewichtskraft
    wirken. Die Gewichtskraft ergibt sich aus den Massen der
    Punkte und dem Schwerebeschleunigungsvektor `g_vect`, der in
    der Standardeinstellung für ein 2-dimensionales Stabwerk in
    die negative y-Richtung und für ein 3-dimensionales Stabwerk
    in die negative z-Richtung zeigt.

    Args:
        punkte (np.ndarray):
            Ortsvektoren der Punkte (n_punkte × n_dim).
        stuetz (list[int]):
            Indizes der Punkte, bei denen es sich um Stützpunkte
            handelt.
        staebe (np.ndarray):
            Jede Zeile enthält die Indizes der miteinander
            verbundenen Punkte (n_staebe × 2).
        punktmassen (np.ndarray):
            Massen der Punkte (n_punkte).
        kraefte_ext (np.ndarray):
            Äußere Kräfte [N], die auf die Punkte wirken
            (n_punkte × n_dim).
    """

    def __init__(self, punkte, stuetz, staebe,
                 punktmassen=None, kraefte_ext=None):
        self.punkte = np.array(punkte)
        """np.ndarray: Koordinaten der Punkte (n_punkte × n_dim)."""
        self.staebe = np.array(staebe)
        """np.ndarray: Punktindizes der Stäbe (n_staebe × 2)."""
        self.indizes_stuetz = list(stuetz)
        """list[int]: Indizes der Stützpunkte (n_stuetz)."""
        self.kraefte_ext = kraefte_ext
        """np.ndarray: Äußere Kräfte [N] (n_punkte × n_dim)."""
        self.punktmassen = punktmassen
        """np.ndarray: Massen der Punkte [kg] (n_punkte)."""
        self.rtol = 1e-6
        """float: Relative Genauigkeit des Gleichgewichts."""
        self.atol = 1e-12
        """float: Absolute Genauigkeit des Gleichgewichts [N]."""

        # Setze alle externen Kräfte auf null, wenn keine
        # äußeren Kräfte angegeben wurden.
        if self.kraefte_ext is None:
            self.kraefte_ext = np.zeros((self.n_punkte, self.n_dim))

        # Setze alle Massen auf null, wenn keine Massen angegeben
        # wurden.
        if punktmassen is None:
            self.punktmassen = np.zeros(self.n_punkte)

        # Lege den Vektor des Gravitationsfeldes fest.
        self.g_vector = np.zeros(self.n_dim)
        """np.ndarray: Vektor der Schwerebeschleunigung [m/s²]."""

        # Der Vektor zeigt im -y-Richtung im 2D-Fall und in
        # -z-Richtung im 3D-Fall.
        self.g_vector[-1] = -9.81

    @property
    def n_punkte(self):
        """int: Gesamtanzahl der Punkte."""
        return self.punkte.shape[0]

    @property
    def n_dim(self):
        """int: Anzahl der Raumdimensionen (2 oder 3)."""
        return self.punkte.shape[1]

    @property
    def n_staebe(self):
        """int: Anzahl der Stäbe."""
        return len(self.staebe)

    @property
    def n_stuetz(self):
        """int: Anzahl der Stützpunkte."""
        return len(self.indizes_stuetz)

    @property
    def n_knoten(self):
        """int: Anzahl der Knotenpunkte."""
        return self.n_punkte - self.n_stuetz

    @property
    def indizes_knoten(self):
        """list[int]: Indizes der Knotenpunkte des Stabwerks."""
        menge_punkte = set(range(len(self.punkte)))
        menge_stuetzpunkte = set(self.indizes_stuetz)
        return list(menge_punkte - menge_stuetzpunkte)

    def einheitsvektor(self, i_punkt, i_stab):
        """Bestimme den Einheitsvektor zu dem Punkt in Stabrichtung.

        Der Einheitsvektor ist so orientiert, dass er immer vom
        angegebenen Punkt in Richtung des angegebenen Stabes zeigt.

        Args:
            i_punkt (int):
                Index des Punktes.
            i_stab (int):
                Index des Stabes.

        Returns:
            np.ndarray: Einheitsvektor oder der Nullvektor, wenn
                        der Stab den Punkt nicht enthält (n_dim).
        """
        stab = self.staebe[i_stab]
        if i_punkt not in stab:
            return np.zeros(self.n_dim)
        if i_punkt == stab[0]:
            vec = self.punkte[stab[1]] - self.punkte[i_punkt]
        else:
            vec = self.punkte[stab[0]] - self.punkte[i_punkt]
        return vec / np.linalg.norm(vec)

    def stabkraefte_scal(self):
        """Berechne die Stabkräfte.

        Eine negative Stabkraft entspricht einer Druckspannung:
        Der Stab versucht die beiden Massenpunkte auseinander zu
        drücken.

        Eine positive Stabkraft entspricht einer Zugspannung:
        Der Stab versucht die beiden Massenpunkte zueinander zu
        ziehen.

        Returns:
            np.ndarray: Die Stabkräfte [N] (n_staebe).
        """
        raise NotImplementedError()

    def stabkraft(self, i_punkt, i_stab):
        """Bestimme die Kraft eines Stabs auf einen Punkt.

        Args:
            i_punkt (int):
                Index des betrachteten Punktes.
            i_stab (int):
                Index des betrachteten Stabes.

        Returns:
            np.ndarray: Berechneter Kraftvektor [N] (n_dim).
        """
        einheitsvektor = self.einheitsvektor(i_punkt, i_stab)
        return self.stabkraefte_scal()[i_stab] * einheitsvektor

    def stabkraefte(self):
        """Bestimme die Summe der Stabkräfte auf die Punkte.

        Returns:
            np.ndarray: Kraftvektoren [N] (n_punkte × n_dim).
        """
        kraefte = np.zeros((self.n_punkte, self.n_dim))
        for i_stab, stab in enumerate(self.staebe):
            for i_punkt in stab:
                kraefte[i_punkt] += self.stabkraft(i_punkt, i_stab)
        return kraefte

    def gewichtskraefte(self):
        """Bestimme die Summe der Gewichtskräfte auf die Punkte.

        Returns:
            np.ndarray: Vektor der Gewichtskraft [N] (n_dim).
        """
        return self.punktmassen.reshape(-1, 1) * self.g_vector

    def gesamtkraefte_ohne_stuetzkraefte(self):
        """Bestimme die Gesamtkraft ohne die Stützkräfte.

        Returns:
            np.ndarray: Kraftvektoren [N] (n_punkte × n_dim).
        """
        return (self.kraefte_ext + self.gewichtskraefte()
                + self.stabkraefte())

    def stuetzkraefte(self):
        """Bestimme die Stützkräfte auf die Stützpunkte.

        Returns:
            np.ndarray: Kraftvektoren [N] (n_stuetz × n_dim).
        """
        kraefte = self.gesamtkraefte_ohne_stuetzkraefte()
        return -kraefte[self.indizes_stuetz]

    def gesamtkraefte(self):
        """Bestimme die Gesamtkräfte auf die Punkte.

        Returns:
            np.ndarray: Kraftvektoren [N] (n_punkte × n_dim).
        """
        kraefte = self.gesamtkraefte_ohne_stuetzkraefte()
        kraefte[self.indizes_stuetz] += self.stuetzkraefte()
        return kraefte

    def stablaengen(self):
        """Bestimme die aktuellen Längen der Stäbe.

        Returns:
            np.ndarray: Stablängen [m] (n_staebe).
        """
        laengen = np.empty(self.n_staebe)
        for i, stab in enumerate(self.staebe):
            laengen[i] = np.linalg.norm(self.punkte[stab[0]]
                                        - self.punkte[stab[1]])
        return laengen

    def ist_im_gleichgewicht(self):
        """Überprüfe, ob das System im Gleichgewicht ist.

        Returns:
            bool: True, wenn das System innerhalb der Toleranzen
                  im Gleichgewicht ist. False, sonst.
        """
        kraft_stab = np.max(np.abs(self.stabkraefte_scal()))
        kraft_ext = np.max(np.linalg.norm(self.kraefte_ext, axis=1))
        kraft_gew = np.max(np.linalg.norm(self.gewichtskraefte(),
                                          axis=1))
        kraft_max = max(kraft_stab, kraft_ext, kraft_gew)
        delta_kraft = self.rtol * kraft_max + self.atol

        kraefte = self.gesamtkraefte_ohne_stuetzkraefte()
        kraefte = kraefte[self.indizes_knoten]
        kraefte = np.linalg.norm(kraefte, axis=1)
        return np.all(kraefte < delta_kraft)


class StabwerkStarr(Stabwerk):
    """Ein starres Stabwerk mit unendlich steifen Stäben."""

    def _systemmatrix_starr(self):
        """Bestimme die Systemmatrix A.

        Die Systemmatrix A ist so festgelegt, dass die
        Matrixmultiplikation der Matrix A mit dem Vektor der
        skalaren Stabkräfte einen Vektor aller Kraftkomponenten
        auf die Knotenpunkte ergibt:

        A * Stabkräfte = Kraftkomponenten der Stäbe auf die Knoten.

        Der Eintrag A_ij gibt also an, mit welchem Faktor die
        Stabkraft j in die Kraftkomponente i eingeht.

        Returns:
            np.ndarray:
                Systemmatrix (n_knoten · n_dim × n_staebe)
        """
        A = np.zeros((self.n_knoten, self.n_dim, self.n_staebe))
        for n, k in enumerate(self.indizes_knoten):
            for i in range(self.n_staebe):
                A[n, :, i] = self.einheitsvektor(k, i)
        A = A.reshape(self.n_knoten * self.n_dim, self.n_staebe)
        return A

    def stabkraefte_scal(self):
        # Berechne die Stabkräfte durch Lösen des linearen
        # Gleichungssystems
        #    A * Stabkräfte + externe Kräfte + Gewichtskräfte = 0,
        # wobei A die Systemmatrix ist und bei den Kräften nur
        # die Kräfte auf die Knotenpunkte berücksichtigt werden.
        a = self._systemmatrix_starr()
        b = self.kraefte_ext + self.gewichtskraefte()
        b = b[self.indizes_knoten].reshape(-1)
        return np.linalg.solve(a, -b)


class StabwerkElastisch(Stabwerk):
    """Ein Stabwerk mit elastischen Stäben.

    Die Stäbe können gedehnt und gestaucht werden. Sie verbiegen
    sich allerdings nicht. Die Dehnung der Stäbe wird mit dem
    hookeschen Gesetz berechnet.

    Args:
        punkte (np.ndarray):
            Ortsvektoren der Punkte (n_punkte × n_dim).
        stuetz (list[int]):
            Indizes der Punkte, bei denen es sich um Stützpunkte
            handelt.
        staebe (np.ndarray):
            Jede Zeile enthält die Indizes der miteinander
            verbundenen Punkte (n_staebe × 2).
        steifigkeiten (np.ndarray):
            Steifigkeiten der Stäbe [N] (n_staebe).
        **kwargs:
            Weitere Schlüsselwortargumente für `Stabwerk`.
    """

    def __init__(self, punkte, stuetz, staebe, steifigkeiten=None,
                 **kwargs):
        super().__init__(punkte, stuetz, staebe, **kwargs)

        self.steifigkeiten = steifigkeiten
        """np.ndarray: Steifigkeiten der Stäbe [N] (n_staebe)."""

        # Setze die Steifigkeiten auf einen Wert von 10^8 N,
        # wenn keine Steifigkeiten angegeben wurden.
        if self.steifigkeiten is None:
            self.steifigkeiten = 1e8 * np.ones(self.n_staebe)

        self.stablaengen0 = self.stablaengen()
        """np.ndarray: Die entspannten Stablängen [m] (n_staebe)."""

    def stabkraefte_scal(self):
        # Berechne die Stabkräfte über das hookesche Gesetz.
        ursprungslaengen = self.stablaengen0
        laengen = self.stablaengen()
        return self.steifigkeiten * (laengen / ursprungslaengen - 1)

    def _funktion_opti(self, x):
        """Gib die Kräfte auf die Knotenpunkte als 1D-Array zurück.

        Achtung: Diese Methode verändert das Array self.punkte.

        Args:
            x (np.ndarray):
                Komponenten der Ortsvektoren (n_knoten*n_dim).

        Returns:
            np.ndarray: Komponenten der Kräfte (n_knoten*n_dim).
        """
        self.punkte[self.indizes_knoten] = x.reshape(self.n_knoten,
                                                     self.n_dim)
        F_knoten = self.gesamtkraefte()[self.indizes_knoten]
        return F_knoten.reshape(-1)

    def suche_gleichgewichtsposition(self, **kwargs):
        """Bestimme das statische Gleichgewicht.

        Die Suche der Gleichgewichtsposition erfolgt mithilfe der
        Funktion `scipy.optimize.root`. Wenn die Optimierung
        konvergiert, dann werden die Knotenpunkte im Array
        `punkte` entsprechend neu gesetzt. Andernfalls bleiben
        diese unverändert.

        Args:
            **kwargs:
                Schlüsselwortargumente für die Funktion
                `scipy.optimize.root`.

        Returns:
            OptimizeResult: Ergebnis von `scipy.optimize.root`
        """
        punkte = self.punkte
        self.punkte = self.punkte.copy()
        x0 = self.punkte[self.indizes_knoten]
        result = scipy.optimize.root(self._funktion_opti, x0,
                                     **kwargs)
        self.punkte = punkte

        if result.success:
            knoten = result.x.reshape(self.n_knoten, self.n_dim)
            self.punkte[self.indizes_knoten] = knoten
        return result


class StabwerkElastischLin(StabwerkElastisch):
    """Ein Stabwerk mit elastischen Stäben in linearer Näherung."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Das Stabwerk `stabwerk_zuvor` wird nur zur Berechnung
        # der Stabkräfte verwendet. Die äußeren Kräfte und die
        # Gewichtskräfte müssen hier nicht mit übergeben werden.
        self.stabwerk_zuvor = StabwerkElastisch(self.punkte,
                                                self.indizes_stuetz,
                                                self.staebe,
                                                self.steifigkeiten)
        """Stabwerk: Zustand vor der letzten linearen Näherung."""

    @classmethod
    def from_stabwerk_elastisch(cls, orig):
        """Erzeuge eine linearisierte Version eines Stabwerks.

        Args:
            orig (StabwerkElastisch): Das Ursprungsstabwerk.

        Returns:
            StabwerkElastischLin:
                Die um die aktuellen Positionen linearisierte
                Variante des Stabwerks.
        """
        # Erzeuge ein neues Objekt vom Typ StabwerkElastischLin.
        neu = cls(orig.punkte, orig.indizes_stuetz, orig.staebe)

        # Übertrage alle Attribute auf das neue Stabwerk.
        for attribut, wert in vars(orig).items():
            setattr(neu, attribut, copy.copy(wert))

        # Setze den Ausgangszustand des Stabwerks vor der
        # Linearisierung auf den aktuellen Zustand.
        neu.stabwerk_zuvor = copy.copy(orig)

        return neu

    def stabkraefte_scal(self):
        # Berechne die Stabkräfte, die durch die aktuelle lineare
        # Näherung hinzugekommen sind, mithilfe des hookeschen
        # Gesetzes.
        delta_r = self.punkte - self.stabwerk_zuvor.punkte
        kraefte = self.stabwerk_zuvor.stabkraefte_scal()
        for i_stab, (j, k) in enumerate(self.staebe):
            S = self.steifigkeiten[i_stab]
            l0 = self.stablaengen0[i_stab]
            ev = self.stabwerk_zuvor.einheitsvektor(k, i_stab)
            kraefte[i_stab] += S/l0 * ev @ (delta_r[j] - delta_r[k])
        return kraefte

    def _systemmatrix(self):
        """Bestimme die Systemmatrix A.

        Das System wird um die aktuelle Lage der Knotenpositionen
        linearisiert. Die Systemmatrix A ist so festgelegt

        Die Systemmatrix A ist so festgelegt, dass die
        Matrixmultiplikation der Matrix A mit dem Vektor der
        Verschiebungen der Knotenpunkte die Kraftkomponenten der
        Stabkräfte auf die Knotenpunkte ergibt:

            A * Verschiebungsvektoren = Knotenkräfte.

        Returns:
            np.ndarray:
                Systemmatrix (n_knoten · n_dim × n_knoten · n_dim)
        """
        A = np.zeros((self.n_knoten, self.n_dim,
                      self.n_knoten, self.n_dim))
        S = self.steifigkeiten
        L = self.stablaengen()
        L0 = self.stablaengen0
        E = np.eye(self.n_dim)
        for n, k in enumerate(self.indizes_knoten):
            for m, j in enumerate(self.indizes_knoten):
                for i, stab in enumerate(self.staebe):
                    e_ki_e_ji = np.outer(self.einheitsvektor(k, i),
                                         self.einheitsvektor(j, i))
                    A[n, :, m, :] -= S[i] * (e_ki_e_ji / L[i])
                    if (k == j) and (k in stab):
                        A[n, :, m, :] -= S[i] * E * (1/L0[i]-1/L[i])
                    if (k != j) and (k in stab) and (j in stab):
                        A[n, :, m, :] += S[i] * E * (1/L0[i]-1/L[i])
        return A.reshape((self.n_knoten * self.n_dim,
                          self.n_knoten * self.n_dim))

    def suche_gleichgewichtsposition(self):
        """Bestimme das statische Gleichgewicht (linearisiert).

        Die Suche der Gleichgewichtsposition erfolgt über die
        Lösung des linearen Gleichungssystems

           A * Verschiebungen + aktuelle Gesamtkräfte = 0,

        wobei A die Systemmatrix ist und bei den Kräften nur
        die Kräfte auf die Knotenpunkte berücksichtigt werden. Die
        so ermittelten Verschiebungen werden zu den aktuellen
        Positionen der Knoten addiert.
        """
        # Speichere den aktuellen Zustand als neuen Ausgangszustand.
        self.stabwerk_zuvor.punkte = self.punkte.copy()

        # Löse das Gleichungssystem A @ dr = -b, wobei
        # b die aktuell vorhandenen Kräfte sind.
        A = self._systemmatrix()
        b = self.gesamtkraefte()
        b = b[self.indizes_knoten].reshape(-1)
        dr = np.linalg.solve(A, -b)
        dr = dr.reshape(self.n_knoten, self.n_dim)

        self.punkte[self.indizes_knoten] += dr

    def eigenmoden(self):
        """Bestimme die Eigenmoden des linearisierten Stabwerks.

        Returns:
            (np.ndarray, np.ndarray):
                - Eigenfrequenzen [Hz] (n_moden).
                - Eigenmoden (n_moden × n_knoten × n_dim).
        """
        # Überprüfe, ob das System wirklich im Gleichgewichtszustand
        # ist. Andernfalls ergibt die Berechnung der Eigenwerte
        # der Systemmatrix keinen Sinn.
        if not self.ist_im_gleichgewicht():
            raise ValueError('Eigenmoden können nur um einen '
                             'Gleichgewichtszustand bestimmt '
                             'werden. Das vorliegende System ist '
                             'nicht im statischen Gleichgewicht.')

        # Erzeuge ein Array, das die Masse für jede Koordinate
        # der Knotenpunkte enthält.
        massen = np.repeat(self.punktmassen[self.indizes_knoten],
                           self.n_dim)
        massen = massen.reshape(-1, 1)

        # Berechne die Matrix Lambda.
        Lambda = -self._systemmatrix() / massen

        # Bestimme die Eigenwerte und die Eigenvektoren.
        eigenwerte, eigenvektoren = np.linalg.eig(Lambda)

        # Eigentlich sollten alle Eigenwerte reell sein.
        if np.any(np.iscomplex(eigenwerte)):
            print('Achtung: Einige Eigenwerte sind komplex.')
            print('Der Imaginärteil wird ignoriert')
            eigenwerte = np.real(eigenwerte)
            eigenvektoren = np.real(eigenvektoren)

        # Eigentlich sollte es keine negativen Eigenwerte geben.
        eigenwerte[eigenwerte <= 0] = 0

        # Sortiere die Eigenmoden nach aufsteigender Frequenz.
        idx_sortiere_eigenwerte = np.argsort(eigenwerte)
        eigenwerte = eigenwerte[idx_sortiere_eigenwerte]
        eigenvektoren = eigenvektoren[:, idx_sortiere_eigenwerte]

        # Berechne die Eigenfrequenzen.
        eigenfrequenzen = np.sqrt(eigenwerte) / (2 * np.pi)

        # Der erste Index der Eigenmoden soll die Mode indizieren,
        # der zweite Index den Punkt und der dritte Index die
        # Koordinatenrichtung.
        eigenvektoren = eigenvektoren.T
        eigenvektoren = eigenvektoren.reshape(-1, self.n_knoten,
                                              self.n_dim)

        # Ergänze die Eigenvektoren mit Nullen für die Stützpunkte.
        eigenmoden = np.zeros((len(eigenfrequenzen),
                               self.n_punkte, self.n_dim))
        eigenmoden[:, self.indizes_knoten, :] = eigenvektoren

        return eigenfrequenzen, eigenmoden
