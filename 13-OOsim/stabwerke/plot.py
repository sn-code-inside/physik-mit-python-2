"""Grafische Darstellung von Stabwerken mit Matplotlib."""

import copy
import numpy as np
import matplotlib as mpl
import matplotlib.animation
import matplotlib.colors
import mpl_toolkits.mplot3d


class PlotStabwerk:
    """Darstellung eines Stabwerks (2D oder 3D) mit Matplotlib.

    Die Klasse erzeugt in einer angegebenen Axes grafische
    Elemente, die ein Stabwerk darstellen. Wenn das Stabwerk
    nachträglich verändert wird, können die Grafikelemente mit
    der Methode `update_stabwerk` aktualisiert werden. Dabei darf
    sich die Anzahl der Punkte nicht verändern und das Attribut
    `staebe` des Stabwerks muss unverändert bleiben.

    In der Voreinstellung wird die folgende Farbzuordnung verwendet:
        - Knotenpunkte: blau
        - Stützpunkte: rot
        - Stäbe: schwarz
        - Externe Kräfte: grün
        - Stabkräfte: blau
        - Gewichtskräfte: orange
        - Stützkräfte: rot

    Args:
        ax (mpl.axes.Axes):
            Axes, in die geplottet werden soll.
        stabwerk (Stabwerk):
            Das darzustellende Stabwerk.
        copy (bool):
            Soll eine tiefe Kopie des Stabwerks für die Darstellung
            verwendet werden?
        cmap (str):
            Colormap für die Darstellung der Stabkräfte.
            Bei cmap=None werden die Stäbe schwarz dargestellt.
        scal_kraft (float):
            Skalierungsfaktor für die Kraftvektoren [m/N].
        arrows_stab (bool):
            Sollen Pfeile für die Stabkräfte dargestellt werden?
        annot_stab (bool):
            Sollen die Stäbe beschriftet werden?
        arrows_ext (bool):
            Sollen Pfeile für die externen Kräfte
            dargestellt werden?
        annot_ext (bool):
            Sollen die Pfeile für die externen Kräfte
            beschriftet werden?
        arrows_grav (bool):
            Sollen Pfeile für die Gewichtskräfte dargestellt werden?
        annot_grav (bool):
            Sollen Pfeile für die Gewichtskräfte beschriftet werden?
        arrows_stuetz (bool):
            Sollen Pfeile für die Stützkräfte dargestellt werden?
        annot_stuetz (bool):
            Sollen Pfeile für die Stützkräfte beschriftet werden?
        arrows (bool):
            Schaltet alle Pfeile an oder aus. Dieses Argument
            überschreibt alle anderen arrows-Argumente.
        annot (bool):
            Schaltet alle Beschriftungen an oder aus. Dieses
            Argument überschreibt alle anderen annot-Argumente.
        linewidth_stab (float):
            Linienbreite für die Stäbe.
        pointsize (float):
            Größe der Punkte für die Knoten- und Stützpunkte.
    """

    def __init__(self, ax, stabwerk, kopie=False,
                 cmap=None,
                 scal_kraft=0.01,
                 arrows_stab=True, annot_stab=True,
                 arrows_ext=True, annot_ext=True,
                 arrows_grav=True, annot_grav=True,
                 arrows_stuetz=True, annot_stuetz=True,
                 arrows=None, annot=None,
                 linewidth_stab=None,
                 pointsize=None):
        self.sw = stabwerk
        """Stabwerk: Das dazustellende Stabwerk."""

        self.ax = ax
        """Axes in der das Stabwerk dargestellt wird."""

        if kopie:
            self.sw = copy.deepcopy(self.sw)

        # Verwende eine schwarze Farbtabelle, falls keine angegeben.
        cmap = cmap or mpl.colors.ListedColormap([0, 0, 0, 1])

        # Triff einige Fallunterscheidungen für 2D und 3D.
        if self.sw.n_dim == 2:
            Annotation = mpl.text.Annotation
            Arrow = mpl.patches.FancyArrowPatch
            empty_data = ([], [])
        else:
            Annotation = Annotation3D
            Arrow = Arrow3D
            empty_data = ([], [], [])

        # Erzeuge einen Tupel für einen Punkt (0, 0) bzw. (0, 0, 0).
        p0 = self.sw.n_dim * (0,)

        self.scal_kraft = scal_kraft
        """float: Skalierungsfaktor für die Kraftpfeile [m/N]."""

        self.format_string_kraft = ' .1f'
        """str: Format-String für die Angabe der Kraft."""

        self.knoten, = ax.plot(*empty_data, 'bo', zorder=131,
                               markersize=pointsize)
        """Punktplot für die Positionen der Knotenpunkte."""

        self.stuetzpunkte, = ax.plot(*empty_data, 'ro', zorder=131,
                                     markersize=pointsize)
        """Punktplot für die Positionen der Stützpunkte."""

        self.staebe = []
        """list: Linienplots der Stäbe."""

        self.annot_stab = []
        """list: Beschriftungen der Stabkräfte."""

        self.pfeile_stab = {}
        """dict: Ein Pfeil für jedes Tupel (i_punkt, i_stab)."""

        self.pfeile_extern = []
        """list: Pfeile für die externen Kräfte."""

        self.annot_extern = []
        """list: Beschriftungen der externen Kräfte."""

        self.pfeile_stuetz = []
        """list: Pfeile für die Stützkräfte."""

        self.annot_stuetz = []
        """list: Beschriftungen der Stützkräfte."""

        self.pfeile_grav = []
        """list: Pfeile für die Gewichtskräfte."""

        self.annot_grav = []
        """list: Beschriftungen der Gewichtskräfte."""

        self.mapper = mpl.cm.ScalarMappable(cmap=cmap)
        """mpl.cm.ScalarMappable: Colormapper für die Stäbe."""

        # Erzeuge Linienplots für die Stäbe.
        for i in range(stabwerk.n_staebe):
            plot, = ax.plot(*empty_data, zorder=130,
                            linewidth=linewidth_stab)
            self.staebe.append(plot)

        # Falls eins der Argument arrows oder annot angegeben
        # wurde, dann setze die Einzeloptionen dementsprechend.
        if arrows is not None:
            arrows_stab = arrows
            arrows_ext = arrows
            arrows_grav = arrows
            arrows_stuetz = arrows
        if annot is not None:
            annot_stab = annot
            annot_ext = annot
            annot_grav = annot
            annot_stuetz = annot

        # Erzeuge die einzelnen Pfeiltypen, falls angefordert.
        style = mpl.patches.ArrowStyle.Simple(head_length=10,
                                              head_width=5)
        if arrows_stab:
            for i_stab, stab in enumerate(self.sw.staebe):
                for i_punkt in stab:
                    pfeil = Arrow(p0, p0, color='blue',
                                  arrowstyle=style, zorder=120)
                    ax.add_patch(pfeil)
                    self.pfeile_stab[(i_punkt, i_stab)] = pfeil
        if arrows_ext:
            for i in range(self.sw.n_punkte):
                pfeil = Arrow(p0, p0, color='green',
                              arrowstyle=style, zorder=121)
                ax.add_patch(pfeil)
                self.pfeile_extern.append(pfeil)
        if arrows_grav:
            for i in range(self.sw.n_punkte):
                pfeil = Arrow(p0, p0, color='orange',
                              arrowstyle=style, zorder=122)
                ax.add_patch(pfeil)
                self.pfeile_grav.append(pfeil)
        if arrows_stuetz:
            for i in range(self.sw.n_stuetz):
                pfeil = Arrow(p0, p0, color='red',
                              arrowstyle=style, zorder=123)
                ax.add_patch(pfeil)
                self.pfeile_stuetz.append(pfeil)

        # Erzeuge die einzelnen Beschriftungen, falls angefordert.
        def erstelle_annot(anzahl, liste, color=None, zorder=None):
            p0 = self.sw.n_dim * (0,)
            for i in range(anzahl):
                annot = Annotation('', p0,
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   color=color, zorder=zorder)
                ax.add_artist(annot)
                annot.draggable(True)
                annot.set_clip_on(False)
                liste.append(annot)
        if annot_stab:
            erstelle_annot(self.sw.n_staebe, self.annot_stab,
                           color='black', zorder=140)
        if annot_ext:
            erstelle_annot(self.sw.n_punkte, self.annot_extern,
                           color='green', zorder=141)
        if annot_grav:
            erstelle_annot(self.sw.n_punkte, self.annot_grav,
                           color='orange', zorder=142)
        if annot_stuetz:
            erstelle_annot(self.sw.n_stuetz, self.annot_stuetz,
                           color='red', zorder=143)

        # Setze einen geeigneten Bereich für die Farbtabelle.
        maximalkraft = np.max(np.abs(self.sw.stabkraefte_scal()))
        self.mapper.set_array([-maximalkraft, maximalkraft])
        self.mapper.autoscale()

        # Aktualisiere die Darstellung mit den aktuellen Daten.
        self.update_stabwerk()

    @property
    def artists(self):
        """list: Liste aller Grafikelemente."""
        return (self.staebe + [self.knoten, self.stuetzpunkte]
                + list(self.pfeile_stab.values()) + self.annot_stab
                + self.pfeile_extern + self.annot_extern
                + self.pfeile_grav + self.annot_grav
                + self.pfeile_stuetz + self.annot_stuetz)

    def _format_kraft(self, kraft):
        """Formatiere eine Kraft als String."""
        if not np.isscalar(kraft):
            kraft = np.linalg.norm(kraft)
        if kraft == 0:
            return ''
        return f'{kraft:{self.format_string_kraft}} N'

    def _farbe_kraft(self, kraft):
        """Bestimme eine Farbe für die angegebene Kraft.

        Returns:
            tuple[int, int, int, int]: Farbe in RGBA-Darstellung.
        """
        if not np.isscalar(kraft):
            kraft = np.linalg.norm(kraft)
        return self.mapper.to_rgba(kraft)

    def update_stabwerk(self):
        """Aktualisiere die Darstellung des Stabwerks."""
        sw = self.sw
        scal_kraft = self.scal_kraft

        # Abhängig davon, ob es sich um einen 2D- oder einen
        # 3D-Plot handelt, muss bei den Linien- und Punktplots die
        # Methode set_data oder set_data_3D aufgerufen werden.
        def set_data(obj, data):
            if self.sw.n_dim == 2:
                obj.set_data(data)
            else:
                obj.set_data_3d(data)

        # Berechne die Stabkräfte vorab, da die Berechnung unter
        # Umständen aufwendig ist.
        stabkraefte = sw.stabkraefte_scal()

        # Aktualisiere die Positionen der Knoten- und Stützpunkte.
        set_data(self.knoten, sw.punkte[sw.indizes_knoten].T)
        set_data(self.stuetzpunkte, sw.punkte[sw.indizes_stuetz].T)

        # Aktualisiere die Koordinaten der Stäbe.
        for i, linie in enumerate(self.staebe):
            set_data(linie, sw.punkte[sw.staebe[i], :].T)
            linie.set_color(self._farbe_kraft(stabkraefte[i]))

        # Aktualisiere die Pfeile der Stabkräfte.
        for (i_punkt, i_stab), pfeil in self.pfeile_stab.items():
            p1 = sw.punkte[i_punkt]
            p2 = p1 + scal_kraft * sw.stabkraft(i_punkt, i_stab)
            pfeil.set_positions(p1, p2)

        # Aktualisiere die Pfeile der externen Kräfte.
        for i, pfeil in enumerate(self.pfeile_extern):
            p1 = sw.punkte[i]
            p2 = p1 + scal_kraft * sw.kraefte_ext[i]
            pfeil.set_positions(p1, p2)

        # Aktualisiere die Pfeile der Gewichtskräfte.
        for i, pfeil in enumerate(self.pfeile_grav):
            p1 = sw.punkte[i]
            p2 = p1 + scal_kraft * sw.gewichtskraefte()[i]
            pfeil.set_positions(p1, p2)

        # Aktualisiere die Pfeile der Stützkräfte.
        for i, pfeil in enumerate(self.pfeile_stuetz):
            p1 = sw.punkte[sw.indizes_stuetz[i]]
            p2 = p1 + scal_kraft * sw.stuetzkraefte()[i]
            pfeil.set_positions(p1, p2)

        # Aktualisiere die Beschriftung der Stabkräfte.
        for i, annot in enumerate(self.annot_stab):
            p = np.mean(sw.punkte[sw.staebe[i]], axis=0)
            annot.set_position(p)
            annot.set_text(self._format_kraft(stabkraefte[i]))
            annot.set_color(self._farbe_kraft(stabkraefte[i]))

        # Aktualisiere die Beschriftung der externen Kräfte.
        for i, annot in enumerate(self.annot_extern):
            kraft = sw.kraefte_ext[i]
            p = sw.punkte[i] + scal_kraft * kraft / 2
            annot.set_position(p)
            annot.set_text(self._format_kraft(kraft))

        # Aktualisiere die Beschriftung der Gewichtskräfte.
        for i, annot in enumerate(self.annot_grav):
            kraft = sw.gewichtskraefte()[i]
            p = sw.punkte[i] + scal_kraft * kraft / 2
            annot.set_position(p)
            annot.set_text(self._format_kraft(kraft))

        # Aktualisiere die Beschriftung der Stützkräfte.
        for i, annot in enumerate(self.annot_stuetz):
            kraft = sw.stuetzkraefte()[i]
            i_punkt = sw.indizes_stuetz[i]
            p = sw.punkte[i_punkt] + scal_kraft * kraft / 2
            annot.set_position(p)
            annot.set_text(self._format_kraft(kraft))


class AnimationEigenmode(PlotStabwerk):
    """Animierte Darstellung einer Eigenmode eines Stabwerks.

    Args:
        ax (mpl.axes.Axes):
            Axes, in die geplottet werden soll.
        stabwerk (StabwerkElastisch):
            Darzustellendes Stabwerk.
        eigenmode (int):
            Index der dazustellenden Eigenmode.
        amplitude (float):
            Amplitude der Eigenmode [m].
        schritte (int):
            Anzahl der Frames pro Periode.
        ruhelage (bool):
            Soll die Ruhelage des Stabwerks dargestellt werden?
        anim_args (dict):
            Schlüsselwortargumente
            für `mpl.animation.FuncAnimation`.
        **kwargs:
            Schlüsselwortargumente für `PlotStabwerk`.
        """

    def __init__(self, ax, stabwerk, eigenmode,
                 amplitude=0.1, schritte=40, ruhelage=True,
                 anim_args=None, **kwargs):
        super().__init__(ax, stabwerk, **kwargs)

        if ruhelage:
            # Keine Pfeile oder Beschriftungen der Ruhelage.
            kwargs = kwargs.copy()
            kwargs['annot'] = False
            kwargs['arrows'] = False
            kwargs['cmap'] = None
            self.stat = PlotStabwerk(ax, stabwerk, **kwargs)
            """PlotStabwerk: Statischer Plot der Ruhelage."""

            # Die Grafikobjekte der Ruhelage sollen wegen der
            # besseren Übersichtlichkeit in stark aufgehellten
            # Farben dargestellt werden. Dazu wird zu den roten,
            # grünen und blauen Farbkomponenten, jeweils eine
            # Konstante addiert. Diese Komponenten werden danach
            # durch eine Normierungskonstante geteilt, sodass die
            # Werte wieder im Zahlenbereich von 0 bis 1 liegen.
            aufhellung = 3.0
            artists = self.stat.staebe + [self.stat.knoten,
                                          self.stat.stuetzpunkte]
            for artist in artists:
                c = np.array(mpl.colors.to_rgba(artist.get_color()))
                c[:3] += aufhellung
                c[:3] /= aufhellung + 1
                artist.set_color(c)

            # Die in PlotStabwerk festgelegten Werte für zorder
            # liegen beginnen bei 120. Wir ziehen jeweils 100
            # davon ab, sodass die Elemente der Darstellung der
            # Ruhelage im Hintergrund dargestellt werden.
            for artist in artists:
                artist.set_zorder(artist.get_zorder() - 100)

        # Bestimme die Eigenmoden und wähle die angegebene aus.
        frequenzen, moden = self.sw.eigenmoden()
        self.freq = frequenzen[eigenmode]
        self.mode = moden[eigenmode]
        self.amplitude = amplitude
        self.schritte = schritte

        # Bestimme einen sinnvollen Wert der Maximalkraft für die
        # Farbtabelle, indem die Kräfte in einem Maximum der
        # Eigenmode ausgewertet werden.
        punkte_ruhelage = self.sw.punkte
        maximalkraft = 0
        for x in np.linspace(0, 2 * np.pi, self.schritte):
            auslenkungen = self.amplitude * self.mode * np.sin(x)
            self.sw.punkte = punkte_ruhelage + auslenkungen
            stabkraefte = self.sw.stabkraefte_scal()
            maximalkraft = max(maximalkraft,
                               np.max(np.abs(stabkraefte)))
        self.sw.punkte = punkte_ruhelage
        self.mapper.set_array([-maximalkraft, maximalkraft])
        self.mapper.autoscale()
        self.update_stabwerk()
        
        # Sorge dafür, dass die animierten Elemente zunächst
        # noch nicht dargestellt werden.
        self.knoten.set_data([], [])
        for stab in self.staebe:
            stab.set_data([], [])

        # Wenn keine Argumente für FuncAnimation angegeben wurden,
        # werden sinnvolle Standardwerte gesetzt.
        if anim_args is None:
            anim_args = {'interval': 30, 'blit': True}

        # Erzeuge eine Animation.
        self.ani = mpl.animation.FuncAnimation(ax.figure,
                                               self._update,
                                               **anim_args)

    def _update(self, n):
        """Aktualisiere die Grafik zum n-ten Zeitschritt."""
        # Verschiebe temporär die Punkte des Stabwerks und
        # plotte dieses.
        punkte_kopie = self.sw.punkte.copy()
        self.sw.punkte += (self.amplitude
                           * np.sin(2 * np.pi * n / self.schritte)
                           * self.mode)
        self.update_stabwerk()
        self.sw.punkte = punkte_kopie
        return self.artists


class Arrow3D(mpl.patches.FancyArrowPatch):
    """Darstellung eines Pfeiles in einer 3D-Grafik.

    Args:
        posA (tuple):
            Koordinaten (x, y, z) des Startpunktes.
        posB (tuple):
            Koordinaten (x, y, z) des Endpunktes.
        *args:
            Argumente für `matplotlib.patches.FancyArrowPatch`.
        **kwargs:
            Schlüsselwortargumente für
                `matplotlib.patches.FancyArrowPatch`.
    """

    def __init__(self, posA, posB, *args, **kwargs):
        super().__init__(posA[0:2], posB[0:2], *args, **kwargs)
        self._pos = np.array([posA, posB])

    def set_positions(self, pos_a, pos_b):
        """Setze den Start- und Endpunkt des Pfeils."""
        self._pos = np.array([pos_a, pos_b])

    def do_3d_projection(self, renderer=None):
        """Projiziere die Punkte in die Bildebene."""
        p = mpl_toolkits.mplot3d.proj3d.proj_transform(*self._pos.T, self.axes.M)
        p = np.array(p)
        super().set_positions(p[:, 0], p[:, 1])
        return np.min(p[2, :])


class Annotation3D(mpl.text.Annotation):
    """Darstellung einer Annotation in einer 3D-Grafik.

    Args:
        s (str):
            Dazustellender Text.
        pos (tuple):
            Koordinaten (x, y, z) der Annotation.
        *args:
            Argumente für `matplotlib.text.Annotation`.
        **kwargs:
            Schlüsselwortargumente für
            `matplotlib.text.Annotation`.
    """

    def __init__(self, s, pos, *args, **kwargs):
        super().__init__(s, xy=(0, 0), *args,
                         xytext=(0, 0),
                         textcoords='offset points',
                         **kwargs)
        self._pos = np.array(pos)

    def set_position(self, pos):
        """Setze die Position der Beschriftung."""
        self._pos = np.array(pos)

    def draw(self, renderer):
        """Zeichne die Annotation."""
        p = mpl_toolkits.mplot3d.proj3d.proj_transform(
            *self._pos, self.axes.M)
        self.xy = p[0:2]
        super().draw(renderer)

    def draggable(self, state=None, use_blit=False):
        """Die Anforderung wird ignoriert.

        Die Annotation3D ist nicht manuell verschiebbar. Das
        manuelle Verschieben einer Annotation mit der Maus in
        einem 3D-Koordinatensystem ist schwer realisierbar,
        weil das Ziehen mit der Maus bereits für die Kontrolle
        des Blickwinkels verwendet wird.
        """
        super().draggable(False)
