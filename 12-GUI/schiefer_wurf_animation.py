"""GUI-Applikation: Animierte Darstellung des schiefen Wurfs."""

# Importiere Matplotlib.
import matplotlib as mpl
import matplotlib.backends.backend_qt5agg
import matplotlib.figure
import matplotlib.animation

# Importiere die notwendigen Elemente für die GUI.
from PyQt5 import QtWidgets
from PyQt5 import uic

# Importiere sonstige Module.
import math
import numpy as np
import scipy.integrate

# Lade das mit dem Qt Designer erstellte Benutzerinterface.
Ui_MainWindow = uic.loadUiType('schiefer_wurf_animation_gui.ui')[0]


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Hauptfenster der Anwendung."""

    def __init__(self):
        """Initialisiere das Hauptfenster."""
        super().__init__()
        self.setupUi(self)

        self.eingabe_okay = True
        """bool: Statusvariable (False, bei einer Fehleingabe)."""
        self.animation_laeuft = False
        """bool: Statusvariable (True, wenn die Animation läuft)"""
        self.bildnummer = 0
        """int: Nummer des Bildes, das aktuell dargestellt wird."""
        self.fig = mpl.figure.Figure()
        """Figure: Figure für den Plot der Bahnkurve."""
        self.ax = self.fig.add_subplot(1, 1, 1)
        """Axes: Axes für den Plot der Bahnkurve."""

        # Konfiguriere die Figure und die Axes.
        self.fig.set_tight_layout(True)
        self.ax.set_xlabel('$x$ [m]')
        self.ax.set_ylabel('$y$ [m]')
        self.ax.set_aspect('equal')
        self.ax.grid()

        # Füge die Zeichenfläche (canvas) in die GUI ein.
        mpl.backends.backend_qt5agg.FigureCanvasQTAgg(self.fig)
        self.box_plot.layout().addWidget(self.fig.canvas)

        # Erzeuge einen Linienplot für die Bahnkurve.
        self.plot_bahn, = self.ax.plot([0], [0], zorder=5)
        """Line2D: Plot der Bahnkurve."""

        # Erzeuge einen Punktplot für die aktuelle Position.
        self.plot_punkt, = self.ax.plot([], [], 'or', zorder=6)
        """Line2D: Punktplot für die aktuelle Position."""

        # Erzeuge ein Textfeld zur Ausgabe der aktuellen Zeit.
        self.text_zeit = self.ax.text(0.01, 0.99, '',
                                      verticalalignment='top',
                                      transform=self.ax.transAxes)
        """Text: Ausgabe der aktuellen Zeit."""

        # Initialisiere die Attribute für das Simulationsergebnis.
        n_punkte = 500
        self.t = np.zeros(n_punkte)
        """np.ndarray: Berechnete Zeitpunkte."""
        self.r = np.zeros((2, n_punkte))
        """np.ndarray: Berechnete Ortsvektoren (2 × n)."""
        self.v = np.zeros((2, n_punkte))
        """np.ndarray: Berechnete Geschwindigkeiten (2 × n)."""

        # Starte die Animation und übergib den Generator.
        self.anim = mpl.animation.FuncAnimation(
            self.fig, self.update_anim, frames=self.frames(),
            interval=30, blit=True)
        """FuncAnimation: Matplotlib-Animationsobjekt."""

        # Starte erstmalig die Simulation.
        self.winkelanzeige()
        self.simulation()

        # Wenn sich der Wert des Sliders ändert, dann soll auch
        # das Feld mit dem numerischen Wert des Winkels
        # aktualisiert werden.
        self.slider_alpha.valueChanged.connect(self.winkelanzeige)

        # Wenn einer der Eingabewerte verändert wird, dann soll
        # automatisch eine neue Simulation gestartet werden.
        self.slider_alpha.valueChanged.connect(self.simulation)
        self.edit_h.editingFinished.connect(self.simulation)
        self.edit_v.editingFinished.connect(self.simulation)
        self.edit_m.editingFinished.connect(self.simulation)
        self.edit_cwArho.editingFinished.connect(self.simulation)
        self.edit_g.editingFinished.connect(self.simulation)
        self.edit_xmax.editingFinished.connect(self.simulation)
        self.edit_ymax.editingFinished.connect(self.simulation)

        # Verbinde die Knöpfe zum Starten und Beenden der Animation.
        self.button_start.clicked.connect(self.start_animation)
        self.button_stop.clicked.connect(self.stop_animation)

    def winkelanzeige(self):
        """Aktualisiere das Feld für die Winkelangabe."""
        alpha = self.slider_alpha.value()
        self.label_alpha.setText(f'{alpha}°')

    def eingabe_float(self, field):
        """Lies eine Gleitkommazahl aus einem Textfeld aus."""
        try:
            value = float(field.text())
        except ValueError:
            self.eingabe_okay = False
            field.setStyleSheet("background: pink")
            self.statusbar.showMessage('Fehlerhafte Eingabe!')
        else:
            field.setStyleSheet("")
            return value

    def start_animation(self):
        """Starte die Animation."""
        self.animation_laeuft = True
        self.anim.event_source.start()

    def stop_animation(self):
        """Halte die Animation an."""
        self.animation_laeuft = False
        self.bildnummer = 0

    def simulation(self):
        """Starte eine Simulation mit neuen Parametern."""
        # Setze die Statusvariable zurück.
        self.eingabe_okay = True

        # Lies die Parameter aus den Eingabefeldern.
        hoehe = self.eingabe_float(self.edit_h)
        geschw = self.eingabe_float(self.edit_v)
        m = self.eingabe_float(self.edit_m)
        cwArho = self.eingabe_float(self.edit_cwArho)
        g = self.eingabe_float(self.edit_g)
        xmax = self.eingabe_float(self.edit_xmax)
        ymax = self.eingabe_float(self.edit_ymax)
        alpha = math.radians(self.slider_alpha.value())

        # Überprüfe, ob alle Eingaben gültig sind.
        if not self.eingabe_okay:
            return

        # Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
        r0 = np.array([0, hoehe])
        v0 = geschw * np.array([math.cos(alpha), math.sin(alpha)])
        u0 = np.concatenate((r0, v0))

        def dgl(t, u):
            """Berechne die rechte Seite der Differentialgl."""
            r, v = np.split(u, 2)
            # Luftreibungskraft.
            Fr = -0.5 * cwArho * np.linalg.norm(v) * v
            # Schwerkraft.
            Fg = m * g * np.array([0, -1])
            # Beschleunigung.
            a = (Fr + Fg) / m
            return np.concatenate([v, a])

        def aufprall(t, u):
            """Ereignisfunktion: Detektiere den Aufprall."""
            r, v = np.split(u, 2)
            return r[1]

        # Beende die Integration, beim Auftreffen auf den Boden.
        aufprall.terminal = True
        aufprall.direction = -1

        # Löse die Bewegungsgleichung.
        result = scipy.integrate.solve_ivp(dgl, [0, np.inf],
                                           u0,
                                           events=aufprall,
                                           dense_output=True)

        # Berechne die Interpolation auf einem feinen Raster.
        self.t = np.linspace(0, np.max(result.t), self.t.size)
        self.r, self.v = np.split(result.sol(self.t), 2)

        # Aktualisiere den Plot.
        self.plot_bahn.set_data(self.r[0], self.r[1])
        self.ax.set_xlim(0, xmax)
        self.ax.set_ylim(0, ymax)

        # Zeichne die Grafikelemente neu.
        self.fig.canvas.draw()

        # Lösche den Statustext der vorherigen Simulation.
        self.statusbar.clearMessage()

    def frames(self):
        """Generiere die Folge der Bildnummern.

        Es wird eine sich endlos wiederholende Folge der Indizes
        der Zeitpunkte erzeugt.
        """
        while True:
            yield self.bildnummer
            if self.animation_laeuft:
                self.bildnummer += 1
            if self.bildnummer >= self.t.size:
                self.bildnummer = 0

    def update_anim(self, n):
        """Aktualisiere die Grafik zum n-ten Zeitschritt."""
        self.plot_punkt.set_data(self.r[0, n].reshape(-1, 1),
                                 self.r[1, n].reshape(-1, 1))
        self.text_zeit.set_text(f'$t$ = {self.t[n]:.2f} s')

        return self.plot_punkt, self.plot_bahn, self.text_zeit

    def closeEvent(self, event):
        """Wird aufgerufen, wenn das Fenster geschlossen wird."""
        self.stop_animation()


# Erzeuge eine QApplication und das Hauptfenster.
app = QtWidgets.QApplication([])
window = MainWindow()

# Zeige das Fenster an und starte die QApplication.
window.show()
app.exec_()
