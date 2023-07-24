"""GUI-Applikation eines Spielwürfels.

In diesem Beispiel wird die mit dem QtDesigner erstellte .ui-Datei
direkt mithilfe der Funktion loadUiType geladen.
"""

import random
from PyQt5 import QtWidgets
from PyQt5.uic import loadUiType

# Lade das mit dem Qt Designer erstellte Benutzerinterface.
Ui_MainWindow = loadUiType('wuerfel_gui.ui')[0]


# Definiere eine Klasse, die von QMainWindow abgeleitet wird.
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    """Hauptfenster der Anwendung."""

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Lösche beim Drücken des Knopfes das Ergebnis und würfle
        # beim Loslassen des Knopfes neu.
        self.button_wuerfeln.pressed.connect(self.loeschen)
        self.button_wuerfeln.released.connect(self.wuerfeln)

    def loeschen(self):
        """Lösche die Anzeige des Würfels."""
        self.label_anzeige.setText('')

    def wuerfeln(self):
        """Würfle eine Zahl zwischen 1 und 6 und zeige diese an."""
        zahl = random.randint(1, 6)
        self.label_anzeige.setText(f'{zahl}')


# Erzeuge eine QApplication und das Hauptfenster.
app = QtWidgets.QApplication([])
window = MainWindow()

# Zeige das Fenster an und starte die QApplication.
window.show()
app.exec()
