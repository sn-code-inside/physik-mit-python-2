"""GUI mit PyQT ohne irgendwelche Funktionalität."""

from PyQt5 import QtWidgets

# Erzeuge eine QApplication und das Hauptfenster.
app = QtWidgets.QApplication([])
window = QtWidgets.QMainWindow()

# Zeige das Fenster an und starte die QApplication.
window.show()
app.exec_()
