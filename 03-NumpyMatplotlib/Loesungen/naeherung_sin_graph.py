"""Berechnung des prozentualen Fehlers der Näherung sin(x) = x.

In diesem Beispiel wird der prozentuale Fehler der Näherung
sin(x) = x im Bereich von 5 Grad bis 90 Grad in 5
Grad-Schritten bestimmt. Das Ergebnis wird in NumPy-Arrays
gespeichert.
"""

import numpy as np
import matplotlib.pyplot as plt

# Lege ein Array der Winkel in Grad an.
winkel = np.linspace(1, 45, 500)

# Wandle die Winkel in das Bogenmaß um.
x = np.radians(winkel)

# Berechne die relativen Fehler.
fehler = 100 * (x - np.sin(x)) / np.sin(x)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Beschrifte die Achsen, lege den Wertebereich fest
# und erzeuge ein Gitternetz.
ax.set_title('Fehler der Näherung sin(x) $\\approx$ x')
ax.set_xlabel('Winkel [Grad]')
ax.set_ylabel('Relative Abweichung [%]')
ax.set_xlim(0, np.max(winkel))
ax.set_ylim(0, np.max(fehler))
ax.grid()

# Plotte die Funktionsgraphen und erzeuge eine Legende.
ax.plot(winkel, fehler)

# Zeige die Grafik an.
plt.show()
