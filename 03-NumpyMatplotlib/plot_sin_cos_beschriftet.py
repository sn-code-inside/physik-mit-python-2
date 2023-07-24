"""Graph der Sinus- und Kosinusfunktion mit Beschriftung."""

import numpy as np
import matplotlib.pyplot as plt

# Erzeuge ein Array für die x-Werte in Grad.
x = np.linspace(0, 360, 500)

# Erzeuge je ein Array für die zugehörigen Funktionswerte.
y_sin = np.sin(np.radians(x))
y_cos = np.cos(np.radians(x))

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Beschrifte die Achsen. Lege den Wertebereich fest und erzeuge
# ein Gitternetz.
ax.set_title('Sinus- und Kosinusfunktion')
ax.set_xlabel('Winkel [Grad]')
ax.set_ylabel('Funktionswert')
ax.set_xlim(0, 360)
ax.set_ylim(-1.1, 1.1)
ax.grid()

# Plotte die Funktionsgraphen und erzeuge eine Legende.
ax.plot(x, y_sin, label='Sinus')
ax.plot(x, y_cos, label='Kosinus')
ax.legend()

# Zeige die Grafik an.
plt.show()
