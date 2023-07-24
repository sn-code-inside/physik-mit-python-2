"""Funktionsgraph der Sinusfunktion."""

import numpy as np
import matplotlib.pyplot as plt

# Erzeuge ein Array für die x-Werte in Grad und für die y-Werte.
x = np.linspace(0, 360, 500)
y = np.sin(np.radians(x))

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Plotte die Funktionswerte.
ax.plot(x, y)

# Zeige die Grafik an.
plt.show()
