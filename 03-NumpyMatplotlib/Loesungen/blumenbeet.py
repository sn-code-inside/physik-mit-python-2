"""Darstellung und Flächenberechnung eines Blumenbeets."""

import numpy as np
import matplotlib.pyplot as plt

# Koordinaten der Eckpunkte [m].
x = np.array([0.0, 0.0, 1.0, 2.2, 2.8, 3.8, 4.6,
              5.7, 6.4, 7.1, 7.6, 7.9, 7.9, 0.0])
y = np.array([1.0, 2.8, 3.3, 3.5, 3.4, 2.7, 2.4,
              2.3, 2.1, 1.6, 0.9, 0.5, 0.0, 1.0])

# Berechne die Fläche mit der gaußschen Trapezformel.
flaeche = 0.5 * abs((y + np.roll(y, 1)) @ (x - np.roll(x, 1)))

# Gib ds Ergebnis als Antwortsatz aus.
print(f"Die Fläche beträgt {flaeche:.1f} m².")

# Erzeuge die Figure und das Axes-Objekt. Damit das Blumenbeet
# nicht verzerrt dargestellt wird, sorgen wir mit
#    ax.set_aspect('equal')
# dafür, dass die Skalierung in x- und y-Richtung gleich ist.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Flächenberechnung eines Blumenbeetes.')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.grid()

# Plotte die Umrandung des Blumenbeetes.
ax.plot(x, y)

# Erzeuge ein Textfeld mit der Flächenangabe in der Mitte der Axes.
ax.text(0.5, 0.5, f'A = {flaeche:.1f} m²', transform=ax.transAxes)

# Zeige die Grafik an.
plt.show()
