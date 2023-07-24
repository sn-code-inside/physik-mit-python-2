"""Modellierung eines Skihangs mit einem Polynom."""

import numpy as np
import matplotlib.pyplot as plt

# Stützstellen (Koordinaten) des Hangs [m].
x_hang = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 35.0,
                   40.0, 45.0, 55.0, 70.0])
y_hang = np.array([10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0,
                   3.5, 1.5,  0.02,  0.0])

# Fitte ein Polynom an die Stützpunkte an.
polynom = np.polyfit(x_hang, y_hang, 10)

# Erzeuge ein fein aufgelöstes Array von x-Werten und werte das
# Polynom an diesen Stellen aus.
x_polynom = np.linspace(x_hang[0], x_hang[-1], 500)
y_polynom = np.polyval(polynom, x_polynom)

# Erstelle eine Figure mit einer Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.grid()
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')

# Plotte die Stützstellen als blau Kreise und die interpolierte
# Funktion als blaue Linie.
ax.plot(x_hang, y_hang, 'ob')
ax.plot(x_polynom, y_polynom, '-b')

# Zeige die Grafik an.
plt.show()
