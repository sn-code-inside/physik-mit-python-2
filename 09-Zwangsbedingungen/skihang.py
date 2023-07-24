"""Modellierung eines Skihangs mit kubischen Splines."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# Stützstellen (Koordinaten) des Hangs [m].
x_hang = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 35.0,
                   40.0, 45.0, 55.0, 70.0])
y_hang = np.array([10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0,
                   3.5, 1.5, 0.02, 0.0])

# Interpolation mit kubischen Splines.
hangfunktion = scipy.interpolate.CubicSpline(x_hang, y_hang,
                                             bc_type='natural')

# Berechne die Ableitung der Funktion f.
d_hangfunktion = hangfunktion.derivative(1)

# Erzeuge ein fein aufgelöstes Array von x-Werten zur
# Auswertung der interpolierten Funktion.
x_plot = np.linspace(x_hang[0], x_hang[-1], 500)

# Erstelle eine Figure mit einer Axes.
fig = plt.figure()
fig.set_tight_layout(True)
ax_ort = fig.add_subplot(1, 1, 1)
ax_ort.set_xlabel('$x$ [m]')
ax_ort.set_ylabel('$y$ [m]')
ax_ort.grid()

# Plotte die Stützstellen als blaue Kreise und die interpolierte
# Funktion als blaue Linie.
ax_ort.plot(x_hang, y_hang, 'ob')
ax_ort.plot(x_plot, hangfunktion(x_plot), '-b')

# Erzeuge eine zweite Axes mit roter Beschriftung und plotte
# die Steigung der interpolierten Funktion in Prozent.
ax_steigung = ax_ort.twinx()
ax_steigung.set_ylabel('Steigung [%]', color='red')
ax_steigung.tick_params(axis='y', labelcolor='red')
ax_steigung.plot(x_plot, 100 * d_hangfunktion(x_plot), '-r')

# Zeige die Grafik an.
plt.show()
