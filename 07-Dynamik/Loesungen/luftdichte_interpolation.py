"""Vergleich unterschiedlicher Interpolationsverfahren.

Interpolation der Messdaten der Luftdichte als Funktion der
Höhe. In diesem Programm werden die unterschiedlichen
Interpolationsarten verglichen:
   1.) Nächste-Nachbarn-Interpolation (nearest)
   2.) Lineare Interpolation (linear)
   3.) Kubische Interpolation (cubic)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# Messdaten Höhe [km] und Luftdichte [kg/m³].
messwerte_h = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        11.02, 15, 20.06, 25, 32.16, 40])
messwerte_rho = np.array([1.225, 1.112, 1.007, 0.909, 0.819, 0.736,
                          0.660, 0.590, 0.526, 0.467, 0.414, 0.364,
                          0.195, 0.0880, 0.0401, 0.0132, 0.004])

# Erzeuge die Interpolationsfunktionen.
interp_cubic = scipy.interpolate.interp1d(messwerte_h,
                                          messwerte_rho,
                                          kind='cubic')
interp_linear = scipy.interpolate.interp1d(messwerte_h,
                                           messwerte_rho,
                                           kind='linear')
interp_nearest = scipy.interpolate.interp1d(messwerte_h,
                                            messwerte_rho,
                                            kind='nearest')

# Erzeuge ein fein aufgelöstes Array von Höhen.
h = np.linspace(0, np.max(messwerte_h), 1000)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$h$ [km]')
ax.set_ylabel('$\\rho$ [kg/m³]')
ax.set_xlim(10, 40)
ax.set_ylim(0, 0.4)
ax.grid()

# Plotte die Messdaten und die drei Interpolationsfunktionen.
ax.plot(messwerte_h, messwerte_rho, 'or', label='Messung', zorder=5)
ax.plot(h, interp_nearest(h), '-k', label='nearest')
ax.plot(h, interp_linear(h), '-b', label='linear')
ax.plot(h, interp_cubic(h), '-r', label='cubic')
ax.legend()

# Zeige die Grafik an.
plt.show()
