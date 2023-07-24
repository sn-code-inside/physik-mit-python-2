"""Halblogarithmische Auftragung der Messwerte."""

import numpy as np
import matplotlib.pyplot as plt

# Dicke der Filter [mm].
messwerte_d = np.array([0.000, 0.029, 0.039, 0.064, 0.136, 0.198,
                        0.247, 0.319, 0.419, 0.511, 0.611, 0.719,
                        0.800, 0.900, 1.000, 1.100, 1.189])
# Gemessene Intensität [Impulse / min].
messwerte_n = np.array([2193, 1691, 1544, 1244, 706, 466,
                        318, 202, 108, 80, 52, 47,
                        45, 46, 47, 42, 43], dtype=float)
# Fehler der gemessenen Intensität [Impulse / min].
fehlerwerte_n = np.array([47, 41, 39, 35, 26, 22,
                          18, 14, 10, 9, 7, 7,
                          7, 7, 7, 7, 7], dtype=float)

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Filterdicke $d$ [mm]')
ax.set_ylabel('Intensität $n$ [1/min]')
ax.set_yscale('log')
ax.grid()

# Stelle die Messdaten mit Fehlerbalken dar.
ax.errorbar(messwerte_d, messwerte_n, yerr=fehlerwerte_n,
            fmt='.', capsize=2)
plt.show()
