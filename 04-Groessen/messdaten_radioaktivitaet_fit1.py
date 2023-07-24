"""Kurvenanpassung an Messwerte."""

import numpy as np
import scipy.optimize
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


def fitfunktion(x, nu, n0, alpha):
    """Berechne die anzufittende Funktion."""
    return nu + n0 * np.exp(-alpha * x)


# Führe die Kurvenanpassung durch.
popt, pcov = scipy.optimize.curve_fit(fitfunktion, messwerte_d,
                                      messwerte_n, [40, 2200, 10])
fitwert_nu, fitwert_n0, fitwert_alpha = popt
fehler_nu, fehler_n0, fehler_alpha = np.sqrt(np.diag(pcov))

# Gib das Ergebnis der Kurvenanpassung aus.
print('Ergebnis der Kurvenanpassung:')
print(f'  n_u = ({fitwert_nu:4.0f} +- {fehler_nu:2.0f}) 1/min.')
print(f'  n_0 = ({fitwert_n0:4.0f} +- {fehler_n0:2.0f}) 1/min.')
print(f'alpha = ({fitwert_alpha:.2f} +- {fehler_alpha:.2f}) 1/mm.')

# Erzeuge die Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Filterdicke $d$ [mm]')
ax.set_ylabel('Intensität $n$ [1/min]')
ax.set_yscale('log')
ax.grid()

# Plotte die angepasste Funktion.
d = np.linspace(np.min(messwerte_d), np.max(messwerte_d), 500)
n = fitfunktion(d, fitwert_nu, fitwert_n0, fitwert_alpha)
ax.plot(d, n, '-')

# Plotte die Messwerte und zeige die Grafik an.
ax.errorbar(messwerte_d, messwerte_n, yerr=fehlerwerte_n,
            fmt='.', capsize=2)
plt.show()
