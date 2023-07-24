"""Kurvenanpassung: Strömungswiderstand."""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# Geschwindigkeit [m/s].
messwerte_v = np.array([5.8, 7.3, 8.9, 10.6, 11.2])

# Fehler der Geschwindigkeit [m/s].
fehlerwerte_v = np.array([0.3, 0.3, 0.2, 0.2, 0.1])

# Kraft [N].
messerwerte_F = np.array([0.10, 0.15, 0.22, 0.33, 0.36])

# Fehler der Kraft [N].
fehlerwerte_F = np.array([0.02, 0.02, 0.02, 0.02, 0.02])


def fitfunktion(v, b, n):
    """Berechne die anzufittende Funktion."""
    return b * v ** n


# Führe die Kurvenanpassung durch.
popt, pcov = scipy.optimize.curve_fit(fitfunktion, messwerte_v,
                                      messerwerte_F, [1.5, 2.0],
                                      sigma=fehlerwerte_F)
fitwert_b, fitwert_n = popt
fehler_b, fehler_n = np.sqrt(np.diag(pcov))

# Gib das Ergebnis der Kurvenanpassung aus.
print('Ergebnis der Kurvenanpassung:')
print(f'  b = ({fitwert_b:6.4f} +- {fehler_b:6.4f}) N / (m/s)^n.')
print(f'  n = {fitwert_n:4.2f} +- {fehler_n:2.2f}')

# Erzeuge die Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Geschwindigkeit $v$ [m/s]')
ax.set_ylabel('Widerstandskraft $F$ [N]')
ax.grid()

# Plotte die angepasste Funktion mit einer hohen Auflösung.
v = np.linspace(np.min(messwerte_v), np.max(messwerte_v), 500)
F = fitfunktion(v, fitwert_b, fitwert_n)
ax.plot(v, F, '-')

# Plotte die Messwerte und zeige die Grafik an.
ax.errorbar(messwerte_v, messerwerte_F,
            xerr=fehlerwerte_v, yerr=fehlerwerte_F,
            fmt='.', capsize=2)
plt.show()
