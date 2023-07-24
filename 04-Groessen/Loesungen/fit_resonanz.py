"""Kurvenanpassung: Resonanzkurve."""

import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# Anregungsfrequenz [Hz].
messwerte_f = np.array([0.2, 0.5, 0.57, 0.63, 0.67,
                        0.71, 0.80, 1.00, 1.33])

# Amplitude [cm].
messwerte_A = np.array([0.84, 1.42, 1.80, 2.10, 2.22,
                        2.06, 1.45, 0.64, 0.30])

# Fehler der Amplitude [cm].
fehlerwerte_A = np.array([0.04, 0.07, 0.09, 0.11, 0.11,
                          0.10, 0.08, 0.03, 0.02])


def fitfunktion(f, A0, f0, delta):
    """Berechne die anzufittende Funktion."""
    return A0 * f0**2 / np.sqrt((f**2 - f0**2)**2
                                + (delta * f / math.pi)**2)


# Führe die Kurvenanpassung durch.
popt, pcov = scipy.optimize.curve_fit(fitfunktion, messwerte_f,
                                      messwerte_A, [0.8, 0.7, 0.3],
                                      sigma=fehlerwerte_A)
fitwert_A0, fitwert_f0, fitwert_delta = popt
fehler_A0, fehler_f0, fehler_delta = np.sqrt(np.diag(pcov))

# Gib das Ergebnis der Kurvenanpassung aus.
print('Ergebnis der Kurvenanpassung:')
print(f'     A0 = ({fitwert_A0:.2f} +- {fehler_A0:.2f}) cm.')
print(f'     f0 = ({fitwert_f0:.3f} +- {fehler_f0:.3f}) Hz.')
print(f'  delta = ({fitwert_delta:.2f} +- {fehler_delta:.2f}) 1/s')


# Erzeuge die Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Anregungsfrequenz $f$ [Hz]')
ax.set_ylabel('Amplitude $A$ [cm]')
ax.grid()

# Plotte die angepasste Funktion.
f = np.linspace(np.min(messwerte_f), np.max(messwerte_f), 500)
A = fitfunktion(f, fitwert_A0, fitwert_f0, fitwert_delta)
ax.plot(f, A, '-', zorder=2)

# Plotte die Messwerte und zeige die Grafik an.
ax.errorbar(messwerte_f, messwerte_A, yerr=fehlerwerte_A,
            fmt='.', capsize=2)
plt.show()
