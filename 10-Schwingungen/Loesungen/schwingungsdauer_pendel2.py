"""Schwingungsdauer des mathematischen Pendels.

Die Schwingungsdauer des mathematischen Pendels wird für große
Auslenkungswinkel mit der Integralformel bestimmt und mit dem
Ergebnis aus der numerischen Lösung der Differentialgleichung
verglichen.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Anzahl der Winkel.
n_winkel = 1000
# Minimaler und maximaler Auslenkwinkel [rad].
phi_min = math.radians(1)
phi_max = math.radians(175)

# Die Erdbeschleunigung g [m/s²] und die Pendellänge L [m]
# werden so gewählt, dass sich für kleine Auslenkungen eine
# Schwingungsdauer von 1 s ergibt.
g = 9.81
L = g / (2 * math.pi) ** 2


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    phi, dphi = u
    return [dphi, - g / L * np.sin(phi)]


def nulldurchgang(t, u):
    """Ereignisfunktion: Nulldurchgänge der Schwingung."""
    phi, dphi = u
    return phi


# Wir starten mit einer Auslenkung in positiver Richtung
# und ohne Anfangsgeschwindigkeit. Beim ersten
# Nulldurchgang des Winkels in negativer Richtung ist
# gerade 1/4 Periode vorbei.
nulldurchgang.terminal = True
nulldurchgang.direction = -1


def schwingungsdauer_dgl(phi0):
    """Berechne die Schwingungsdauer.

    Die Schwingungsdauer eines mathematischen Pendels der Länge
    L bei der Schwerebeschleunigung g wird über die numerische
    Lösung der Differentialgleichung bestimmt.

    Args:
        phi0 (float):
            Maximalwinkel [rad].

    Returns:
        float: Schwingungsdauer [s].
    """
    # Integriere die Differentialgleichung numerisch bis zum
    # ersten Nulldurchgang.
    u0 = np.array([phi0, 0])
    result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                       events=nulldurchgang)

    # Es wurde über eine viertel Periode integriert.
    return 4 * result.t[-1]


def f(phi, phi_max):
    """Integrand zur Berechnung der Periodendauer."""
    a = np.sqrt(2 * L / g)
    return a / np.sqrt(np.cos(phi) - np.cos(phi_max))


# Werte die Funktion `schwingungsdauer_dgl` für n Winkel zwischen
# phi_min und phi_max aus.
phi0 = np.linspace(phi_min, phi_max, n_winkel)
schwingungsdauern_dgl = np.empty(n_winkel)
for i in range(n_winkel):
    schwingungsdauern_dgl[i] = schwingungsdauer_dgl(phi0[i])

# Werte das Integral an den entsprechenden Winkeln aus.
schwingungsdauern_int = np.empty(n_winkel)
for i in range(n_winkel):
    schwingungsdauern_int[i], err = scipy.integrate.quad(
        f, -phi0[i], phi0[i], args=(phi0[i],))

# Berechne die relative Abweichung der beiden Ergebnisse.
abweichung_rel = schwingungsdauern_int / schwingungsdauern_dgl - 1

# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Erzeuge eine Axes und plotte die berechneten Schwingungsdauern
# als Funktion der Auslenkung.
ax_periode = fig.add_subplot(1, 1, 1)
ax_periode.set_xlabel('Maximalauslenkung [°]')
ax_periode.set_ylabel('Schwingungsdauer [s]')
ax_periode.grid()
ax_periode.plot(np.degrees(phi0), schwingungsdauern_dgl,
                'b--', zorder=5, label='num. Lösung der Dgl.')
ax_periode.plot(np.degrees(phi0), schwingungsdauern_int,
                'r-', zorder=4, label='Integralformel')
ax_periode.legend(loc='upper left')

# Erzeuge eine zweite Axes und plotte die relative Abweichung
# der beiden Ergebnisse.
ax_fehler = ax_periode.twinx()
ax_fehler.set_ylabel('Relative Abweichung [ppm]', color='green')
ax_fehler.tick_params(axis='y', labelcolor='green')
ax_fehler.plot(np.degrees(phi0), 1e6 * abweichung_rel,
               'g', zorder=3, label='rel. Abw.')
ax_fehler.legend(loc='upper right')

# Zeige die Grafik an.
plt.show()
