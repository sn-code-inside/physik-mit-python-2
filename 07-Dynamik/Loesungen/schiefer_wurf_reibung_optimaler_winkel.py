"""Simulation eines fliegenden Balls mit Luftreibung.

In diesem Programm wird für verschiedene Abwurfgeschwindigkeiten
jeweils der optimale Abwurfwinkel bestimmt. Die sonstigen
Parameter entsprechen der Simulation des Tischtennisballs.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize

# Masse des Körpers [kg].
m = 2.7e-3
# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.45 * math.pi * 20e-3 ** 2
# Bereich von Abwurfgeschwindigkeit [m/s].
v0_min = 0.1
v0_max = 50
# Anzahl der Abwurfgeschwindigkeiten in angegebenen Bereich von
# v0_min bis v0_max.
n_abwurfgeschwindigkeiten = 100
# Erdbeschleunigung [m/s²].
g = 9.81
# Luftdichte [kg/m³].
rho = 1.225
# Abwurfhöhe [m].
h = 1.1


def bahnkurve(alpha, h0, v0, n=1000):
    """Berechne die Bahnkurve eines schiefen Wurfs mit Luftreibung.

    Args:
        alpha (float):
            Abwurfwinkel [rad].
        h0 (float):
            Abwurfhöhe [m].
        v0 (float):
            Anfangsgeschwindigkeit [m/s].
        n (int):
            Anzahl der Datenpunkte.

    Returns:
        tuple[np.ndarray, np.ndarray]:
           - Die Zeitpunkte der Bahnkurve (n).
           - Die zugehörigen Ortsvektoren (n × 2).
    """
    # Lege den Anfangsort und die Anfangsgeschwindigkeit fest.
    r0 = np.array([0, h0])
    v0 = np.array([v0 * math.cos(alpha), v0 * math.sin(alpha)])

    def F(v):
        """Berechne die Kraft als Funktion der Geschwindigkeit."""
        Fr = -0.5 * rho * cwA * np.linalg.norm(v) * v
        Fg = m * g * np.array([0, -1])
        return Fg + Fr

    def dgl(t, u):
        """Berechne die rechte Seite der Differentialgleichung."""
        r, v = np.split(u, 2)
        return np.concatenate([v, F(v) / m])

    def aufprall(t, u):
        """Ereignisfunktion: Detektiere den Aufprall."""
        r, v = np.split(u, 2)
        return r[1]

    # Beende die Simulation beim Auftreten des Ereignisses.
    aufprall.terminal = True

    # Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
    u0 = np.concatenate((r0, v0))

    # Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde
    result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                       events=aufprall,
                                       dense_output=True)

    # Berechne die Interpolation auf einem feinen Raster.
    t = np.linspace(0, np.max(result.t), n)
    r, v = np.split(result.sol(t), 2)

    return t, r


def wurfweite(alpha, h0, v0):
    """Berechne die Wurfweite des schiefen Wurfs.

    Args:
        alpha (float):
            Abwurfwinkel [rad].
        h0 (float):
            Abwurfhöhe [m].
        v0 (float):
            Anfangsgeschwindigkeit [m/s].

    Returns:
        float: Die Wurfweite in horizontaler Richtung.
    """
    t, r = bahnkurve(alpha, h0, v0)
    # Die Wurfweite ist die x-Koordinate des letzten berechneten
    # Datenpunkts.
    return r[0, -1]


def negative_wurfweite(alpha, h0, v0):
    """Berechne die negative Wurfweite des schiefen Wurfs."""
    return -wurfweite(alpha, h0, v0)


# Erzeuge ein Array mit Anfangsgeschwindigkeiten.
abwurfgeschwindigkeiten = np.linspace(v0_min, v0_max,
                                      n_abwurfgeschwindigkeiten)

# Erzeuge ein leeres Array, das für jede Anfangsgeschwindigkeit
# den optimalen Abwurfwinkel aufnimmt.
alpha = np.empty(n_abwurfgeschwindigkeiten)

# Führe für jeden Wert der Anfangsgeschwindigkeit die Optimierung
# durch. Die Anfangshöhe und Geschwindigkeit muss als
# zusätzliches Argument an die Funktion func übergeben werden.
# Dies wird mit der Option arg=(h, v) bewirkt. Da der Winkel bei
# der Funktion `negative_wurfweite` im Bogenmaß angegeben wird,
# ist es sinnvoll, den Suchbereich mit bounds=(0, math.pi/2) auf
# den Bereich von 0 bis 90 Grad einzuschränken.
for i, v in enumerate(abwurfgeschwindigkeiten):
    result = scipy.optimize.minimize_scalar(negative_wurfweite,
                                            bounds=(0, math.pi/2),
                                            args=(h, v),
                                            method='bounded')
    alpha[i] = result.x

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Abwurfgeschwindigkeit [m/s]')
ax.set_ylabel('Optimaler Abwurfwinkel [°]')
ax.grid()

# Plotte das Ergebnis.
ax.plot(abwurfgeschwindigkeiten, np.degrees(alpha))

# Zeige die Grafik an.
plt.show()
