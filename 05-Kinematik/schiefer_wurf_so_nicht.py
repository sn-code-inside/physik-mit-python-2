"""Bahnkurve des schiefen Wurfs: Funktioniert so nicht."""

import math
import numpy as np
import matplotlib.pyplot as plt

# Anfangshöhe [m].
h = 10.0
# Abwurfgeschwindigkeit [m/s].
betrag_v0 = 9.0
# Abwurfwinkel [rad].
alpha = math.radians(25.0)
# Schwerebeschleunigung [m/s²].
g = 9.81

# Stelle die Vektoren als 1-dimensionale Arrays dar.
r0 = np.array([0, h])
v0 = betrag_v0 * np.array([math.cos(alpha), math.sin(alpha)])
a = np.array([0, -g])

# Berechne den Auftreffzeitpunkt auf dem Boden.
t_ende = v0[1] / g + math.sqrt((v0[1] / g) ** 2 + 2 * r0[1] / g)

# Erzeuge ein Array von Zeitpunkten.
t = np.linspace(0, t_ende, 1000)

# Berechne die Ortsvektoren für diese Zeitpunkte.
r = r0 + v0 * t + 0.5 * a * t**2
