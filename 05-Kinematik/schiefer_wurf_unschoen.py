"""Bahnkurve des schiefen Wurfs ohne Reibung: Ineffizient."""

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

# Berechne die Ortsvektoren für alle Zeitpunkte im Array t.
r = np.empty((t.size, r0.size))
for i in range(t.size):
    for j in range(r0.size):
        r[i, j] = r0[j] + v0[j] * t[i] + 0.5 * a[j] * t[i]**2

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')
ax.grid()

# Plotte die Bahnkurve und zeige die Grafik an.
ax.plot(r[:, 0], r[:, 1])
plt.show()
