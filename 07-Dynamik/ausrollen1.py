"""Simulation des Ausrollens eines Fahrzeugs: Euler-Verfahren."""

import numpy as np
import matplotlib.pyplot as plt

# Zeitdauer, die simuliert werden soll [s].
t_max = 20
# Zeitschrittweite [s].
dt = 0.2
# Masse des Fahrzeugs [kg].
m = 15.0
# Reibungskoeffizient [kg / m].
b = 2.5
# Anfangsort [m].
x0 = 0
# Anfangsgeschwindigkeit [m/s].
v0 = 10.0


def F(v):
    """Berechne die Kraft als Funktion der Geschwindigkeit v."""
    return - b * v * np.abs(v)


# Lege Arrays für das Simulationsergebnis an.
t = np.arange(0, t_max, dt)
x = np.empty(t.size)
v = np.empty(t.size)

# Lege die Anfangsbedingungen fest.
x[0] = x0
v[0] = v0

# Schleife der Simulation.
for i in range(t.size - 1):
    x[i+1] = x[i] + v[i] * dt
    v[i+1] = v[i] + F(v[i]) / m * dt

# Erzeuge eine Figure.
fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)

# Plotte das Geschwindigkeits-Zeit-Diagramm.
ax_geschw = fig.add_subplot(1, 2, 1)
ax_geschw.set_xlabel('$t$ [s]')
ax_geschw.set_ylabel('$v$ [m/s]')
ax_geschw.grid()
ax_geschw.plot(t, v0 / (1 + v0 * b / m * t),
               '-b', label='analytisch')
ax_geschw.plot(t, v, '.r', label='simuliert')
ax_geschw.legend()

# Plotte das Orts-Zeit-Diagramm.
ax_ort = fig.add_subplot(1, 2, 2)
ax_ort.set_xlabel('$t$ [s]')
ax_ort.set_ylabel('$x$ [m]')
ax_ort.grid()
ax_ort.plot(t, m / b * np.log(1 + v0 * b / m * t),
            '-b', label='analytisch')
ax_ort.plot(t, x, '.r', label='simuliert')
ax_ort.legend()

# Zeige die Grafik an.
plt.show()
