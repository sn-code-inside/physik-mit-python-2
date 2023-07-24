"""Simulation der Hundekurve."""

import numpy as np
import matplotlib.pyplot as plt

# Startposition (x, y) des Hundes [m].
r0_hund = np.array([0.0, 10.0])
# Startposition (x, y) des Menschen [m].
r0_mensch = np.array([0.0, 0.0])
# Vektor der Geschwindigkeit (vx, vy) des Menschen [m/s].
v_mensch = np.array([2.0, 0.0])
# Betrag der Geschwindigkeit des Hundes [m/s].
betrag_v_hund = 3.0
# Maximale Simulationsdauer [s].
t_max = 500
# Zeitschrittweite [s].
dt = 0.01

# Mindestabstand, bei dem die Simulation abgebrochen wird.
mindestabstand = betrag_v_hund * dt

# Lege Listen an, um das Simulationsergebnis zu speichern.
t = [0]
r_hund = [r0_hund]
r_mensch = [r0_mensch]
v_hund = []

# Schleife der Simulation.
while True:
    # Berechne den Geschwindigkeitsvektor des Hundes.
    r_hund_mensch = r_mensch[-1] - r_hund[-1]
    abstand = np.linalg.norm(r_hund_mensch)
    v_hund.append(betrag_v_hund * r_hund_mensch / abstand)

    # Beende die Simulation, wenn der Abstand von Hund und
    # Mensch klein genug ist oder die maximale Simulationszeit
    # überschritten ist.
    if (abstand < mindestabstand) or (t[-1] > t_max):
        break

    # Berechne die neuen Positionen und die neue Zeit.
    r_hund.append(r_hund[-1] + dt * v_hund[-1])
    r_mensch.append(r_mensch[-1] + dt * v_mensch)
    t.append(t[-1] + dt)

# Wandele die Listen in Arrays um. Die Zeilen entsprechen den
# Zeitpunkten und die Spalten entsprechen den Koordinaten.
t = np.array(t)
r_hund = np.array(r_hund)
v_hund = np.array(v_hund)
r_mensch = np.array(r_mensch)

# Berechne die Beschleunigungsvektoren des Hundes. Achtung:
# Dieses Array hat eine Zeile weniger, als es Zeitpunkte gibt.
a_hund = (v_hund[1:] - v_hund[:-1]) / dt

# Erzeuge eine Figure der Größe 10 inch x 3 inch.
fig = plt.figure(figsize=(10, 3))
fig.set_tight_layout(True)

# Plotte die Bahnkurve des Hundes.
ax_bahn = fig.add_subplot(1, 3, 1)
ax_bahn.set_xlabel('$x$ [m]')
ax_bahn.set_ylabel('$y$ [m]')
ax_bahn.set_aspect('equal')
ax_bahn.grid()
ax_bahn.plot(r_hund[:, 0], r_hund[:, 1])

# Plotte den Abstand von Hund und Mensch.
ax_dist = fig.add_subplot(1, 3, 2)
ax_dist.set_xlabel('$t$ [s]')
ax_dist.set_ylabel('Abstand [m]')
ax_dist.grid()
ax_dist.plot(t, np.linalg.norm(r_hund - r_mensch, axis=1))

# Plotte den Betrag der Beschleunigung des Hundes.
ax_beschl = fig.add_subplot(1, 3, 3)
ax_beschl.set_xlabel('$t$ [s]')
ax_beschl.set_ylabel('Beschl. [m/s²]')
ax_beschl.grid()
ax_beschl.plot(t[1:], np.linalg.norm(a_hund, axis=1))

# Zeige die Grafik an.
plt.show()
