"""Simulation und animierte Darstellung der Hundekurve.

In diesem Programm wird die berechnete Lösung mit einer
analytischen Lösung verglichen, die für den hier gezeigten
Spezialfall gültig ist.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Startposition (x, y) des Hundes [m].
r0_hund = np.array([0.0, 10.0])
# Startposition (x, y) des Menschen [m].
r0_mensch = np.array([0.0, 0.0])
# Vektor der Geschwindigkeit (vx, vy) des Menschen [m/s].
v_mensch = np.array([2.0, 0.0])
# Betrag der Geschwindigkeit des Hundes.
betrag_v_hund = 3.0
# Maximale Simulationsdauer [s].
t_max = 500
# Zeitschrittweite [s].
dt = 0.01

# Mindestabstand, bei dem die Simulation abgebrochen wird.
mindestabstand = betrag_v_hund * dt

# Lege Listen an, um die Simulationsergebnisse zu speichern.
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

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-0.2, 15)
ax.set_ylim(-0.2, 10)
ax.set_aspect('equal')
ax.grid()

# Plotte die analytische Lösung der Bahnkurve.
y0 = r0_hund[1]
k = np.linalg.norm(v_mensch) / betrag_v_hund
y = np.linspace(0, y0, 500)
x = 0.5 * y0 * (
        (1 - (y / y0) ** (1 - k)) / (1 - k) -
        (1 - (y / y0) ** (1 + k)) / (1 + k))
ax.plot(x, y, '--k')

# Erzeuge einen leeren Linienplot für die Bahnkurve des Hundes und
# zwei leere Punktplots für die Positionen von Hund und Mensch.
plot_bahn_hund, = ax.plot([], [])
plot_hund, = ax.plot([], [], 'o', color='blue')
plot_mensch, = ax.plot([], [], 'o', color='red')

# Erzeuge zwei Pfeile für die Geschwindigkeit und die
# Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='black',
                                      arrowstyle=style)

# Füge die Grafikobjekte zur Axes hinzu.
ax.add_patch(pfeil_v)
ax.add_patch(pfeil_a)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Setze den Start- und Endpunkt des Geschwindigkeitspfeils.
    pfeil_v.set_positions(r_hund[n], r_hund[n] + v_hund[n])

    # Setze den Start- und Endpunkt des Beschleunigungspfeils.
    if n < len(a_hund):
        pfeil_a.set_positions(r_hund[n], r_hund[n] + a_hund[n])

    # Aktualisiere die Positionen von Hund und Mensch.
    plot_hund.set_data(r_hund[n].reshape(-1, 1))
    plot_mensch.set_data(r_mensch[n].reshape(-1, 1))

    # Plotte die Bahnkurve des Hundes bis zur aktuellen Zeit.
    plot_bahn_hund.set_data(r_hund[:n + 1, 0], r_hund[:n + 1, 1])

    return plot_bahn_hund, plot_hund, plot_mensch, pfeil_v, pfeil_a


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
