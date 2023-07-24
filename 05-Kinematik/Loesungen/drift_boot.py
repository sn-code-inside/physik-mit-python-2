"""Ein Boot steuert in einer Strömung ein Ziel an."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Startposition (x, y) des Bootes [m].
r0_boot = np.array([-10, 2.0])
# Position (x, y) des Zieles [m].
r_ziel = np.array([0.0, 0.0])
# Vektor der Geschwindigkeit (vx, vy) der Strömung [m/s].
v_stroem = np.array([0.0, -2.5])
# Betrag der Relativgeschwindigkeit des Bootes zum Wasser [m/s].
betrag_v_rel_boot = 3.0
# Maximale Simulationsdauer [s].
t_max = 500
# Zeitschrittweite [s].
dt = 0.01

# Mindestabstand, bei dem die Simulation abgebrochen wird.
mindestabstand = betrag_v_rel_boot * dt

# Lege Listen an, um die Simulationsergebnisse zu speichern.
t = [0]

# Liste der Ortsvektoren des Bootes.
r_boot = [r0_boot]     # Ortsvektor des Bootes.
# Liste der Geschwindigkeitsvektoren des Bootes.
v_boot = []       # Geschwindigkeitsvektor des Bootes.
# Liste mit den Richtungsvektoren des Bootes. Jeder Eintrag
# enthält den Einheitsvektor, der angibt in welche Richtung die
# Bootsspitze zeigt.
richtungen = []

# Schleife der Simulation.
while True:
    # Lege die Richtung fest, in die das Boot steuert.
    r_boot_ziel = r_ziel - r_boot[-1]
    abstand = np.linalg.norm(r_boot_ziel)
    richtungen.append(r_boot_ziel / np.linalg.norm(r_boot_ziel))

    # Berechne den neuen Geschwindigkeitsvektor des Bootes.
    v_boot.append(v_stroem
                  + betrag_v_rel_boot * richtungen[-1])

    # Beende die Simulation, wenn der Abstand von Boot und
    # Ziel klein genug ist oder die maximale Simulationszeit
    # überschritten ist.
    if (abstand < mindestabstand) or (t[-1] > t_max):
        break

    # Berechne die neue Position des Bootes und die neue Zeit.
    r_boot.append(r_boot[-1] + dt * v_boot[-1])
    t.append(t[-1] + dt)

# Wandele die Listen in Arrays um. Die Zeilen entsprechen den
# Zeitpunkten und die Spalten entsprechen den Koordinaten.
t = np.array(t)
r_boot = np.array(r_boot)
v_boot = np.array(v_boot)
richtungen = np.array(richtungen)

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-11, 1)
ax.set_ylim(-5, 3)
ax.set_aspect('equal')
ax.grid()

# Erzeuge einen leeren Plot für die Bahnkurve des Bootes und zwei
# Punktplots für die Positionen von Boot und Ziel.
plot_bahn_boot, = ax.plot([], [])
plot_boot, = ax.plot([], [], 'o', color='blue')
plot_ziel, = ax.plot(r_ziel[0], r_ziel[1], 'o', color='red')

# Erzeuge zwei Pfeile für die Geschwindigkeit und die aktuelle
# Ausrichtung des Bootes
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style)
pfeil_richtung = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                             color='black',
                                             arrowstyle=style)

# Füge die Grafikobjekte zur Axes hinzu.
ax.add_patch(pfeil_v)
ax.add_patch(pfeil_richtung)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Setze den Start- und Endpunkt des Geschwindigkeitspfeils.
    pfeil_v.set_positions(r_boot[n], r_boot[n] + v_boot[n])

    # Setze den Start- und Endpunkt des Pfeiles für die
    # Bootsausrichtung.
    pfeil_richtung.set_positions(r_boot[n],
                                 r_boot[n] + richtungen[n])

    # Aktualisiere die Position des Bootes.
    plot_boot.set_data(r_boot[n].reshape(-1, 1))

    # Plotte die Bahnkurve des Bootes bis zum aktuellen
    # Zeitpunkt.
    plot_bahn_boot.set_data(r_boot[:n + 1, 0], r_boot[:n + 1, 1])

    return plot_boot, pfeil_v, pfeil_richtung, plot_bahn_boot


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
