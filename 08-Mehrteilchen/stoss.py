"""Animation: Schräger Stoß zweier kreisförmiger Objekte."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Simulationszeit und Zeitschrittweite [s].
t_max = 8
dt = 0.02
# Massen der beiden Teilchen [kg].
m1 = 1.0
m2 = 2.0
# Radien der beiden Teilchen [m].
radius1 = 0.1
radius2 = 0.3
# Anfangspositionen [m].
r0_1 = np.array([-2.0, 0.1])
r0_2 = np.array([0.0, 0.0])
# Anfangsgeschwindigkeiten [m/s].
v1 = np.array([1.0, 0.0])
v2 = np.array([0.0, 0.0])


def koll_teilchen(r1, r2, v1, v2):
    """Berechne die Zeit bis zur nächsten Teilchenkollision.

    Args:
        r1 (np.ndarray):
            Ortsvektor des ersten Teilchens.
        r2 (np.ndarray):
            Ortsvektor des zweiten Teilchens.
        v1 (np.ndarray):
            Geschwindigkeitsvektor des ersten Teilchens.
        v2 (np.ndarray):
            Geschwindigkeitsvektor des zweiten Teilchens.

    Returns:
        float: Zeit bis zur Kollision oder NaN, falls die Teilchen
               nicht kollidieren.
    """
    # Differenz der Orts- und Geschwindigkeitsvektoren.
    dr = r1 - r2
    dv = v1 - v2

    # Um den Zeitpunkt der Kollision zu bestimmen, muss eine
    # quadratische Gleichung der Form
    #          t² + 2 a t + b = 0
    # gelöst werden. Nur die kleinere Lösung ist relevant.
    a = (dv @ dr) / (dv @ dv)
    b = (dr @ dr - (radius1 + radius2) ** 2) / (dv @ dv)
    D = a**2 - b
    t = -a - np.sqrt(D)
    return t


def stoss_teilchen(m1, m2, r1, r2, v1, v2):
    """Berechne die Geschwindigkeiten nach einem elastischen Stoß.

    Args:
        m1 (float):
            Masse des ersten Teilchens.
        m2 (float):
            Masse des zweiten Teilchens.
        r1 (np.ndarray):
            Ortsvektor des ersten Teilchens.
        r2 (np.ndarray):
            Ortsvektor des zweiten Teilchens.
        v1 (np.ndarray):
            Geschwindigkeitsvektor des ersten Teilchens.
        v2 (np.ndarray):
            Geschwindigkeitsvektor des zweiten Teilchens.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Die Geschwindigkeiten beider Teilchen nach dem Stoß.
    """
    # Berechne die Schwerpunktsgeschwindigkeit.
    v_schwerpunkt = (m1 * v1 + m2 * v2) / (m1 + m2)

    # Berechne die Richtung, in der der Stoß stattfindet.
    richtung = (r1 - r2) / np.linalg.norm(r1 - r2)

    # Berechne die neuen Geschwindigkeiten nach dem Stoß.
    v1_neu = v1 + 2 * (v_schwerpunkt - v1) @ richtung * richtung
    v2_neu = v2 + 2 * (v_schwerpunkt - v2) @ richtung * richtung
    return v1_neu, v2_neu


# Berechne Energie und Gesamtimpuls am Anfang.
E_anfang = 1 / 2 * m1 * v1 @ v1 + 1 / 2 * m2 * v2 @ v2
p_anfang = m1 * v1 + m2 * v2

# Lege Arrays für das Simulationsergebnis an.
t = np.arange(0, t_max, dt)
r1 = np.empty((t.size, r0_1.size))
r2 = np.empty((t.size, r0_2.size))

# Lege die Anfangsbedingungen fest.
r1[0] = r0_1
r2[0] = r0_2

# Berechne den Zeitpunkt der Kollision.
t_kollision = koll_teilchen(r1[0], r2[0], v1, v2)

# Schleife der Simulation.
for i in range(1, t.size):
    # Kopiere die Positionen aus dem vorherigen Zeitschritt.
    r1[i] = r1[i - 1]
    r2[i] = r2[i - 1]

    # Kollidieren die Teilchen in diesem Zeitschritt?
    if t[i - 1] < t_kollision <= t[i]:
        # Bewege die Teilchen bis zum Kollisionszeitpunkt.
        r1[i] += v1 * (t_kollision - t[i - 1])
        r2[i] += v2 * (t_kollision - t[i - 1])

        # Führe den Stoß aus.
        v1, v2 = stoss_teilchen(m1, m2, r1[i], r2[i], v1, v2)

        # Bewege die Teilchen bis zum Ende des Zeitschritts.
        r1[i] += v1 * (t[i] - t_kollision)
        r2[i] += v2 * (t[i] - t_kollision)
    else:
        # Bewege die Teilchen ohne Kollision.
        r1[i] += v1 * dt
        r2[i] += v2 * dt

# Berechne Energie und Gesamtimpuls am Ende.
E_ende = 1 / 2 * m1 * v1 @ v1 + 1 / 2 * m2 * v2 @ v2
p_ende = m1 * v1 + m2 * v2

# Gib die Energie und den Impuls vor und nach dem Stoß aus.
print('                      vorher     nachher')
print(f'Energie [J]:         {E_anfang:8.5f}   {E_ende:8.5f}')
print(f'Impuls x [kg m / s]: {p_anfang[0]:8.5f}   {p_ende[0]:8.5f}')
print(f'Impuls y [kg m / s]: {p_anfang[1]:8.5f}   {p_ende[1]:8.5f}')

# Erstelle eine Figure und eine Axes mit Beschriftung.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid()

# Lege die Linienplots für die Bahnkurve an.
plot_bahn1, = ax.plot([], [], '-r', zorder=4)
plot_bahn2, = ax.plot([], [], '-b', zorder=3)

# Erzeuge zwei Kreise für die Darstellung der Körper.
kreis1 = mpl.patches.Circle([0, 0], radius1, visible=False,
                            color='red', zorder=4)
kreis2 = mpl.patches.Circle([0, 0], radius2, visible=False,
                            color='blue', zorder=3)
ax.add_patch(kreis1)
ax.add_patch(kreis2)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position der beiden Körper.
    kreis1.set_center(r1[n])
    kreis2.set_center(r2[n])
    kreis1.set_visible(True)
    kreis2.set_visible(True)

    # Plotte die Bahnkurve bis zum aktuellen Zeitpunkt.
    plot_bahn1.set_data(r1[:n, 0], r1[:n, 1])
    plot_bahn2.set_data(r2[:n, 0], r2[:n, 1])
    return kreis1, kreis2, plot_bahn1, plot_bahn2


# Erstelle die Animation und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
