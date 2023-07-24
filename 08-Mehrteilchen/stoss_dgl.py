"""Simulation des schrägen Stoßes zweier kugelförmiger Objekte.

In diesem Programm wird eine elastische Kraft zwischen den
beiden Körpern angenommen, die anfängt zu wirken, sobald sich
die Körper berühren. Die Bewegung wird über die newtonsche
Bewegungsgleichung mit solve_ivp gelöst.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeit und Zeitschrittweite [s].
t_max = 8
dt = 0.02
# Federkonstante beim Aufprall [N/m].
D = 5e3
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
v0_1 = np.array([1.0, 0.0])
v0_2 = np.array([0.0, 0.0])


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r1, r2, v1, v2 = np.split(u, 4)

    # Berechne den Abstand der Mittelpunkte.
    dr = np.linalg.norm(r1 - r2)

    # Berechne, wie weit die Kugeln ineinander eingedrungen sind.
    federweg = max(radius1 + radius2 - dr, 0)

    # Die Kraft soll proportional zum Federweg sein.
    F = D * federweg

    # Berechne die Vektoren der Beschleunigung. Der
    # Beschleunigungsvektor ist jeweils parallel zur
    # Verbindungslinie der beiden Kugelmittelpunkte.
    richtungsvektor = (r1 - r2) / dr
    a1 = F / m1 * richtungsvektor
    a2 = -F / m2 * richtungsvektor

    # Gib die Zeitableitung des Zustandsvektors zurück.
    return np.concatenate([v1, v2, a1, a2])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0_1, r0_2, v0_1, v0_2))

# Löse die Bewegungsgleichung bis zum Zeitpunkt t_max.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                   max_step=dt,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r1, r2, v1, v2 = np.split(result.y, 4)

# Berechne Energie und Gesamtimpuls vor und nach dem Stoß und
# gib diese Werte aus.
E_anfang = 1/2 * (m1 * np.sum(v1[:, 0] ** 2) +
                  m2 * np.sum(v2[:, 0] ** 2))
E_ende = 1/2 * (m1 * np.sum(v1[:, -1] ** 2) +
                m2 * np.sum(v2[:, -1] ** 2))
p_anfang = m1 * v1[:, 0] + m2 * v2[:, 0]
p_ende = m1 * v1[:, -1] + m2 * v2[:, -1]

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
    kreis1.set_center(r1[:, n])
    kreis2.set_center(r2[:, n])
    kreis1.set_visible(True)
    kreis2.set_visible(True)

    # Plotte die Bahnkurve bis zum aktuellen Zeitpunkt.
    plot_bahn1.set_data(r1[0, :n + 1], r1[1, :n + 1])
    plot_bahn2.set_data(r2[0, :n + 1], r2[1, :n + 1])
    return kreis1, kreis2, plot_bahn1, plot_bahn2


# Erstelle die Animation und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
