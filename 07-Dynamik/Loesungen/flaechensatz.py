"""Animation des keplerschen Flächensatzes.

Es werden die Verhältnisse bei einer stark elliptischen
Planetenbahn dargestellt.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Konstanten: 1 Tag, 1 Jahr [s] und die Astronomische Einheit [m].
tag = 24 * 60 * 60
jahr = 365.25 * tag
AE = 1.495978707e11

# Simulationszeit und Zeitschrittweite [s].
t_max = 3 * jahr
dt = 0.5 * tag

# Anzahl der Zeitschritte, die für die Darstellung der Fläche
# des Fahrstrahls verwendet wird
n_zeitschritte = 40

# Anfangspositionen des Planeten [m].
r0 = np.array([152.10e9, 0.0])
v0 = np.array([0, 15e3])

# Massen der Sonne M [kg].
M = 1.9889e30

# Gravitationskonstante [m³ / (kg * s²)].
G = 6.6743e-11


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)
    a = - G * M * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-9,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [AE]')
ax.set_ylabel('$y$ [AE]')
ax.grid()

# Plotte die Bahnkurve des Himmelskörpers.
ax.plot(r[0] / AE, r[1] / AE, '-b')

# Erzeuge Punktplots für die Positionen der Himmelskörper.
plot_planet, = ax.plot([], [], 'o', color='red')
plot_sonne, = ax.plot([0], [0], 'o', color='gold')

# Erzeuge ein Polygon zur Darstellung der überstrichenen Fläche
# und füge dieses der Axes hinzu.
plot_flaeche = mpl.patches.Polygon([[0, 0], [0, 0]], closed=True,
                                   alpha=0.5, facecolor='red')
ax.add_patch(plot_flaeche)

# Erzeuge zwei Textfelder für die Angabe der verstrichenen Zeit
# und die berechnete Fläche.
text_t = ax.text(0.01, 0.95, '', color='black',
                 transform=ax.transAxes)
text_flaeche = ax.text(0.01, 0.90, '', color='black',
                       transform=ax.transAxes)


def polygon_flaeche(x, y):
    """Berechne die Fläche eines Polygons.

    Die Berechnung der Fläche erfolgt mithilfe der gaußschen
    Trapezformel.

    Args:
        x (np.ndarray):
            x-Koordinaten der Eckpunkte.
        y (np.ndarray):
            y-Koordinaten der Eckpunkte.

    Returns:
        float: Fläche des Polygons.
    """
    return 0.5 * abs((y + np.roll(y, 1)) @ (x - np.roll(x, 1)))


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position des Himmelskörpers.
    plot_planet.set_data(r[:, n] / AE)

    # Aktualisiere des Polygon und die Angabe der Fläche.
    if n >= n_zeitschritte:
        # Erzeuge ein (n_zeitschritte + 2) × 2 - Array. Als
        # ersten Punkt enthält dies die Position (0, 0) der Sonne
        # und die weiteren Punkte sind vergangenen
        # n_zeitschritte+1 Punkte der Bahnkurve des Planeten.
        xy = np.zeros((n_zeitschritte + 2, 2))
        xy[1:, :] = r[:, (n - n_zeitschritte):(n + 1)].T / AE
        plot_flaeche.set_xy(xy)

        # Berechne die Fläche des Polygons und gib diese aus.
        A = polygon_flaeche(xy[:, 0], xy[:, 1])
        text_flaeche.set_text(f'$A$ = {A:.2e} AE²')
    else:
        # Zu Beginn der Animation kann noch keine Fläche
        # dargestellt werden.
        plot_flaeche.set_xy([[0, 0], [0, 0]])
        text_flaeche.set_text('')

    # Aktualisiere das Textfeld für die Zeit.
    text_t.set_text(f'$t$ = {t[n] / tag:.0f} d')

    return plot_planet, text_t, text_flaeche, plot_flaeche


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
