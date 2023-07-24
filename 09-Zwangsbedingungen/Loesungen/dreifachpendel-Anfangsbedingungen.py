"""Simultane Simulation von zwei (oder mehr) Dreifachpendeln.

Mehrere Dreifachpendel werden mit leicht unterschiedlichen
Anfangsbedingungen simuliert und zum Vergleich simultan
dargestellt.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeit und Zeitschrittweite [s].
t_max = 10
dt = 0.002
# Massen der Pendelkörpers [kg].
m1 = 1.0
m2 = 1.0
m3 = 1.0
# Länge der Pendelstangen [m].
l1 = 0.6
l2 = 0.3
l3 = 0.15
# Betrag der Erdbeschleunigung [m/s²].
g = 9.81
# Parameter für die Baumgarte-Stabilisierung [1/s].
beta = alpha = 10.0

# Anfangsauslenkungen für die beiden Simulationsläufe [rad].
phi1 = np.radians(np.array([130.0, 130.001]))
phi2 = np.radians(np.array([0.0, 0.0]))
phi3 = np.radians(np.array([0.0, 0.0]))

# Array der Anfangspositionen [m] (n_dim × n_anfangsbedingungen).
r01 = l1 * np.array([np.sin(phi1), -np.cos(phi1)])
r02 = r01 + l2 * np.array([np.sin(phi2), -np.cos(phi2)])
r03 = r02 + l3 * np.array([np.sin(phi3), -np.cos(phi3)])

# Array mit den Komponenten der Anfangspositionen [m]
# (n_teilchen⋅n_dim × n_anfangsbed).
r0 = np.concatenate((r01, r02, r03))

# Anzahl der Dimensionen, der Teilchen und der Zwangsbedingungen.
n_dim, n_anfangsbed = r01.shape
n_teilchen = r0.shape[0] // n_dim
n_zwangsbed = n_teilchen

# Array mit den Komponenten der Anfangsgeschwindigkeit [m/s]
# (n_teilchen⋅n_dim × n_anfangsbed).
v0 = np.zeros((n_teilchen * n_dim, n_anfangsbed))

# Array der Masse für jede Komponente [kg].
m = np.repeat([m1, m2, m3], n_dim)

# Vektor der Gewichtskraft.
F_g = m * np.array([0, -g, 0, -g, 0, -g])


def h(r):
    """Zwangsbedingungen."""
    r = r.reshape(n_teilchen, n_dim)
    d1 = r[0]
    d2 = r[1] - r[0]
    d3 = r[2] - r[1]
    return np.array([d1@d1 - l1**2, d2@d2 - l2**2, d3@d3 - l3**2])


def grad_h(r):
    """Gradient der Zwangsbed.: g[a, i] =  dh_a / dx_i."""
    r = r.reshape(n_teilchen, n_dim)
    g = np.zeros((n_zwangsbed, n_teilchen, n_dim))

    # Erste Zwangsbedingung.
    g[0, 0] = 2 * r[0]

    # Zweite Zwangsbedingung.
    g[1, 0] = 2 * (r[0] - r[1])
    g[1, 1] = 2 * (r[1] - r[0])

    # Dritte Zwangsbedingung.
    g[2, 1] = 2 * (r[1] - r[2])
    g[2, 2] = 2 * (r[2] - r[1])

    return g.reshape(n_zwangsbed, n_teilchen * n_dim)


def hesse_h(r):
    """Hesse-Matrix: H[a, i, j] =  d²h_a / (dx_i dx_j)."""
    h = np.zeros((n_zwangsbed,
                  n_teilchen, n_dim, n_teilchen, n_dim))

    # Erstelle eine n_dim × n_dim - Einheitsmatrix.
    E = np.eye(n_dim)

    # Erste Zwangsbedingung.
    h[0, 0, :, 0, :] = 2 * E

    # Zweite Zwangsbedingung.
    h[1, 0, :, 0, :] = 2 * E
    h[1, 0, :, 1, :] = -2 * E
    h[1, 1, :, 0, :] = -2 * E
    h[1, 1, :, 1, :] = 2 * E

    # Dritte Zwangsbedingung.
    h[2, 1, :, 1, :] = 2 * E
    h[2, 1, :, 2, :] = -2 * E
    h[2, 2, :, 1, :] = -2 * E
    h[2, 2, :, 2, :] = 2 * E

    return h.reshape(n_zwangsbed,
                     n_teilchen * n_dim, n_teilchen * n_dim)


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)

    # Berechne die lambdas.
    grad = grad_h(r)
    hesse = hesse_h(r)
    A = grad / m @ grad.T
    B = (- v @ hesse @ v - grad @ (F_g / m)
         - 2 * alpha * grad @ v - beta ** 2 * h(r))
    lam = np.linalg.solve(A, B)

    # Berechne die Beschleunigung mithilfe der newtonschen
    # Bewegungsgleichung inkl. Zwangskräften.
    a = (F_g + lam @ grad) / m

    return np.concatenate([v, a])


# Lege die Zustandsvektoren zum Zeitpunkt t=0 fest
# (2⋅n_teilchen⋅n_dim × n_anfangsbedingungen).
u0array = np.concatenate((r0, v0))

# Erzeuge Arrays für die Ergebnisse für beide Simulationsläufe
t = np.arange(0, t_max, dt)
r = np.zeros((n_anfangsbed, n_teilchen, n_dim, t.size))
v = np.zeros((n_anfangsbed, n_teilchen, n_dim, t.size))

# Führe für jeden Satz von Anfangsbedingungen eine Simulation
# durch. Dazu transponieren wir u0array, damit der erste Index
# die Anfangsbedingungen indiziert.
for i, u0 in enumerate(u0array.T):
    # u0 hat die Dimension (2⋅n_teilchen⋅n_dim).
    result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                       rtol=1e-6,
                                       t_eval=t)
    t = result.t
    r[i], v[i] = np.reshape(result.y, (2, n_teilchen, n_dim, -1))


# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 0.4)
ax.set_aspect('equal')
ax.grid()

# Erzeuge je einen Punktplot für die Position der Massen.
plot_masse1, = ax.plot([], [], 'bo', markersize=8, zorder=5)
plot_masse2, = ax.plot([], [], 'ro', markersize=8, zorder=5)
plot_masse3, = ax.plot([], [], 'go', markersize=8, zorder=5)

# Erzeuge für jede Anfangsbedingung einen Linienplot für die
# Darstellung der Pendelstangen.
plots_stangen = []
for i in range(n_anfangsbed):
    plot_stangen, = ax.plot([], [], 'k-', zorder=4)
    plots_stangen.append(plot_stangen)

# Erzeuge ein Textfeld für die Angabe der Zeit.
text_zeit = ax.text(-1.0, 0.3, '')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position der Pendelkörper.
    plot_masse1.set_data(r[:, 0, :, n].T)
    plot_masse2.set_data(r[:, 1, :, n].T)
    plot_masse3.set_data(r[:, 2, :, n].T)

    # Aktualisiere die Pendelstangen.
    for i_anfangsbed, linie in enumerate(plots_stangen):
        punkte = np.zeros((n_dim, n_teilchen + 1))
        punkte[:, 1:] = r[i_anfangsbed, :, :, n].T
        linie.set_data(punkte)

    # Aktualisiere den Text des Textfeldes.
    text_zeit.set_text(f'$t$ = {t[n]:5.2f} s')

    return plots_stangen + [plot_masse1, plot_masse2,
                            plot_masse3, text_zeit]


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
