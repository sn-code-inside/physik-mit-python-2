"""Simulation des Vierfachpendels."""

import math
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
m4 = 1.0
# Längen der Pendelstangen [m].
l1 = 0.8
l2 = 0.4
l3 = 0.2
l4 = 0.1
# Betrag der Erdbeschleunigung [m/s²].
g = 9.81
# Parameter für die Baumgarte-Stabilisierung [1/s].
beta = alpha = 10.0

# Anfangsauslenkung [rad].
phi1 = math.radians(130.0)
phi2 = math.radians(0.0)
phi3 = math.radians(0.0)
phi4 = math.radians(0.0)

# Vektor der Anfangsposition [m].
r01 = l1 * np.array([math.sin(phi1), -math.cos(phi1)])
r02 = r01 + l2 * np.array([math.sin(phi2), -math.cos(phi2)])
r03 = r02 + l3 * np.array([math.sin(phi3), -math.cos(phi3)])
r04 = r03 + l4 * np.array([math.sin(phi4), -math.cos(phi4)])

# Array mit den Komponenten der Anfangspositionen [m].
r0 = np.concatenate((r01, r02, r03, r04))

# Anzahl der Dimensionen, der Teilchen und der Zwangsbedingungen.
n_dim = len(r01)
n_teilchen = len(r0) // n_dim
n_zwangsbed = n_teilchen

# Array mit den Komponenten der Anfangsgeschwindigkeit [m/s].
v0 = np.zeros(n_teilchen * n_dim)

# Array der Masse für jede Komponente [kg].
m = np.array([m1, m1, m2, m2, m3, m3, m4, m4])

# Vektor der Gewichtskraft.
F_g = m * np.array([0, -g, 0, -g, 0, -g, 0, -g])


def h(r):
    """Zwangsbedingungen."""
    r = r.reshape(n_teilchen, n_dim)
    d1 = r[0]
    d2 = r[1] - r[0]
    d3 = r[2] - r[1]
    d4 = r[3] - r[2]
    return np.array([d1 @ d1 - l1 ** 2,
                     d2 @ d2 - l2 ** 2,
                     d3 @ d3 - l3 ** 2,
                     d4 @ d4 - l4 ** 2])


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

    # Vierte Zwangsbedingung.
    g[3, 2] = 2 * (r[2] - r[3])
    g[3, 3] = 2 * (r[3] - r[2])

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

    # Vierte Zwangsbedingung.
    h[3, 2, :, 2, :] = 2 * E
    h[3, 2, :, 3, :] = -2 * E
    h[3, 3, :, 2, :] = -2 * E
    h[3, 3, :, 3, :] = 2 * E

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


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-6,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Zerlege den Ors- und Geschwindigkeitsvektor in die
# entsprechenden Vektoren für die vier Massen.
r1, r2, r3, r4 = np.split(r, 4)
v1, v2, v3, v4 = np.split(v, 4)

# Berechne die tatsächliche Pendellänge für jeden Zeitpunkt.
laenge1 = np.linalg.norm(r1, axis=0)
laenge2 = np.linalg.norm(r1 - r2, axis=0)
laenge3 = np.linalg.norm(r2 - r3, axis=0)
laenge4 = np.linalg.norm(r3 - r4, axis=0)

# Berechne die Gesamtenergie für jeden Zeitpunkt.
E_pot = (m1 * g * r1[1, :] + m2 * g * r2[1, :] +
         m3 * g * r3[1, :] + m4 * g * r4[1, :])
E_kin = 0
E_kin += 0.5 * m1 * np.sum(v1**2, axis=0)
E_kin += 0.5 * m2 * np.sum(v2**2, axis=0)
E_kin += 0.5 * m3 * np.sum(v3**2, axis=0)
E_kin += 0.5 * m4 * np.sum(v4**2, axis=0)
E = E_kin + E_pot

# Gib eine Tabelle der Minimal- und Maximalwerte aus.
print('      minimal        maximal')
print(f'  l1: {np.min(laenge1):.7f} m    {np.max(laenge1):.7f} m')
print(f'  l2: {np.min(laenge2):.7f} m    {np.max(laenge2):.7f} m')
print(f'  l3: {np.min(laenge3):.7f} m    {np.max(laenge3):.7f} m')
print(f'  l4: {np.min(laenge4):.7f} m    {np.max(laenge4):.7f} m')
print(f'   E: {np.min(E):.7f} J    {np.max(E):.7f} J')

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 0.4)
ax.set_aspect('equal')
ax.grid()

# Erzeuge je einen Punktplot für die Position der Massen.
plot_masse1, = ax.plot([], [], 'bo', markersize=8, zorder=5)
plot_masse2, = ax.plot([], [], 'ro', markersize=8, zorder=5)
plot_masse3, = ax.plot([], [], 'go', markersize=8, zorder=5)
plot_masse4, = ax.plot([], [], 'mo', markersize=8, zorder=5)

# Erzeuge einen Linienplot für die Stangen.
plot_stangen, = ax.plot([], [], 'k-', zorder=4)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position der Pendelkörper.
    plot_masse1.set_data(r1[:, n].reshape(-1, 1))
    plot_masse2.set_data(r2[:, n].reshape(-1, 1))
    plot_masse3.set_data(r3[:, n].reshape(-1, 1))
    plot_masse4.set_data(r4[:, n].reshape(-1, 1))

    # Aktualisiere die Position der Pendelstangen.
    p0 = np.array((0, 0))
    punkte = np.array([p0, r1[:, n], r2[:, n], r3[:, n], r4[:, n]])
    plot_stangen.set_data(punkte.T)

    return (plot_masse1, plot_masse2, plot_masse3, plot_masse4,
            plot_stangen)


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
