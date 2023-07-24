"""Simulation eines Fadenpendels mit Stabilisierung."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeit und Zeitschrittweite [s].
t_max = 20
dt = 0.005
# Masse des Körpers [kg].
m = 1.0
# Länge des Pendels [m].
L = 0.7
# Anfangsgeschwindigkeit im tiefsten Punkt [m/s].
v0 = 5.7
# Anfangsposition [m].
r0 = np.array([0.0, -L])
# Anfangsgeschwindigkeit [m/s].
v0 = np.array([v0, 0])
# Toleranz zur Erkennung des Durchhängens [m].
toleranz_r = 0.001
# Erdbeschleunigung [m/s²].
g = 9.81
# Parameter für die Baumgarte-Stabilisierung [1/s].
beta = alpha = 200.0

# Vektor der Gewichtskraft.
F_g = m * g * np.array([0, -1])


def h(r):
    """Zwangsbedingung."""
    return r @ r - L ** 2


def grad_h(r):
    """Gradient der Zwangsbedingung: g[i] = dh / dx_i."""
    return 2 * r


def hesse_h(r):
    """Hesse-Matrix: H[i, j] =  d²h / (dx_i dx_j)."""
    return 2 * np.eye(2)


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)

    # Berechne lambda.
    grad = grad_h(r)
    hesse = hesse_h(r)
    A = grad @ grad / m
    B = (- v @ hesse @ v - grad @ F_g / m
         - 2 * alpha * grad @ v - beta ** 2 * h(r))
    lam = B / A

    # Es tritt keine Zwangskraft auf, wenn diese nach außen
    # gerichtet wäre:
    lam = min(lam, 0)

    # Es tritt keine Zwangskraft auf, wenn das Seil durchhängt.
    if r @ r < (L - toleranz_r) ** 2:
        lam = 0

    # Berechne die Beschleunigung mithilfe der newtonschen
    # Bewegungsgleichung inkl. Zwangskräften.
    a = (F_g + lam * grad) / m

    # Wenn das Seil wieder straff ist und es eine
    # Geschwindigkeitskomponente nach außen gibt, dann wird
    # diese auf null gesetzt.
    if (r @ r > (L + toleranz_r) ** 2) and (grad @ v > 0):
        v -= (grad @ v) * grad / (grad @ grad)

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-6,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1.05 * L, 1.05 * L)
ax.set_ylim(-1.05 * L, 1.05 * L)
ax.set_aspect('equal')
ax.grid()

# Stelle den Kreis dar, der die Zwangsbedingung repräsentiert.
kreis_pendelbahn = mpl.patches.Circle([0, 0], L,
                                      fill=False, zorder=2)
ax.add_patch(kreis_pendelbahn)

# Erzeuge eine Punktplot für die Position des Pendelkörpers,
# einen Linienplot für die Stange und einen Linienplot für die
# Bahnkurve.
plot_koerper, = ax.plot([], [], 'o', color='red', markersize=10,
                        zorder=5)
plot_stange, = ax.plot([], [], '-', color='black',
                       zorder=4)
plot_bahn, = ax.plot([], [], '-b', zorder=3)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position des Pendels.
    plot_stange.set_data([0, r[0, n]], [0, r[1, n]])
    plot_koerper.set_data(r[:, n])

    # Stelle die Bahnkurve bis zum aktuellen Zeitpunkt dar.
    plot_bahn.set_data(r[:, :n + 1])

    # Färbe die Pendelstange hellgrau, wenn das Pendel durchhängt.
    if np.linalg.norm(r[:, n]) < L - toleranz_r:
        plot_stange.set_color('lightgray')
    else:
        plot_stange.set_color('black')

    return plot_stange, plot_koerper, plot_bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
