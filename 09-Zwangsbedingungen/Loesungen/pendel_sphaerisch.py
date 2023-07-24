"""Simulation eines sphärischen Pendels."""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import mpl_toolkits.mplot3d

# Simulationszeit und Zeitschrittweite [s].
t_max = 100.0
dt = 0.02
# Masse des Körpers [kg].
m = 1.0
# Länge des Pendels [m].
L = 0.7
# Anfangsauslenkung [rad].
phi0 = math.radians(40.0)
# Anfangsgeschwindigkeit [m/s].
v0 = 0.4
# Anfangsort [m] und Anfangsgeschwindigkeit [m/s].
r0 = np.array([L * math.sin(phi0), 0, -L * math.cos(phi0)])
# Vektor der Anfangsgeschwindigkeit [m/s].
v0 = np.array([0, v0, 0])
# Erdbeschleunigung [m/s²].
g = 9.81
# Parameter für die Baumgarte-Stabilisierung.
beta = alpha = 10.0

# Vektor der Gewichtskraft.
F_g = np.array([0, 0, -m * g])


def h(r):
    """Zwangsbedingung."""
    return r @ r - L ** 2


def grad_h(r):
    """Gradient der Zwangsbedingung: g[i] = dh / dx_i."""
    return 2 * r


def hesse_h(r):
    """Hesse-Matrix: H[i, j] =  d²h / (dx_i dx_j)."""
    return 2 * np.eye(3)


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

    # Berechne die Zwangskraft und die Beschleunigung.
    F_zwang = lam * grad
    a = (F_zwang + F_g) / m

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d', elev=30, azim=45)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_zlabel('$z$ [m]')
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_zlim(-L, 0)
ax.grid()

# Erzeuge eine Punktplot für die Position des Pendelkörpers,
# einen Linienplot für die Stange und einen Linienplot für die
# Bahnkurve.
plot_koerper, = ax.plot([], [], [], 'o', color='red',
                        markersize=10, zorder=5)
plot_stange, = ax.plot([], [], [], '-', color='black',
                       zorder=4)
plot_bahn, = ax.plot([], [], [], '-b', zorder=3)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Bahnkurve.
    plot_bahn.set_data_3d(r[:, :n + 1])

    # Aktualisiere die Position des Pendelkörpers.
    plot_koerper.set_data_3d(r[:, n])

    # Aktualisiere die Pendelstange.
    plot_stange.set_data_3d([0, r[0, n]],
                            [0, r[1, n]],
                            [0, r[2, n]])

    return plot_koerper, plot_stange, plot_bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
