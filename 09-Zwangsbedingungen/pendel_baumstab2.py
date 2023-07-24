"""Simulation eines ebenen Pendels mit Stabilisierung.

In diesem Programm ist ein etwas formalerer Ansatz über den
Gradienten der Zwangsbedingung gewählt worden.
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeit und Zeitschrittweite [s].
t_max = 20
dt = 0.02
# Masse des Körpers [kg].
m = 1.0
# Länge des Pendels [m].
L = 0.7
# Anfangsauslenkung [rad].
phi0 = math.radians(20.0)
# Erdbeschleunigung [m/s²].
g = 9.81
# Anfangsort [m] und Anfangsgeschwindigkeit [m/s].
r0 = L * np.array([math.sin(phi0), -math.cos(phi0)])
v0 = np.array([0, 0])
# Parameter für die Baumgarte-Stabilisierung [1/s].
beta = alpha = 10.0

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
    return np.array([[2.0, 0.0], [0.0, 2.0]])


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
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')
ax.set_xlim(-1.2 * L * math.sin(phi0), 1.2 * L * math.sin(phi0))
ax.set_ylim(-0.95, 0.05)
ax.grid()

# Erzeuge eine Punktplot für die Position des Pendelkörpers
# und je einen Linienplot für die Pendelstange und die Bahnkurve.
plot_koerper, = ax.plot([], [], 'o', color='red',
                        markersize=10, zorder=5)
plot_stange, = ax.plot([], [], '-', color='black')
plot_bahn, = ax.plot([], [], '-b')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Bahnkurve.
    plot_bahn.set_data(r[0, :n + 1], r[1, :n + 1])

    # Aktualisiere die Position des Pendelkörpers.
    plot_koerper.set_data(r[:, n])

    # Aktualisiere die Pendelstange.
    plot_stange.set_data([0, r[0, n]], [0, r[1, n]])

    return plot_stange, plot_koerper, plot_bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
