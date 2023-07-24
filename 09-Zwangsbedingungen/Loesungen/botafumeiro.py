"""Simulation des Botafumeiro."""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Simulationszeit und Zeitschrittweite [s].
t_max = 1000
dt = 0.5
# Masse des Pendelkörpers [kg].
m = 50.0
# Anfangslänge des Pendels [m].
l0 = 30.0
# Amplitude der Längenvariation [m].
l1 = 0.2
# Betrag der Erdbeschleunigung [m/s²].
g = 9.81
# Kreisfrequenz der Anregung.
omega = 2 * np.sqrt(g / l0)
# Anfangsauslenkung [rad].
phi0 = math.radians(1.0)
# Parameter für die Baumgarte-Stabilisierung [1/s].
beta = alpha = 10.0

# Angansposition [m].
r0 = l0 * np.array([math.sin(phi0), -math.cos(phi0)])

# Anfangsgeschwindigkeit [m/s].
v0 = np.array([0.0, 0.0])

# Vektor der Gewichtskraft.
F_g = m * g * np.array([0, -1])


def L(t):
    """Pendellänge als Funktion der Zeit."""
    return l0 + l1 * np.sin(omega * t)


def dt_L(t):
    """Zeitableitung dL/dt."""
    return l1 * omega * np.cos(omega * t)


def d2t_L(t):
    """Zweite Zeitableitung d²L/dt²."""
    return -l1 * omega**2 * np.sin(omega * t)


def h(t, r):
    """Zwangsbedingung."""
    return r @ r - L(t)**2


def dt_h(t, r):
    """Partielle Zeitableitung der Zwangsbedingung dh / dt."""
    return - 2 * L(t) * dt_L(t)


def d2t_h(t, r):
    """Zweite partielle Zeitableitung der Zwangsbed. d²h / dt²."""
    return -2 * dt_L(t)**2 - 2 * L(t) * d2t_L(t)


def grad_h(t, r):
    """Gradient der Zwangsbedingung: g[i] = dh / dx_i."""
    return 2 * r


def dtgrad_h(t, r):
    """Partielle Zeitableitung des Gradienten: dg[i] / dt."""
    return np.zeros_like(r)


def hesse_h(t, r):
    """Hesse-Matrix: H[i, j] =  d²h / (dx_i dx_j)."""
    return np.array([[2.0, 0.0], [0.0, 2.0]])


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)

    # Berechne lambda unter Berücksichtigung der zusätzlichen
    # Terme bei explizit zeitabhängigen Zwangsbedingungen.
    grad = grad_h(t, r)
    hesse = hesse_h(t, r)
    A = grad @ grad / m
    B = (- v @ hesse @ v - grad @ F_g / m
         - 2 * alpha * grad @ v - beta ** 2 * h(t, r)
         - 2 * alpha * dt_h(t, r)
         - dtgrad_h(t, r) @ v - d2t_h(t, r))
    lam = B / A

    # Berechne die Zwangskraft und die Beschleunigung.
    F_zwang = lam * grad
    a = (F_zwang + F_g) / m

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-6,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$t$ [s]')
ax.set_ylabel('$y$ [m]')
ax.grid()

# Plotte die x-Koordinate des Pendels als Funktion der Zeit.
ax.plot(t, r[0])

# Zeige die Grafik an.
plt.show()
