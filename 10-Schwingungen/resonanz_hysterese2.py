"""Demonstration einer Resonanzkurve mit Hysterese (animiert)."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate
import matplotlib.animation

# Simulationsdauer [s].
t_max = 60
# Anfängliche Federhärte [N/m].
D0 = 400.0
# Masse [kg].
m = 1e-3
# Reibungskoeffizient [kg/s].
b = 0.1
# Konstante für den nichtlinearen Term [m].
alpha = 5e-3
# Anregungsamplitude [m].
amplitude = 1e-3
# Minimale und maximale Kreisfrequenz der Anregung [1/s].
omega_min = 400
omega_max = 1400

# Mittenkreisfrequenz [1/s] und Kreisfrequenzhub [1/s].
omega_0 = (omega_max + omega_min) / 2
omega_hub = (omega_max - omega_min) / 2

# Kreisfrequenz der Modulation [s].
omega_mod = 2 * 2 * np.pi / t_max


def omega_a(t):
    """Anregungskreisfrequenz als Funktion der Zeit."""
    return omega_0 - omega_hub * np.cos(omega_mod * t)


def x_a(t):
    """Anregungsfunktion."""
    phi = omega_0 * t - (
            omega_hub / omega_mod * np.sin(omega_mod * t))
    return amplitude * np.sin(phi)


def Federkraft(x):
    """Berechne die Federkraft."""
    return -D0 * alpha * np.expm1(np.abs(x)/alpha) * np.sign(x)


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    x, v = np.split(u, 2)
    F = Federkraft(x - x_a(t)) - b * v
    return np.concatenate([v, F / m])


def umkehrpunkt(t, u):
    """Ereignisfunktion: Detektiere die Extrema der Schwingung."""
    y, v = u
    return v


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.array([0, 0])

# Löse die Differentialgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-4,
                                   events=umkehrpunkt,
                                   dense_output=True)

# Bestimme die Zeitpunkte der Umkehrpunkte.
t = result.t_events[0]

# Bestimme für diese Zeitpunkte den Betrag x der Auslenkung.
x, v = result.sol(t)
x = np.abs(x)

# Berechne die jeweils aktuelle Anregungskreisfrequenz.
omega = omega_a(t)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$\\omega$ [1/s]')
ax.set_ylabel('Amplitude [m]')
ax.set_xlim(omega_min, omega_max)
ax.set_ylim(0, 1.05 * np.max(x))
ax.grid()

# Erzeuge eine Linienplot und einen Punktplot.
plot_linie, = ax.plot([], [], '-', zorder=4)
plot_punkt, = ax.plot([], [], 'or', zorder=5)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    plot_punkt.set_data(omega[n], x[n])
    plot_linie.set_data(omega[0:n + 1], x[0:n + 1])
    return plot_punkt, plot_linie


# Erzeuge das Animationsobjekt und starte die Animation.
# Wir zeigen dabei nur jeden 4-ten Schritt an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  frames=range(0, t.size, 4),
                                  interval=30, blit=True)
plt.show()
