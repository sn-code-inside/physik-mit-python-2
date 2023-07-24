"""Simulation einer stark elliptischen Planetenbahn."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Konstanten: 1 Tag, 1 Jahr [s] und die Astronomische Einheit [m].
tag = 24 * 60 * 60
jahr = 365.25 * tag
AE = 1.495978707e11

# Skalierungsfaktor für die Darstellung der Beschleunigung
# [AE / (m/s²)] und Geschwindigkeit [AE / (m/s)].
scal_a = 20
scal_v = 1e-5

# Simulationszeit und Zeitschrittweite [s].
t_max = 1 * jahr
dt = 1 * tag

# Anfangsposition [m] und -Geschwindigkeit des Planeten [m].
r0 = np.array([152.10e9, 0.0])
v0 = np.array([0.0, 15e3])

# Masse der Sonne M [kg].
M = 1.9885e30

# Gravitationskonstante [m³ / (kg * s²)].
G = 6.6743e-11


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)
    a = - G * M * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-9,
                                   dense_output=True)
t_stuetz = result.t
r_stuetz, v_stuetz = np.split(result.y, 2)

# Berechne die Interpolation auf einem feinen Raster.
t_interp = np.arange(0, np.max(t_stuetz), dt)
r_interp, v_interp = np.split(result.sol(t_interp), 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [AE]')
ax.set_ylabel('$y$ [AE]')
ax.set_xlim(-0.2, 1.1)
ax.set_ylim(-0.6, 0.6)
ax.grid()

# Plotte die Bahnkurve der Himmelskörper.
ax.plot(r_stuetz[0] / AE, r_stuetz[1] / AE, '.b')
ax.plot(r_interp[0] / AE, r_interp[1] / AE, '-b')

# Erzeuge Punktplots für die Positionen der Himmelskörper.
plot_planet, = ax.plot([], [], 'o', color='red')
plot_sonne, = ax.plot([0], [0], 'o', color='gold')

# Erzeuge Pfeile für die Geschwindigkeit und die Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red',
                                      arrowstyle=style)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='black',
                                      arrowstyle=style)

# Füge die Pfeile zur Axes hinzu.
ax.add_patch(pfeil_a)
ax.add_patch(pfeil_v)

# Füge ein Textfeld für die Angabe der verstrichenen Zeit hinzu.
text_t = ax.text(0.01, 0.95, '', color='blue',
                 transform=ax.transAxes)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    t = t_interp[n]
    r = r_interp[:, n]
    v = v_interp[:, n]

    # Berechne die aktuelle Beschleunigung.
    u_punkt = dgl(t, np.concatenate([r, v]))
    a = np.split(u_punkt, 2)[1]

    # Aktualisiere die Position des Himmelskörpers und die Pfeile.
    plot_planet.set_data(r / AE)
    pfeil_a.set_positions(r / AE, r / AE + scal_a * a)
    pfeil_v.set_positions(r / AE, r / AE + scal_v * v)

    # Aktualisiere das Textfeld für die Zeit.
    text_t.set_text(f'$t$ = {t / tag:.0f} d')

    return plot_planet, pfeil_v, pfeil_a, text_t


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t_interp.size,
                                  interval=30, blit=True)
plt.show()
