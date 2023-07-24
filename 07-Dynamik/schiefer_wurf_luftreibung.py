"""Simulation eines fliegenden Balls mit Luftreibung."""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Skalierungsfaktoren für den Geschwindigkeitsvektor [1/s]
# und Beschleunigungsvektor [1/s²].
scal_v = 0.1
scal_a = 0.1

# Masse des Körpers [kg].
m = 2.7e-3
# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.45 * math.pi * 20e-3 ** 2
# Abwurfwinkel [rad].
alpha = math.radians(40.0)
# Abwurfhöhe [m].
h = 1.1
# Betrag der Abwurfgeschwindigkeit [m/s].
betrag_v0 = 20
# Erdbeschleunigung [m/s²].
g = 9.81
# Luftdichte [kg/m³].
rho = 1.225

# Lege den Anfangsort und die Anfangsgeschwindigkeit fest.
r0 = np.array([0, h])
v0 = betrag_v0 * np.array([math.cos(alpha), math.sin(alpha)])


def F(v):
    """Berechne die Kraft als Funktion der Geschwindigkeit v."""
    Fr = -0.5 * rho * cwA * np.linalg.norm(v) * v
    Fg = m * g * np.array([0, -1])
    return Fg + Fr


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)
    return np.concatenate([v, F(v) / m])


def aufprall(t, u):
    """Ereignisfunktion: Detektiere das Erreichen des Erdbodens."""
    r, v = np.split(u, 2)
    return r[1]


# Beende die Simulation beim Auftreten des Ereignisses.
aufprall.terminal = True

# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde.
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                   events=aufprall,
                                   dense_output=True)
t_stuetz = result.t
r_stuetz, v_stuetz = np.split(result.y, 2)

# Berechne die Interpolation auf einem feinen Raster.
t_interp = np.linspace(0, np.max(t_stuetz), 1000)
r_interp, v_interp = np.split(result.sol(t_interp), 2)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(9, 4))

# Plotte die Bahnkurve.
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')
ax.grid()
ax.plot(r_stuetz[0], r_stuetz[1], '.b')
ax.plot(r_interp[0], r_interp[1], '-b')

# Erzeuge einen Punktplot für die Position des Balles.
plot_ball, = ax.plot([], [], 'o', color='red', zorder=4)

# Erzeuge Pfeile für die Geschwindigkeit und die Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style, zorder=3)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='black',
                                      arrowstyle=style, zorder=3)

# Füge die Grafikobjekte zur Axes hinzu.
ax.add_patch(pfeil_v)
ax.add_patch(pfeil_a)

# Erzeuge Textfelder für die Anzeige des aktuellen
# Geschwindigkeits- und Beschleunigungsbetrags.
text_t = ax.text(2.1, 1.5, '', color='blue')
text_v = ax.text(2.1, 1.1, '', color='red')
text_a = ax.text(2.1, 0.7, '', color='black')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    t = t_interp[n]
    r = r_interp[:, n]
    v = v_interp[:, n]

    # Berechne die Momentanbeschleunigung.
    a = F(v_interp[:, n]) / m

    # Aktualisiere die Position des Balls.
    plot_ball.set_data(r)

    # Aktualisiere die Pfeile für Geschw. und Beschleunigung.
    pfeil_v.set_positions(r, r + scal_v * v)
    pfeil_a.set_positions(r, r + scal_a * a)

    # Aktualisiere die Textfelder.
    text_t.set_text(f'$t$ = {t:.2f} s')
    text_v.set_text(f'$v$ = {np.linalg.norm(v):.1f} m/s')
    text_a.set_text(f'$a$ = {np.linalg.norm(a):.1f} m/s²')

    return plot_ball, pfeil_v, pfeil_a, text_v, text_a, text_t


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t_interp.size,
                                  interval=30, blit=True)
plt.show()
