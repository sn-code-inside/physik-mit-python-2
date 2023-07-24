"""Simulation eines Kanonenschusses mit Corioliskraft."""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

# Masse des Körpers [kg].
m = 14.5
# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.45 * math.pi * 8e-2 ** 2
# Abschusswinkel [rad].
alpha = math.radians(42.0)
# Abschusshöhe [m].
h = 10.0
# Mündungsgeschwindigkeit [m/s].
betrag_v0 = 150.0
# Erdbeschleunigung [m/s²].
g = 9.81
# Luftdichte [kg/m³].
rho = 1.225
# Breitengrad [rad].
theta = math.radians(49.4)
# Betrag der Winkelgeschwindigkeit der Erde [rad/s].
betrag_omega = 7.292e-5

# Lege den Anfangsort und die Anfangsgeschwindigkeit fest.
r0 = np.array([0, 0, h])
v0 = betrag_v0 * np.array([math.cos(alpha), 0, math.sin(alpha)])

# Vektor der Winkelgeschwindigkeit [rad/s].
omega = betrag_omega * np.array(
    [0, math.cos(theta), math.sin(theta)])


def F(v):
    """Berechne die Kraft als Funktion der Geschwindigkeit v."""
    Fr = -0.5 * rho * cwA * np.linalg.norm(v) * v
    Fg = m * g * np.array([0, 0, -1])
    Fc = -2 * m * np.cross(omega, v)
    return Fg + Fr + Fc


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)
    return np.concatenate([v, F(v) / m])


def aufprall(t, u):
    """Ereignisfunktion: Detektiere das Erreichen des Erdbodens."""
    r, v = np.split(u, 2)
    return r[2]


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
fig = plt.figure(figsize=(9, 6))
fig.set_tight_layout(True)

# Plotte die Bahnkurve in der Seitenansicht.
ax_seite = fig.add_subplot(2, 1, 1)
ax_seite.tick_params(labelbottom=False)
ax_seite.set_ylabel('$z$ [m]')
ax_seite.set_aspect('equal')
ax_seite.grid()
ax_seite.plot(r_stuetz[0], r_stuetz[2], '.b')
ax_seite.plot(r_interp[0], r_interp[2], '-b')

# Plotte die Bahnkurve in der Aufsicht.
ax_aufsicht = fig.add_subplot(2, 1, 2, sharex=ax_seite)
ax_aufsicht.set_xlabel('$x$ [m]')
ax_aufsicht.set_ylabel('$y$ [m]')
ax_aufsicht.grid()
ax_aufsicht.plot(r_stuetz[0], r_stuetz[1], '.b')
ax_aufsicht.plot(r_interp[0], r_interp[1], '-b')

# Zeige die Grafik an.
plt.show()
