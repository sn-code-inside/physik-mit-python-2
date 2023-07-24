"""Simulation eines Swing-by-Manövers."""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Konstanten: 1 Stunde, 1 Tag und 1 Jahr [s].
stunde = 60 * 60
tag = 24 * stunde
jahr = 365.25 * tag

# Simulationszeit und Zeitschrittweite [s].
t_max = 36 * tag
dt = 1 * stunde

# Masse des Planeten und der Raumsonde [kg].
m_planet = 1.898e27
m_sonde = 1e3

# Anflugwinkel der Raumsonde relativ zur Bewegungsrichtung
# des Planeten.
alpha = math.radians(60)

# Anfangsentfernung der Körper vom Koordinatenursprung.
abstand_sonde = 15e9
abstand_planet = 20.18e9

# Radius des Planeten.
radius_planet = 6.9911e7

# Anfangsgeschwindigkeiten der Körper.
betrag_v0_sonde = 9e3
betrag_v0_planet = 13e3

# Anfangspositionen der Körper [m].
r0_planet = np.array([abstand_planet, 0.0])
r0_sonde = abstand_sonde * np.array([-np.cos(alpha),
                                     -np.sin(alpha)])

# Gravitationskonstante [m³ / (kg * s²)].
G = 6.6743e-11

# Anfangsgeschwindigkeiten der Körper [m/s].
v0_planet = betrag_v0_planet * np.array([-1.0, 0.0])
v0_sonde = betrag_v0_sonde * np.array([np.cos(alpha),
                                       np.sin(alpha)])


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r1, r2, v1, v2 = np.split(u, 4)
    a1 = G * m_sonde / np.linalg.norm(r2 - r1) ** 3 * (r2 - r1)
    a2 = G * m_planet / np.linalg.norm(r1 - r2) ** 3 * (r1 - r2)
    return np.concatenate([v1, v2, a1, a2])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0_planet, r0_sonde, v0_planet, v0_sonde))

# Löse die Bewegungsgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-9,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r_sonde, r_planet, v_sonde, v_planet = np.split(result.y, 4)

# Berechne den Abstand der Raumsonde vom Planeten.
abstand = np.linalg.norm(r_sonde - r_planet, axis=0)

# Gibt die minimale Entfernung der Raumsonde vom Planeten in
# Vielfachen des Radius des Planeten an.
abstand_min = np.min(abstand)
print(f"Min. Abstand: {abstand_min / radius_planet:.1f} "
      "Planetenradien")

# Berechne den Geschwindigkeitsbetrag der Raumsonde.
betrag_v_sonde = np.linalg.norm(v_planet, axis=0)

# Erzeuge eine Figure und eine Axes für die Animation.
fig_anim = plt.figure()
fig_anim.set_tight_layout(True)
ax_anim = fig_anim.add_subplot(1, 1, 1)
ax_anim.set_xlabel('$x$ [m]')
ax_anim.set_ylabel('$y$ [m]')
ax_anim.grid()

# Plotte die Bahnkurve der beiden Körper.
ax_anim.plot(r_sonde[0], r_sonde[1], '-r')
ax_anim.plot(r_planet[0], r_planet[1], '-b')

# Erzeuge eine zweite Figure für die Plots.
fig_plot = plt.figure()
fig_plot.set_tight_layout(True)

# Erzeuge eine Axes für den Abstand als Funktion der Zeit.
ax_abstand = fig_plot.add_subplot(1, 2, 1)
ax_abstand.set_xlabel("$t$ [Tage]")
ax_abstand.set_ylabel("Abstand [m]")
ax_abstand.grid()
ax_abstand.plot(t / tag, abstand)

# Erzeuge eine Axes für die Geschwindigkeit der Raumsonde.
ax_geschwindigkeit = fig_plot.add_subplot(1, 2, 2)
ax_geschwindigkeit.set_xlabel("$t$ [Tage]")
ax_geschwindigkeit.set_ylabel("Geschwindigkeit [km/s]")
ax_geschwindigkeit.grid()
ax_geschwindigkeit.plot(t / tag, betrag_v_sonde / 1e3)

# Erzeuge Punktplots für die Positionen der Himmelskörper.
plot_sonde, = ax_anim.plot([], [], 'o', color='red')
plot_planet, = ax_anim.plot([], [], 'o', color='blue')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    plot_sonde.set_data(r_sonde[:, n])
    plot_planet.set_data(r_planet[:, n])
    return plot_sonde, plot_planet


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig_anim, update, frames=t.size,
                                  interval=40, blit=True)
plt.show()
