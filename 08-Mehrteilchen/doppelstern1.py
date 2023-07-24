"""Simulation eines Doppelsternsystems."""

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
# [AE / (m/s²)].
scal_a = 20

# Simulationszeit und Zeitschrittweite [s].
t_max = 2 * jahr
dt = 1 * tag

# Massen der beiden Sterne [kg].
m1 = 2.0e30
m2 = 4.0e29

# Anfangspositionen der Sterne [m].
r0_1 = AE * np.array([0.0, 0.0])
r0_2 = AE * np.array([0.0, 1.0])

# Anfangsgeschwindigkeiten der Sterne [m/s].
v0_1 = np.array([0.0, 0.0])
v0_2 = np.array([25e3, 0.0])

# Gravitationskonstante [m³ / (kg * s²)].
G = 6.6743e-11


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r1, r2, v1, v2 = np.split(u, 4)
    a1 = G * m2 / np.linalg.norm(r2 - r1)**3 * (r2 - r1)
    a2 = G * m1 / np.linalg.norm(r1 - r2)**3 * (r1 - r2)
    return np.concatenate([v1, v2, a1, a2])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0_1, r0_2, v0_1, v0_2))

# Löse die Bewegungsgleichung bis zum Zeitpunkt t_max.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-9,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r1, r2, v1, v2 = np.split(result.y, 4)

# Berechne die verschiedenen Energiebeiträge.
E_kin1 = 1/2 * m1 * np.sum(v1 ** 2, axis=0)
E_kin2 = 1/2 * m2 * np.sum(v2 ** 2, axis=0)
E_pot = - G * m1 * m2 / np.linalg.norm(r1 - r2, axis=0)

# Berechne den Gesamtimpuls.
impuls = m1 * v1 + m2 * v2

# Berechne den Drehimpuls.
drehimpuls = (m1 * np.cross(r1, v1, axis=0) +
              m2 * np.cross(r2, v2, axis=0))

# Erzeuge eine Figure.
fig = plt.figure(figsize=(10, 7))
fig.set_tight_layout(True)

# Erzeuge eine Axes für die Bahnkurve der Sterne.
ax_bahn = fig.add_subplot(2, 2, 1)
ax_bahn.set_xlabel('$x$ [AE]')
ax_bahn.set_ylabel('$y$ [AE]')
ax_bahn.set_aspect('equal')
ax_bahn.grid()

# Plotte die Bahnkurven der Sterne.
ax_bahn.plot(r1[0] / AE, r1[1] / AE, '-r')
ax_bahn.plot(r2[0] / AE, r2[1] / AE, '-b')

# Erzeuge Punktplots für die Positionen der Sterne.
plot_stern1, = ax_bahn.plot([], [], 'o', color='red')
plot_stern2, = ax_bahn.plot([], [], 'o', color='blue')

# Erzeuge zwei Pfeile für die Beschleunigungsvektoren.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_a1 = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red',
                                       arrowstyle=style)
pfeil_a2 = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='blue',
                                       arrowstyle=style)

# Füge die Pfeile zur Axes hinzu.
ax_bahn.add_patch(pfeil_a1)
ax_bahn.add_patch(pfeil_a2)

# Erzeuge eine Axes und plotte die Energie.
ax_energ = fig.add_subplot(2, 2, 2)
ax_energ.set_title('Energie')
ax_energ.set_xlabel('$t$ [d]')
ax_energ.set_ylabel('$E$ [J]')
ax_energ.grid()
ax_energ.plot(t / tag, E_kin1, '-r', label='$E_{kin,1}$')
ax_energ.plot(t / tag, E_kin2, '-b', label='$E_{kin,2}$')
ax_energ.plot(t / tag, E_pot, '-c', label='$E_{pot}$')
ax_energ.plot(t / tag, E_pot + E_kin1 + E_kin2,
              '-k', label='$E_{ges}$')
ax_energ.legend()

# Erzeuge eine Axes und plotte den Drehimpuls.
ax_drehimpuls = fig.add_subplot(2, 2, 3)
ax_drehimpuls.set_title('Drehimpuls')
ax_drehimpuls.set_xlabel('$t$ [d]')
ax_drehimpuls.set_ylabel('$L$ [kg m² / s]')
ax_drehimpuls.grid()
ax_drehimpuls.plot(t / tag, drehimpuls)

# Erzeuge eine Axes und plotte den Impuls.
ax_impuls = fig.add_subplot(2, 2, 4)
ax_impuls.set_title('Impuls')
ax_impuls.set_xlabel('$t$ [d]')
ax_impuls.set_ylabel('$p$ [kg m / s]')
ax_impuls.grid()
ax_impuls.plot(t / tag, impuls[0, :], label='$p_x$')
ax_impuls.plot(t / tag, impuls[1, :], label='$p_y$')
ax_impuls.legend()

# Sorge dafür, dass die nachfolgenden Linien nicht mehr die
# y-Skalierung verändern.
ax_energ.set_ylim(auto=False)
ax_drehimpuls.set_ylim(auto=False)
ax_impuls.set_ylim(auto=False)

# Erzeuge drei schwarze Linien, die die aktuelle Zeit in den
# Plots für Energie, Impuls und Drehimpuls darstellen.
linie_t_energ, = ax_energ.plot([], [], '-k')
linie_t_drehimp, = ax_drehimpuls.plot([], [], '-k')
linie_t_impuls, = ax_impuls.plot([], [], '-k')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position der Sterne.
    plot_stern1.set_data(r1[:, n].reshape(-1, 1) / AE)
    plot_stern2.set_data(r2[:, n].reshape(-1, 1) / AE)

    # Berechne die Momentanbeschleunigung und aktualisiere die
    # Vektorpfeile.
    u = np.concatenate([r1[:, n], r2[:, n], v1[:, n], v2[:, n]])
    a_1, a_2 = np.split(dgl(t[n], u), 4)[2:]
    pfeil_a1.set_positions(r1[:, n] / AE,
                           r1[:, n] / AE + scal_a * a_1)
    pfeil_a2.set_positions(r2[:, n] / AE,
                           r2[:, n] / AE + scal_a * a_2)

    # Stelle die Zeit in den drei anderen Diagrammen dar.
    x_pos = t[n] / tag
    linien = [linie_t_energ, linie_t_drehimp, linie_t_impuls]
    for linie in linien:
        linie.set_data([[x_pos, x_pos], linie.axes.get_ylim()])

    return linien + [plot_stern1, plot_stern2, pfeil_a1, pfeil_a2]


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
