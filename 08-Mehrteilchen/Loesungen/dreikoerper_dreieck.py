"""Simulation von drei Körpern auf einer Kreisbahn."""

import numpy as np
import scipy.integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Konstanten: 1 Tag, 1 Jahr [s] und die Astronomische Einheit [m].
tag = 24 * 60 * 60
jahr = 365.25 * tag
AE = 1.495978707e11

# Anzahl der Körper und Raumdimension.
n_koerper = 3
n_dim = 2

# Simulationszeit und Zeitschrittweite [s].
t_max = 50 * jahr
dt = 2 * tag

# Newtonsche Gravitationskonstante [m³ / (kg * s²)].
G = 6.6743e-11

# Massen der Körper [kg].
m0 = 2e30

# Lege ein Array der Massen der Körper an, sodass wir den
# Berechnungsteil von der Sonnensystemsimulation unverändert
# übernehmen können.
m = m0 * np.ones(n_koerper)

# Abstand der Körper vom Schwerpunkt.
d = 1 * AE

# Betrag der Geschwindigkeit der Körper.
betrag_v0 = np.sqrt(G * m0 / d / np.sqrt(3))

# Die Körper werden auf ein gleichseitiges Dreieck gesetzt.
alpha = np.radians([0, 120, 240])

# Anfangspositionen der Körper
r0 = d * np.array([np.cos(alpha), np.sin(alpha)]).T

# Anfangsgeschwindigkeit der Körper.
v0 = betrag_v0 * np.array([-np.sin(alpha), np.cos(alpha)]).T

# Farben für die drei Körper.
farben = ['red', 'green', 'blue']


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)
    r = r.reshape(n_koerper, n_dim)
    a = np.zeros((n_koerper, n_dim))
    for i in range(n_koerper):
        for j in range(i):
            dr = r[j] - r[i]
            gr = G / np.linalg.norm(dr) ** 3 * dr
            a[i] += gr * m[j]
            a[j] -= gr * m[i]
    return np.concatenate([v, a.reshape(-1)])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))

# Löse die Bewegungsgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-9,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Wandle r und v in ein 3-dimensionales Array um:
#    1. Index - Himmelskörper
#    2. Index - Koordinatenrichtung
#    3. Index - Zeitpunkt
r = r.reshape(n_koerper, n_dim, -1)
v = v.reshape(n_koerper, n_dim, -1)

# Berechne die verschiedenen Energiebeiträge.
E_kin = 1/2 * m @ np.sum(v * v, axis=1)
E_pot = np.zeros(t.size)
for i in range(n_koerper):
    for j in range(i):
        dr = np.linalg.norm(r[i] - r[j], axis=0)
        E_pot -= G * m[i] * m[j] / dr
E = E_pot + E_kin
dE_rel = (np.max(E) - np.min(E)) / E[0]
print(f'Relative Energieänderung: {dE_rel:.2g}')

# Erzeuge eine Figure und eine Axes für die Animation.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [AE]')
ax.set_ylabel('$y$ [AE]')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.grid()

# Plotte für jeden Planeten die Bahnkurve.
for ort, farbe in zip(r, farben):
    ax.plot(ort[0] / AE, ort[1] / AE, '-',
            color=farbe, linewidth=0.2)

# Erzeuge für jeden Himmelskörper einen Punktplot in der
# entsprechenden Farbe und speichere diesen in der Liste.
plots_himmelskoerper = []
for farbe in farben:
    plot, = ax.plot([], [], 'o', color=farbe)
    plots_himmelskoerper.append(plot)

# Füge ein Textfeld für die Anzeige der verstrichenen Zeit hinzu.
text_zeit = ax.text(-4.5, 4.5, '')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    for plot, ort in zip(plots_himmelskoerper, r):
        plot.set_data(ort[:, n] / AE)
    text_zeit.set_text(f'{t[n] / jahr:.2f} Jahre')
    return plots_himmelskoerper + [text_zeit]


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
