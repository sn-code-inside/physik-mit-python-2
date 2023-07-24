"""Simulation der Bewegung einer Kette mit Reibung.

Die Kette besteht aus 11 Punktmassen und es wird eine
zur Geschwindigkeit proportionale Reibungskraft angenommen.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeit und Zeitschrittweite [s].
t_max = 20
dt = 0.01
# Reibungskonstante [kg/s].
b = 1.5
# Masse jedes Teilchens [kg].
masse = 1.0
# Betrag der Erdbeschleunigung [m/s²].
g = 9.81
# Parameter für die Baumgarte-Stabilisierung [1/s].
beta = alpha = 10.0

# Lege die Anfangspositionen der Punkte fest [m].
punkte = np.array([[0.00,  0.00], [0.00, -0.25], [0.00, -0.50],
                   [0.25, -0.50], [0.50, -0.50], [0.75, -0.50],
                   [1.00, -0.50], [1.25, -0.50], [1.50, -0.50],
                   [1.75, -0.50], [2.00, -0.50], [2.00, -0.25],
                   [2.00,  0.0]])

# Definiere die Dimension sowie die Anzahl der Punkte, Stäbe, etc.
n_punkte, n_dim = punkte.shape
n_staebe = n_punkte - 1
n_stuetz = 2
n_knoten = n_punkte - n_stuetz

# Massen der einzelnen Körper [kg].
# Der Wert der Masse für die Stützpunkte ist irrelevant.
massen = masse * np.ones(n_punkte)

# Machde den ersten und den letzten Punkt zu Stützpunkten.
indizes_stuetz = [0, n_punkte - 1]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab. Hier wird
# jeder Punkt mit seinem Nachfolger verbunden.
staebe = []
for i in range(n_punkte - 1):
    staebe.append([i, i+1])
staebe = np.array(staebe)

# Berechne die Länge der Stäbe aus den Anfangspositionen.
laengen = np.empty(n_staebe)
for i, stab in enumerate(staebe):
    r1, r2 = punkte[stab]
    laengen[i] = np.linalg.norm(r1 - r2)

# Erzeuge eine Liste mit den Indizes der Knoten.
indizes_knoten = list(set(range(n_punkte)) - set(indizes_stuetz))

# Array mit den Komponenten der Anfangspositionen der beweglichen
# Massen [m].
r0 = punkte[indizes_knoten].reshape(-1)

# Array mit den Komponenten der Anfangsgeschwindigkeit [m/s].
v0 = np.zeros(n_knoten * n_dim)

# Array der Massen für jede Koordinate [kg].
m = np.repeat(massen[indizes_knoten], n_dim)

# Gewichtskraft in -y-Richtung.
F_g = np.zeros((n_knoten, n_dim))
F_g[:, 1] = -g
F_g = m * F_g.reshape(-1)


def h(r):
    """Zwangsbedingungen."""
    # Erzeuge ein Array mit den aktuellen Positionen aller
    # Punkte, wobei die Positionen der beweglichen Massen aus dem
    # Array r übernommen werden.
    punkte_akt = punkte.copy()
    punkte_akt[indizes_knoten] = r.reshape(n_knoten, n_dim)

    # Die Zwangsbedingungen legen fest, dass die Länge jedes
    # Stabes konstant ist.
    h = np.zeros(n_staebe)
    for i, stab in enumerate(staebe):
        ra, rb = punkte_akt[stab]
        h[i] = (ra - rb) @ (ra - rb) - laengen[i] ** 2
    return h


def grad_h(r):
    """Gradient der Zwangsbed.: g[a, i] =  dh_a / dx_i."""
    g = np.zeros((n_staebe, n_knoten, n_dim))
    # Erzeuge ein Array mit den Positionen aller Punkte, wobei
    # die Positionen der beweglichen Massen aus dem Array r
    # übernommen werden.
    punkte_aktuell = punkte.copy()
    punkte_aktuell[indizes_knoten] = r.reshape(n_knoten, n_dim)

    # In der Zwangsbedingung taucht der quadratische Abstand
    # der durch einen Stab verbundenen Punkte auf. In der
    # Ableitung steht daher jeweils die doppelte Differenz der
    # Ortsvektoren.
    for i, stab in enumerate(staebe):
        dr = punkte_aktuell[stab[0]] - punkte_aktuell[stab[1]]
        if stab[0] in indizes_knoten:
            k = indizes_knoten.index(stab[0])
            g[i, k] += 2 * dr
        if stab[1] in indizes_knoten:
            k = indizes_knoten.index(stab[1])
            g[i, k] -= 2 * dr

    return g.reshape(n_staebe, n_knoten * n_dim)


def hesse_h(r):
    """Hesse-Matrix: H[a, i, j] =  d²h_a / (dx_i dx_j)."""
    h = np.zeros((n_staebe,
                  n_knoten, n_dim, n_knoten, n_dim))

    # Erstelle eine n_dim × n_dim - Einheitsmatrix.
    E = np.eye(n_dim)

    # Für die Hesse-Matrix muss ganz analog zum Beispiel mit dem
    # Drei- oder Vierfachpendel an den entsprechenden Stellen
    # eine 2 bzw. -2 eingetragen werden.
    for i, stab in enumerate(staebe):
        if stab[0] in indizes_knoten:
            k1 = indizes_knoten.index(stab[0])
            h[i, k1, :, k1, :] = 2 * E
        if stab[1] in indizes_knoten:
            k2 = indizes_knoten.index(stab[1])
            h[i, k2, :, k2, :] = 2 * E
        if ((stab[0] in indizes_knoten)
                and (stab[1] in indizes_knoten)):
            h[i, k1, :, k2, :] = -2 * E
            h[i, k2, :, k1, :] = -2 * E

    return h.reshape(n_staebe,
                     n_knoten * n_dim, n_knoten * n_dim)


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)

    # Berechne die lambdas.
    grad = grad_h(r)
    hesse = hesse_h(r)
    A = grad / m @ grad.T
    B = (- v @ hesse @ v - grad @ (F_g / m)
         - 2 * alpha * grad @ v - beta ** 2 * h(r))
    lam = np.linalg.solve(A, B)

    # Berücksichtige die Reibungskraft.
    F_reib = - b * v

    # Berechne die Beschleunigung mithilfe der newtonschen
    # Bewegungsgleichung inkl. Zwangskräften.
    a = (F_g + F_reib + lam @ grad) / m

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-6,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-1.5, 0.5)
ax.set_aspect('equal')
ax.grid()

# Plotte die Stützpunkte in Rot.
plot_stuetz, = ax.plot(punkte[indizes_stuetz, 0],
                       punkte[indizes_stuetz, 1], 'ro', zorder=5)

# Lege einen Punktplot für die Kontenpunkte in Blau an.
plot_knoten, = ax.plot([], [], 'bo', zorder=5)

# Lege Linienplots für die Stäbe an.
plots_stab = []
for stab in staebe:
    s, = ax.plot([], [], color='black', zorder=4)
    plots_stab.append(s)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Erzeuge ein Array mit den aktuellen Positionen aller
    # Punkte.
    punkt_akt = punkte.copy()
    punkt_akt[indizes_knoten] = r[:, n].reshape(n_knoten, n_dim)

    # Aktualisiere die Position der Massen.
    plot_knoten.set_data(punkt_akt[indizes_knoten, 0],
                         punkt_akt[indizes_knoten, 1])

    # Aktualisiere die Stäbe.
    for n, stab in enumerate(staebe):
        plots_stab[n].set_data(punkt_akt[stab, 0],
                               punkt_akt[stab, 1])

    return plots_stab + [plot_knoten]


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
