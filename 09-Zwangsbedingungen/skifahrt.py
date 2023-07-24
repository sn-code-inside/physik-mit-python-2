"""Simulation einer Skifahrt."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import scipy.interpolate

# Stützstellen (Koordinaten) des Hangs [m].
x_hang = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 35.0,
                   40.0, 45.0, 55.0, 70.0])
y_hang = np.array([10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0,
                   3.5, 1.5, 0.02, 0.0])

# Zeitschrittweite für die Animation [s].
dt = 0.01
# Masse des Skifahrers [kg].
m = 90.0
# Erdbeschleunigung [m/s²].
g = 9.81
# Luftdichte [kg/m³].
rho = 1.3
# Produkt aus cw-Wert und Frontfläche [m²].
cwA = 0.47
# Gleitreibungskoeffizient.
mu = 0.02
# Parameter für die Baumgarte-Stabilisierung [1/s].
beta = alpha = 20.0

# Toleranz in vertikaler Richtung zur Erkennung eines Aufpralls
# oder eines Ablösens von der Hangoberfläche [m].
toleranz_y = 0.001

# Vektor der Gewichtskraft.
F_g = m * g * np.array([0, -1])

# Anfangsposition [m] und Vektor der Anfangsgeschwindigkeit [m/s].
r0 = np.array([x_hang[0], y_hang[0]])
v0 = np.array([0, 0])

# Erzeuge eine Interpolation der Stützpunkte des Hangs mit
# kubischen Splines und bilde die ersten beiden Ableitungen.
hangfunktion = scipy.interpolate.CubicSpline(x_hang, y_hang,
                                             bc_type='natural')
d_hangfunktion = hangfunktion.derivative(1)
d2_hangfunktion = hangfunktion.derivative(2)


def h(r):
    """Zwangsbedingung."""
    x, y = r
    return hangfunktion(x) - y


def grad_h(r):
    """Gradient der Zwangsbedingung: g[i] = dh / dx_i."""
    x, y = r
    return np.array([d_hangfunktion(x), -1])


def hesse_h(r):
    """Hesse-Matrix: H[i, j] =  d²h / (dx_i dx_j)."""
    x, y = r
    return np.array([[d2_hangfunktion(x), 0], [0, 0]])


def zwangskraft(r, v):
    """Berechne den Vektor der Zwangskraft."""
    # Berechne lambda.
    grad = grad_h(r)
    hesse = hesse_h(r)
    A = grad @ grad / m
    B = (- v @ hesse @ v - grad @ F_g / m
         - 2 * alpha * grad @ v - beta ** 2 * h(r))
    lam = B / A

    # Es tritt keine Zwangskraft auf, wenn diese in den Hang
    # hinein gerichtet wäre.
    lam = min(lam, 0)

    # Es tritt keine Zwangskraft auf, wenn der Skifahrer den
    # Hang gar nicht mehr berührt.
    if r[1] > hangfunktion(r[0]) + toleranz_y:
        lam = 0

    # Berechne den Vektor der Zwangskraft (Normalkraft).
    return lam * grad


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)

    # Berechne die Normalkraft, die vom Skihang ausgeübt wird.
    F_N = zwangskraft(r, v)

    # Berechne die Gleitreibungskraft.
    if np.linalg.norm(v) > 0:
        F_r = -mu * np.linalg.norm(F_N) * v / np.linalg.norm(v)
    else:
        F_r = 0.0

    # Berechne die Luftwiderstandskraft.
    F_luft = -0.5 * rho * cwA * v * np.linalg.norm(v)

    # Berechne die Beschleunigung mithilfe der newtonschen
    # Bewegungsgleichung inkl. Zwangskräften.
    a = (F_g + F_luft + F_r + F_N) / m

    # Wenn der Skifahrer in den Hang eindringt, dann wird die
    # Geschwindigkeitskomponente, die senkrecht auf den Hang
    # steht, auf null gesetzt.
    grad = grad_h(r)
    if (r[1] < hangfunktion(r[0]) - toleranz_y) and (grad @ v > 0):
        v -= (grad @ v) * grad / (grad @ grad)

    return np.concatenate([v, a])


def ziel_erreicht(t, u):
    """Ereignisfunktion: Detektiere das Erreichen des Ziels."""
    r, v = np.split(u, 2)
    return r[0] - x_hang[-1]


def stehen_geblieben(t, u):
    """Ereignisfunktion: Detektiere das Anhalten des Skifahrers."""
    r, v = np.split(u, 2)
    return v[0]


# Beende die Simulation beim Erreichen des Ziels oder wenn die
# Geschwindigkeit in x-Richtung von positiv auf negativ
# wechselt. Das ist dann der Fall, wenn der Skifahrer unterwegs
# an einem ansteigenden Stück stehen bleibt und zurückrutscht.
ziel_erreicht.terminal = True
stehen_geblieben.terminal = True
stehen_geblieben.direction = -1

# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                   rtol=1e-5,
                                   events=[ziel_erreicht,
                                           stehen_geblieben],
                                   dense_output=True)

# Berechne die Interpolation auf einem feinen Raster.
t = np.arange(0, np.max(result.t), dt)
r, v = np.split(result.sol(t), 2)

# Berechne den Betrag der Zwangskraft für jeden Zeitpunkt.
F_zwang = np.zeros(t.size)
for i in range(t.size):
    F_zwang[i] = np.linalg.norm(zwangskraft(r[:, i], v[:, i]))

# Bestimme einige Kenngrößen der Simulation:
v_max = np.max(np.linalg.norm(v, axis=0))
v_end = np.linalg.norm(v[:, -1])
print(f'Anzahl Funktionsaufrufe: {result.nfev}')
print(f'Fahrtdauer:              {t[-1]:4.2f} s')
print(f'Maximalgeschwindigkeit:  {v_max*3.6:4.1f} km/h')
print(f'Endgeschwindigkeit:      {v_end*3.6:4.1f} km/h')

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(12, 3))
fig.set_tight_layout(True)
ax_ort = fig.add_subplot(1, 1, 1)
ax_ort.set_xlabel('$x$ [m]')
ax_ort.set_ylabel('$y$ [m]')
ax_ort.grid()

# Plotte den Hang als graue Linie und die Stützpunkte als
# graue Punkte.
ax_ort.plot(x_hang, y_hang, '.', color='gray')
x_hangplot = np.linspace(x_hang[0], x_hang[-1], 501)
y_hangplot = hangfunktion(x_hangplot)
ax_ort.plot(x_hangplot, y_hangplot, '--', color='gray')

# Erzeuge einen schwarzen Linienplot für die
# interpolierte Bahnkurve.
plot_bahn, = ax_ort.plot([], [], '-k')

# Plotte den Geschwindigkeitsbetrag mit einer zweiten y-Achse.
ax_geschw = ax_ort.twinx()
ax_geschw.set_ylabel('Geschwindigkeit [km/h]', color='red')
ax_geschw.tick_params(axis='y', labelcolor='red')
ax_geschw.plot(r[0], 3.6 * np.linalg.norm(v, axis=0), '-r')

# Plotte den Betrag der Normalbeschleunigung mit einer dritten
# y-Achse.
ax_normalbeschl = ax_ort.twinx()
ax_normalbeschl.set_ylabel('Beinkraft [m·g]', color='blue')
ax_normalbeschl.tick_params(axis='y', labelcolor='blue')
ax_normalbeschl.spines['right'].set_position(('outward', 60))
ax_normalbeschl.plot(r[0], F_zwang / (m * g), '-b')

# Beschränke den y-Bereich, weil sonst die Beschleunigungs-
# spitze beim Aufprall den gesamten Plot dominiert.
ax_normalbeschl.set_ylim(0, 3)

# Erzeuge eine Punktplot für die Position des Skifahrers
plot_skifahrer, = ax_ort.plot([], [], 'o', color='red', zorder=5)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position des Skifahrers.
    plot_skifahrer.set_data(r[:, n].reshape(-1, 1))

    # Stelle die Bahnkurve bis zum aktuellen Zeitpunkt dar.
    plot_bahn.set_data(r[:, :n])
    return plot_skifahrer, plot_bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
