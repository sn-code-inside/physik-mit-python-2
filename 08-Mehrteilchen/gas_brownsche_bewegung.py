"""Simulation der brownschen Bewegung."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.cm

# Anzahl der Teilchen.
n_teilchen = 150

# Simulationszeit und Zeitschrittweite [s].
t_max = 200
dt = 0.025

# Für jede Wand wird der Abstand vom Koordinatenursprung
# wand_d und ein nach außen zeigender Normalenvektor angegeben.
wandabstaende = np.array([2.0, 2.0, 2.0, 2.0])
wandnormalen = np.array([[0, -1.0], [0, 1.0], [-1.0, 0], [1.0, 0]])

# Anzahl der Raumdimensionen.
n_dim = wandnormalen.shape[1]

# Positioniere die Massen zufällig auf einem Kreisring mit
# Innenradius 0,5 m und Außenradius 1,9 m.
rho = 0.5 + 1.4 * np.random.rand(n_teilchen)
phi = 2 * np.pi * np.random.rand(n_teilchen)
r0 = (rho * np.array([np.cos(phi), np.sin(phi)])).T

# Wähle zufällige Geschwindigkeiten mit Komponenten zwischen -1
# und +1 [m/s].
v0 = 2 * (-0.5 + np.random.rand(n_teilchen, n_dim))

# Alle Teilchen bekommen den gleichen Radius [m].
radien = 0.02 * np.ones(n_teilchen)

# Alle Teilchen bekommen die gleiche Masse [kg].
m = 0.1 * np.ones(n_teilchen)

# Das Teilchen mit dem Index 0 stellt den Pollenkörper dar.
r0[0] = 0
v0[0] = 0
radien[0] = 0.3
m[0] = 1.0

# Kleinste Zeitdifferenz, bei der Stöße als gleichzeitig
# angenommen werden [s].
delta_t_min = 1e-9

# Lege Arrays für das Simulationsergebnis an.
t = np.arange(0, t_max, dt)
r = np.empty((t.size, n_teilchen, n_dim))
v = np.empty((t.size, n_teilchen, n_dim))
r[0] = r0
v[0] = v0


def koll_teilchen(r, v):
    """Bestimme die nächste stattfindende Teilchenkollision.

    Args:
        r (np.ndarray):
            Ortsvektoren der Teilchen (n_teilchen × n_dim).
        v (np.ndarray):
            Geschwindigkeitsvektoren (n_teilchen × n_dim).

    Returns:
        tuple[float, list[tuple[int, int]]]:
            - Die Zeitdauer bis zur nächsten Kollision oder inf,
              falls keine Teilchen mehr kollidieren.
            - Eine Liste der zugehörigen Kollisionspartner.
              Jeder Listeneintrag enthält zwei Teilchenindizes.
    """
    # Erstelle n_teilchen × n_teilchen × n_dim-Arrays, die die
    # paarweisen Orts- und Geschwindigkeitsdifferenzen enthalten:
    # dr[i, j] ist der Vektor r[i] - r[j]
    # dv[i, j] ist der Vektor v[i] - v[j]
    dr = r.reshape(n_teilchen, 1, n_dim) - r
    dv = v.reshape(n_teilchen, 1, n_dim) - v

    # Erstelle ein n_teilchen × n_teilchen-Array, das das
    # Betragsquadrat der Vektoren aus dem Array dv enthält.
    dv_quadrat = np.sum(dv * dv, axis=2)

    # Erstelle ein n_teilchen × n_teilchen-Array, das die
    # paarweisen Summen der Radien der Teilchen enthält.
    radiensummen = radien + radien.reshape(n_teilchen, 1)

    # Um den Zeitpunkt der Kollision zu bestimmen, muss eine
    # quadratische Gleichung der Form
    #          t² + 2 a t + b = 0
    # gelöst werden. Nur die kleinere Lösung ist relevant.
    a = np.sum(dv * dr, axis=2) / dv_quadrat
    b = (np.sum(dr * dr, axis=2) - radiensummen ** 2) / dv_quadrat
    D = a**2 - b
    t = -a - np.sqrt(D)

    # Suche den kleinsten positiven Zeitpunkt einer Kollision.
    t[t <= 0] = np.nan
    t_min = np.nanmin(t)

    # Suche die entsprechenden Teilchenindizes heraus.
    teilchen1, teilchen2 = np.where(np.abs(t - t_min) < delta_t_min)

    # Bilde eine Liste mit Tupeln der Kollisionspartner.
    partner = list(zip(teilchen1, teilchen2))

    # Entferne die doppelten Teilchenpaarungen.
    partner = partner[:len(partner) // 2]

    # Setze den Zeitpunkt auf inf, wenn keine Kollision stattfindet.
    if np.isnan(t_min):
        t_min = np.inf

    # Gib den Zeitpunkt und die Teilchenindizes zurück.
    return t_min, partner


def koll_wand(r, v):
    """Bestimme die nächste stattfindende Wandkollision.

    Args:
        r (np.ndarray):
            Ortsvektoren der Teilchen (n_teilchen × n_dim).
        v (np.ndarray):
            Geschwindigkeitsvektoren (n_teilchen × n_dim).

    Returns:
        tuple[float, list[tuple[int, int]]]:
            - Die Zeitdauer bis zur nächsten Kollision oder inf,
              falls keine Kollisionen mehr stattfinden.
            - Eine Liste der zugehörigen Kollisionspartner.
              Jeder Listeneintrag enthält den Teilchenindex und
              den entsprechenden Wandindex.
    """
    # Berechne die Zeitpunkte der Kollisionen der Teilchen mit
    # einer der Wände.
    # Das Ergebnis ist ein n_teilchen × Wandanzahl-Array.
    z = wandabstaende - radien.reshape(-1, 1) - r @ wandnormalen.T
    t = z / (v @ wandnormalen.T)

    # Ignoriere alle nichtpositiven Zeiten.
    t[t <= 0] = np.nan

    # Ignoriere alle Zeitpunkte, bei denen sich das Teilchen
    # entgegen den Normalenvektor bewegt. Eigentlich dürfte
    # so etwas gar nicht vorkommen, aber aufgrund von
    # Rundungsfehlern kann es passieren, dass ein Teilchen
    # sich leicht außerhalb einer Wand befindet.
    t[(v @ wandnormalen.T) < 0] = np.nan

    # Suche den kleinsten Zeitpunkt, der noch übrig bleibt.
    t_min = np.nanmin(t)

    # Setze den Zeitpunkt auf inf, wenn keine Kollision stattfindet.
    if np.isnan(t_min):
        t_min = np.inf

    # Bilde eine Liste mit Tupeln der Kollisionspartner.
    teilchen, wand = np.where(np.abs(t - t_min) < delta_t_min)
    partner = list(zip(teilchen, wand))

    # Gib den Zeitpunkt und die Indizes der Partner zurück.
    return t_min, partner


def stoss_teilchen(m1, m2, r1, r2, v1, v2):
    """Berechne die Geschwindigkeiten nach einem elastischen Stoß.

    Args:
        m1 (float):
            Masse des ersten Teilchens.
        m2 (float):
            Masse des zweiten Teilchens.
        r1 (np.ndarray):
            Ortsvektor des ersten Teilchens.
        r2 (np.ndarray):
            Ortsvektor des zweiten Teilchens.
        v1 (np.ndarray):
            Geschwindigkeitsvektor des ersten Teilchens.
        v2 (np.ndarray):
            Geschwindigkeitsvektor des zweiten Teilchens.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Die Geschwindigkeiten beider Teilchen nach dem Stoß.
    """
    # Berechne die Schwerpunktsgeschwindigkeit.
    v_schwerpunkt = (m1 * v1 + m2 * v2) / (m1 + m2)

    # Berechne die Richtung, in der der Stoß stattfindet.
    richtung = (r1 - r2) / np.linalg.norm(r1 - r2)

    # Berechne die neuen Geschwindigkeiten nach dem Stoß.
    v1_neu = v1 + 2 * (v_schwerpunkt - v1) @ richtung * richtung
    v2_neu = v2 + 2 * (v_schwerpunkt - v2) @ richtung * richtung
    return v1_neu, v2_neu


def stoss_wand(v, wandnormale):
    """Berechne die Geschwindigkeit nach einem Stoß an einer Wand.

    Args:
        v (np.ndarray):
            Geschwindigkeitsvektor des Teilchens.
        wandnormale (np.ndarray):
            Normalenvektor der Wand.

    Returns:
        np.ndarray: Geschwindigkeit nach dem Stoß.
    """
    return v - 2 * v @ wandnormale * wandnormale


# Berechne die Zeitdauer bis zur ersten Kollision und die
# beteiligten Partner.
dt_teil, stosspartner_teilchen = koll_teilchen(r[0], v[0])
dt_wand, stosspartner_wand = koll_wand(r[0], v[0])
dt_koll = min(dt_teil, dt_wand)

# Schleife über die Zeitschritte.
for i in range(1, t.size):
    # Kopiere die Werte aus dem vorherigen Zeitschritt.
    r[i] = r[i - 1]
    v[i] = v[i - 1]

    # Zeit, die in diesem Zeitschritt schon simuliert wurde.
    t1 = 0

    # Behandle nacheinander alle Kollisionen in diesem Zeitschritt.
    while t1 + dt_koll <= dt:
        # Bewege alle Teilchen bis zur nächsten Kollision vorwärts.
        r[i] += v[i] * dt_koll

        # Lass die Teilchen untereinander kollidieren.
        if dt_teil <= dt_wand:
            for teilch1, teilch2 in stosspartner_teilchen:
                v_neu = stoss_teilchen(m[teilch1], m[teilch2],
                                       r[i, teilch1], r[i, teilch2],
                                       v[i, teilch1], v[i, teilch2])
                v[i, teilch1], v[i, teilch2] = v_neu

        # Lass die Teilchen mit Wänden kollidieren.
        if dt_wand <= dt_teil:
            for teilchen, wand in stosspartner_wand:
                v[i, teilchen] = stoss_wand(v[i, teilchen],
                                            wandnormalen[wand])

        # Innerhalb dieses Zeitschritts wurde damit eine
        # Zeitdauer dt_koll bereits behandelt.
        t1 += dt_koll

        # Da Kollisionen stattgefunden haben, müssen wir diese
        # neu berechnen.
        dt_teil, stosspartner_teilchen = koll_teilchen(r[i], v[i])
        dt_wand, stosspartner_wand = koll_wand(r[i], v[i])
        dt_koll = min(dt_teil, dt_wand)

    # Bis zum Ende des aktuellen Zeitschrittes (dt) finden nun
    # keine Kollision mehr statt. Wir bewegen alle Teilchen
    # bis zum Ende des Zeitschritts vorwärts und müssen nicht
    # erneut nach Kollisionen suchen.
    r[i] += v[i] * (dt - t1)
    dt_koll -= dt - t1

    # Gib eine Information zum Fortschritt der Simulation aus.
    print(f'Zeitschritt {i + 1} von {t.size}')

# Erzeuge eine Figure und eine Axes mit entsprechenden
# Beschriftungen für die Animation der Bewegung der Teilchen.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-2.1, 2.1)
ax.grid()

# Erzeuge einen Plot für die Bahnkurve des Teilchens.
plot_bahn, = ax.plot([], [], color='red')

# Erstelle die Farbtabelle und erzeuge ein Objekt, das jeder
# Masse eine Farbe zuordnet.
mapper = mpl.cm.ScalarMappable(cmap=mpl.cm.jet)
mapper.set_array(m)
mapper.autoscale()

# Erzeuge für jedes Teilchen einen Kreis mit passendem Radius.
kreise = []
for radius, masse in zip(radien, m):
    kreis = mpl.patches.Circle([0, 0], radius, visible=False,
                               color=mapper.to_rgba(masse))
    ax.add_patch(kreis)
    kreise.append(kreis)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Positionen der Teilchen.
    for kreis, ort in zip(kreise, r[n]):
        kreis.set_center(ort)
        kreis.set_visible(True)

    # Aktualisiere die Trajektorie des Pollenkörpers (Index 0).
    plot_bahn.set_data(r[:n + 1, 0, 0], r[:n + 1, 0, 1])

    return kreise + [plot_bahn]


# Erstelle die Animation und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
