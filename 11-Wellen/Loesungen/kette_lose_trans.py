"""Wellenausbreitung auf einer gespannten Masse-Feder-Kette.

Das Ende der Kette kann sich nun in y-Richtung frei bewegen.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeit und Zeitschrittweite [s].
t_max = 10.0
dt = 0.01
# Dimension des Raumes.
n_dim = 2
# Anzahl der Teilchen ohne das anregende Teilchen ganz links.
n_teilchen = 70
# Federkonstante [N/m].
D = 100
# Masse [kg].
m = 0.05
# Länge der ungespannten Federn [m].
federlaenge = 0.05
# Abstand benachbarter Massen in der Ruhelage [m].
abstand = 0.15
# Amplitude der longitudinalen und transversalen Anregung [m].
A_long = 0.05
A_tran = 0.05

# Lege die Ruhelage der Teilchen auf der x-Achse fest.
r0 = np.zeros((n_teilchen, n_dim))
r0[:, 0] = np.linspace(abstand, n_teilchen * abstand, n_teilchen)


def anregung(t, t0=0.5, delta_t=0.2):
    """Ortsvektor der anregenden Masse zum Zeitpunkt t."""
    pos = np.array([A_long * np.exp(-((t - t0) / delta_t) ** 2),
                    A_tran * np.exp(-((t - t0) / delta_t) ** 2)])
    return pos


def federkraft(r1, r2):
    """Kraft auf die Masse am Ort r1 durch die Masse am Ort r2."""
    abstand = np.linalg.norm(r2 - r1)
    einheitsvektor = (r2 - r1) / abstand
    return D * (abstand - federlaenge) * einheitsvektor


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    r, v = np.split(u, 2)
    r = r.reshape(n_teilchen, n_dim)
    a = np.zeros((n_teilchen, n_dim))

    # Addiere die Beschleunigung durch die jeweils linke Feder.
    for i in range(1, n_teilchen):
        a[i] += federkraft(r[i], r[i-1]) / m

    # Addiere die Beschleunigung durch die jeweils rechte Feder.
    for i in range(n_teilchen - 1):
        a[i] += federkraft(r[i], r[i+1]) / m

    # Addiere die Beschleunigung durch die anregende Masse.
    a[0] += federkraft(r[0], anregung(t)) / m

    # Die lezte Masse soll in y-Richtung frei bewegbar sein.
    # In x-Richtung wollen wir die Masse aber weiterhin
    # festhalten.
    a[-1, 0] = 0

    return np.concatenate([v, a.reshape(-1)])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest. Alle Teilchen
# ruhen in der Ruhelage.
v0 = np.zeros(n_teilchen * n_dim)
u0 = np.concatenate((r0.reshape(-1), v0))

# Löse die Bewegungsgleichung bis zum Zeitpunkt t_max.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r, v = np.split(result.y, 2)

# Wandle r und v in ein 3-dimensionals Array um:
#    1. Index - Teilchennummer
#    2. Index - Koordinatenrichtung
#    3. Index - Zeitpunkt
r = r.reshape(n_teilchen, n_dim, -1)
v = v.reshape(n_teilchen, n_dim, -1)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(12, 4))

# Erzeuge eine Axes für die animierte Darstellung der
# Masse-Feder-Kette.
ax_teilchen = fig.add_subplot(2, 1, 1)
ax_teilchen.set_xlim(-abstand, (n_teilchen + 1) * abstand)
ax_teilchen.set_ylim(-2.2 * A_tran, 2.2 * A_tran)
ax_teilchen.set_ylabel('$y$ [m]')
ax_teilchen.tick_params(labelbottom=False)
ax_teilchen.grid()

# Erzeuge Punktplots für die Teilchenpositionen.
plot_teilchen, = ax_teilchen.plot([], [], 'ob')
plot_teilchen_anregung, = ax_teilchen.plot([], [], 'or')

# Erzeuge ein Textfeld für die Angabe des Zeitpunkts.
text_zeit = ax_teilchen.text(0.97, 0.97, '',
                             transform=ax_teilchen.transAxes,
                             horizontalalignment='right',
                             verticalalignment='top')

# Erzeuge eine zweite Axes für die animierte Darstellung der
# transversalen und longitudinalen Auslenkung.
ax_auslenkung = fig.add_subplot(2, 1, 2)
ax_auslenkung.set_xlim(-abstand, (n_teilchen + 1) * abstand)
ax_auslenkung.set_ylim(-2.2 * max(A_tran, A_long),
                       +2.2 * max(A_tran, A_long))
ax_auslenkung.set_ylabel('$u$ [m]')
ax_auslenkung.set_xlabel('$x$ [m]')
ax_auslenkung.grid()

# Erzeuge je einen Linienplot für die Momentanauslenkung.
plot_welle_trans, = ax_auslenkung.plot([], [], label='trans')
plot_welle_long, = ax_auslenkung.plot([], [], label='long')

# Füge eine Legende hinzu.
ax_auslenkung.legend(loc='upper left')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position der simulierten Teilchen.
    plot_teilchen.set_data(r[:, :, n].T)

    # Aktualisiere die Position des anregenden Teilchens.
    plot_teilchen_anregung.set_data(anregung(t[n]))

    # Erzeuge ein Array der Auslenkungen aller Teilchen und ein
    # Array der x-Positionen der Ruhelagen.
    auslenkungen = np.concatenate([anregung(t[n]).reshape(1, n_dim),
                                   r[:, :, n] - r0])
    ruhelage_x = np.concatenate(([0], r0[:, 0]))

    # Aktualisiere den Plot für die Transversal- und
    # Longitudinalwelle.
    plot_welle_long.set_data(ruhelage_x, auslenkungen[:, 0])
    plot_welle_trans.set_data(ruhelage_x, auslenkungen[:, 1])

    # Aktualisiere die Zeitangabe.
    text_zeit.set_text(f'$t$ = {t[n]:.2f} s')

    return (plot_teilchen, plot_teilchen_anregung,
            plot_welle_trans, plot_welle_long, text_zeit)


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
