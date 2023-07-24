"""Dispersion eines gaußförmigen Signals.

Vergleich der Dispersion eines gaußförmigen Wellenpaketes
    a) Betrachtung im Teilchenmodell der Masse-Feder-Kette
    b) Betrachtung im Dispersionsmodell.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationsdauer [s] und berechneter Ortsbereich [m].
t_max = 3.0
x_max = 30.0
# Zeitschrittweite [s] und Ortsauflösung [m].
dt = 0.004
dx = 0.001
# Federkonstante [N/m].
D = 100
# Masse der einzelnen Teilchen [kg].
m = 0.05
# Abstand benachbarter Massen in der Ruhelage [m].
abstand = 0.15
# Länge der ungespannten Federn [m].
federlaenge = 0.05
# Breite des gaußförmigen Wellenpakets im Zeitbereich [s].
delta_t = 0.05
# Zeitpunkt, bei dem das gaußförmige Wellenpaket bei x=0 sein
# Maximum erreicht [s].
t0 = 2.5 * delta_t
# Amplitude der longitudinalen Anregung [m].
amplitude = 0.01

# Anzahl der Teilchen ohne das anregende Teilchen ganz links.
n_teilchen = int(x_max / abstand)

# Dimension des Raumes.
n_dim = 2

# Bestimme die Ausbreitungsgeschwindigkeit der Wellen im Grenzfall
# kleiner Wellenzahlen mithilfe der angegebenen Dispersionsrelation
# und der Näherung sin(x) ≈ x.
c0 = np.sqrt(D / m) * abstand

# Bestimme die Breite des gaußförmigen Wellenpakets im Ortsbereich
# [m].
delta_x = c0 * delta_t

# Mittelpunkt des Wellenpakets zum Zeitpunkt t=0 [m].
x0 = -c0 * t0


def teilchenmodell(t):
    """Werte die Welle im Teilchenmodell aus.

    Args:
        t (np.ndarray):
            Zeitpunkte, an denen das Modell ausgewertet wird.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Ruhepositionen der Teilchen (n_teilchen).
            - Auslenkungen der Teilchen (len(t) × n_teilchen)
    """
    # Lege die Ruhelage der Teilchen auf der x-Achse fest.
    r0 = np.zeros((n_teilchen, n_dim))
    r0[:, 0] = np.linspace(abstand, n_teilchen * abstand,
                           n_teilchen)

    def anregung(t):
        """Ortsvektor der anregenden Masse zum Zeitpunkt t."""
        return np.array(
            [amplitude * np.exp(-((t - t0) / delta_t) ** 2), 0])

    def federkraft(r1, r2):
        """Kraft auf die Masse am Ort r1 durch die am Ort r2."""
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
            a[i] += federkraft(r[i], r[i - 1]) / m

        # Addiere die Beschleunigung durch die jeweils rechte Feder.
        for i in range(n_teilchen - 1):
            a[i] += federkraft(r[i], r[i + 1]) / m

        # Addiere die Beschleunigung durch die anregende Masse.
        a[0] += federkraft(r[0], anregung(t)) / m

        # Die letzte Masse soll festgehalten werden.
        a[-1] = 0

        return np.concatenate([v, a.reshape(-1)])

    # Lege den Zustandsvektor zum Zeitpunkt t=0 fest. Alle Teilchen
    # ruhen in der Ruhelage.
    v0 = np.zeros(n_teilchen * n_dim)
    u0 = np.concatenate((r0.reshape(-1), v0))

    # Löse die Bewegungsgleichung für die angegebenen Zeitpunkte.
    result = scipy.integrate.solve_ivp(dgl, [0, np.max(t)], u0,
                                       t_eval=t)
    t = result.t
    r, v = np.split(result.y, 2)

    # Wandle r in ein 3-dimensionals Array um:
    #    1. Index - Teilchennummer
    #    2. Index - Koordinatenrichtung
    #    3. Index - Zeitpunkt
    r = r.reshape(n_teilchen, n_dim, -1)

    # Wir interessieren uns nur für die x-Koodinaten und die
    # relative Auslenkung aus der Ruhelage.
    x = r0[:, 0]
    u = r[:, 0, :].T - x

    return x, u


def dispersionsmodell(t):
    """Werte die Welle im Dispersionsmodell aus.

    Args:
        t (np.ndarray):
            Zeitpunkte, an denen das Modell ausgewertet wird.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - x-Werte, an denen die Welle ausgewertet wurde.
            - Zugehörige Elongation der Welle (len(t) × len(x)).
    """
    # Erzeuge ein Array von x-Positionen.
    x = np.arange(-x_max, x_max, dx)

    # Lege die Wellenfunktion zum Zeitpunkt t=0 fest.
    u0 = amplitude * np.exp(-((x - x0) / delta_x) ** 2)

    # Führe die Fourier-Transformation durch.
    u_ft = np.fft.fft(u0)

    # Berechne die zugehörigen Wellenzahlen.
    k = 2 * np.pi * np.fft.fftfreq(x.size, d=dx)

    # Implementiere die Dispersionsrelation. Wir müssen auch hier
    # wieder dafür sorgen, dass negative Wellenzahlen eine
    # negative Kreisfrequenz bekommen und lassen daher die
    # Betragsstriche bei der Dispersionsrelation weg.
    omega = 2 * np.sqrt(D / m) * np.sin(k * abstand / 2)

    uu = u_ft * np.exp(-1j * omega * t.reshape(-1, 1))
    u = np.fft.ifft(uu, axis=1)
    return x, np.real(u)


# Werte die beiden Modelle an gegebenen Zeitpunkten aus.
t = np.arange(0, t_max, dt)
x_teilchen, u_teilchen = teilchenmodell(t)
x_dispersion, u_dispersion = dispersionsmodell(t)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('Auslenkung [m]')
ax.set_xlim(0, x_max / 2)
ax.set_ylim(-amplitude, amplitude)
ax.grid()

# Erzeuge Plots für die beiden Modellergebnisse.
plot_teilchen, = ax.plot([], [], '.-', label='Teilchenmodell')
plot_dispersion, = ax.plot([], [], '-', label='Dispersionsmodell')

# Erzeuge ein Textfeld für die Angabe des Zeitpunkts.
text_zeit = ax.text(0.96, 0.97, '',
                    transform=ax.transAxes,
                    horizontalalignment='right',
                    verticalalignment='top')

ax.legend(loc='upper left')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere die Position der simulierten Teilchen.
    plot_teilchen.set_data(x_teilchen, u_teilchen[n])
    plot_dispersion.set_data(x_dispersion, u_dispersion[n])

    # Aktualisiere die Zeitangabe.
    text_zeit.set_text(f'$t$ = {t[n]:.2f} s')

    return plot_teilchen, plot_dispersion, text_zeit


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
