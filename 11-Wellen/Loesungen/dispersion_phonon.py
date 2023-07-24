"""Dispersion einer gaußförmigen Anfangsauslenkung.

Hier wurde die gegebene Dispersionsrelation für Phononen benutzt.
Die anderen Parameter wurden so gewählt, dass sie vergleichbar
zur Situation im Programm `kette_longitudinal1.py` sind.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Simulationsdauer [s] und berechneter Ortsbereich [m].
t_max = 3.0
x_max = 30.0
# Zeitschrittweite [s] und Ortsauflösung [m].
dt = 0.01
dx = 0.001
# Federkonstante [N/m].
D = 100
# Masse der einzelnen Teilchen [kg].
m = 0.05
# Abstand benachbarter Massen in der Ruhelage [m].
abstand = 0.15
# Breite des gaußförmigen Wellenpakets im Zeitbereich [s].
delta_t = 0.05

# Bestimme die Ausbreitungsgeschwindigkeit der Wellen im Grenzfall
# kleiner Wellenzahlen mithilfe der angegebenen Dispersionsrelation
# und der Näherung sin(x) ≈ x.
c0 = np.sqrt(D / m) * abstand

# Bestimme die Breite des gaußförmigen Wellenpakets im Ortsbereich
# [m].
delta_x = c0 * delta_t

# Mittelpunkt des Wellenpakets zum Zeitpunkt t=0 [m].
x0 = 3 * delta_x

# Erzeuge je ein Array von x-Positionen und Zeitpunkten.
x = np.arange(0, x_max, dx)
t = np.arange(0, t_max, dt)

# Lege die Wellenfunktion zum Zeitpunkt t=0 fest.
u0 = np.exp(-((x - x0) / delta_x) ** 2)

# Führe die Fourier-Transformation durch.
u_ft = np.fft.fft(u0)

# Berechne die zugehörigen Wellenzahlen.
k = 2 * np.pi * np.fft.fftfreq(x.size, d=dx)

# Implementiere die Dispersionsrelation. Wir müssen auch hier
# wieder dafür sorgen, dass negative Wellenzahlen eine negative
# Kreisfrequenz bekommen und lassen daher die Betragsstriche bei
# der Dispersionsrelation weg.
omega = 2 * np.sqrt(D / m) * np.sin(k * abstand / 2)

# Erzeuge eine Figure und eine Axes. Wir stellen dabei nur
# die erste Hälte der berechneten x-Werte dar.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('Auslenkung [a.u.]')
ax.set_ylim(-0.5, 1.2)
ax.set_xlim(0, x_max / 2)
ax.grid()

# Erzeuge einen Linienplot für die Welle.
ax.plot(x, np.real(u0), color='lightblue')
plot_welle, = ax.plot([], [], color='blue')

# Erzeuge ein Textfeld für die Ausgabe des Zeitpunktes.
text_zeit = ax.text(0.1, 0.9, '', transform=ax.transAxes)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    u = np.fft.ifft(u_ft * np.exp(-1j * omega * t[n]))
    plot_welle.set_data(x, np.real(u))

    # Aktualisiere das Textfeld.
    text_zeit.set_text(f'$t$ = {t[n]:3.1f} s')

    return plot_welle, text_zeit


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
