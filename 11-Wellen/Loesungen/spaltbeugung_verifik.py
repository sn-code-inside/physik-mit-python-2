"""Beugungsfeld eines Spaltes nach dem huygensschen Prinzip.

In diesem Programm wird das Simulationsergebnis mit der
bekannten Spaltbeugungsformel für das Fernfeld verglichen.
"""

import numpy as np
import matplotlib.pyplot as plt

# Wellenlänge [m].
wellenlaenge = 1.0
# Anzahl der Elementarwellen.
n_wellen = 100
# Spaltbreite [m].
spaltbreite = 10
# Betrachteter Abstand [m].
entfernung = 500

# Lege die Startpositionen jeder Elementarwelle fest.
positionen = np.zeros((n_wellen, 2))
positionen[:, 0] = np.linspace(-spaltbreite / 2, spaltbreite / 2,
                               n_wellen)

# Lege die Phase jeder Elementarwelle fest [rad].
phasen = np.zeros(n_wellen)

# Lege die Amplitude jeder Elementarwelle fest.
amplituden = spaltbreite / n_wellen * np.ones(n_wellen)

# Berechne die Wellenzahl.
k = 2 * np.pi / wellenlaenge

# Werte die Wellenfunktionen einem Halbkreis mit
# Radius R auswerten.
alpha = np.radians(np.linspace(-90, 90, 1000))
x = entfernung * np.sin(alpha)
y = entfernung * np.cos(alpha)

# Lege ein Array komplexer Zahlen der passenden Größe an, das mit
# Nullen gefüllt ist.
u = np.zeros(x.shape, dtype=complex)

# Addiere die komplexe Amplitude jeder Elementarwelle.
for A, (x0, y0), phi0 in zip(amplituden, positionen, phasen):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    cos_theta = (y - y0) / r
    u += A * np.exp(1j * (k * r + phi0)) / np.sqrt(r) * cos_theta

# Normiere die Intensität so, dass Sie beim Winkel alpha=0
# gerade den Wert 1 hat.
I_sim = np.abs(u)**2 / np.max(np.abs(u)**2)

# Berechne die Fernfeldnäherung gemäß der angegebenen Gleichung
# mit I0 = 1.
I_theo = np.sinc(spaltbreite * np.sin(alpha) / wellenlaenge) ** 2

# Erzeuge eine Figure und eine Axes und plotte beide Kurven.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-20, 20)
ax.set_xlabel('$\\alpha$ [°]')
ax.set_ylabel('$I / I_0$')
ax.grid()

ax.plot(np.degrees(alpha), I_sim, label='Simulation')
ax.plot(np.degrees(alpha), I_theo, label='Fernfeldnäherung')
ax.legend()

plt.show()
