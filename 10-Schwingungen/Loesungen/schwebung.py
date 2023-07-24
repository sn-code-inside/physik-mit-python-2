"""Vergleich einer Schwebung mit einer Amplitudenmodulation."""

import numpy as np
import matplotlib.pyplot as plt

# Zeitdauer des Signals [s] und Abtastrate [1/s].
t_max = 0.2
abtastrate = 44100

# Kreisfrequenzen der beiden Schwingungen [1/s].
omega1 = 2 * np.pi * 395
omega2 = 2 * np.pi * 405

# Erzeuge eine Zeitachse.
t = np.linspace(0, t_max, 1000)

# Erzeuge die beiden Signale.
signal1 = 0.5 * np.sin(omega1 * t)
signal2 = 0.5 * np.sin(omega2 * t)

# Bilde das Summensignal.
signal_schweb = signal1 + signal2

# Berechne die passende Modulations- und Mittenkreisfrequenz
# für die Amplitudenmodulation.
omega_mod = omega2 - omega1
omega_0 = (omega1 + omega2) / 2

# Berechne ein amplitudenmoduliertes Signal.
signal_am = 0.5 * (1 + np.cos(omega_mod * t)) * np.sin(omega_0 * t)

# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Stelle die Schwebung dar.
ax_schwebung = fig.add_subplot(2, 1, 1)
ax_schwebung.set_title('Schwebung')
ax_schwebung.set_xlabel('$t$ [s]')
ax_schwebung.set_ylabel('$y$ [a.u.]')
ax_schwebung.set_xlim(0, t_max)
ax_schwebung.grid()
ax_schwebung.plot(t, signal_schweb)

# Stelle die Amplitudenmodulation dar.
ax_am = fig.add_subplot(2, 1, 2)
ax_am.set_title('Amplitudenmodulation')
ax_am.set_xlabel('$t$ [s]')
ax_am.set_ylabel('$y$ [a.u.]')
ax_am.set_xlim(0, t_max)
ax_am.grid()
ax_am.plot(t, signal_am)

# Zeige den Plot an.
plt.show()
