"""Veranschaulichung des Aliasing-Effekts."""

import numpy as np
import matplotlib.pyplot as plt

# Zeitdauer des Signals und Zeitschrittweite [s]
t_max = 20
dt = 1.0

# Frequenzen der beiden zu betrachtenden Signale [Hz].
frequenzen = [1.00, 1.05]

# Erzeuge ein sehr fein aufgelöstes Zeitraster.
t_fein = np.linspace(0, t_max, 1000)

# Erzeuge ein Zeitraster mit dem Abtastintervall dt.
t_abtast = np.arange(0, t_max + dt, dt)


def signal(freq, t):
    """Berechne eine harmonische Schwingung."""
    return np.sin(2 * np.pi * freq * t - np.pi / 4)


# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Iteriere über die zu betrachtenden Frequenzen.
for i, freq in enumerate(frequenzen):
    # Erzeuge eine neue Axes.
    ax = fig.add_subplot(len(frequenzen), 1, i+1)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('Auslenkung')
    ax.set_xlim(0, t_max)
    ax.grid()

    # Berechne die Aliasing-Frequenz.
    freq_alias = freq - 1/dt

    # Plotte das tatsächliche Signal sehr fein aufgelöst.
    ax.plot(t_fein, signal(freq, t_fein),
            'b', label=f'f={freq:.2f} Hz')

    # Plotte das Signal mit der Aliasing-Frequenz.
    ax.plot(t_fein, signal(freq_alias, t_fein),
            'k', label=f'f={freq_alias:.2f} Hz')

    # Plotte das abgetastete Signal.
    ax.plot(t_abtast, signal(freq, t_abtast),
            'or', label='Abtastung')

    # Stelle die Legende dar.
    ax.legend(loc='upper right')

# Zeige die Grafik an.
plt.show()
