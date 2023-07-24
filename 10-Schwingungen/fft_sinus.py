"""Fourier-Transformation einer Sinusschwingung."""

import numpy as np
import matplotlib.pyplot as plt

# Zeitdauer des Signals [s] und Abtastrate [1/s].
t_max = 0.2
abtastrate = 44100
# Signalfrequenz [Hz]
f_signal = 500.0
# Im Plot dargestellter Frequenzbereich [Hz].
frequenzbereich_plot = [-1000, 1000]

# Erzeuge ein sinusförmiges Signal.
t = np.arange(0, t_max, 1 / abtastrate)
x = np.sin(2 * np.pi * f_signal * t)

# Führe die Fourier-Transformation durch.
x_ft = np.fft.fft(x) / x.size

# Bestimme die zugehörigen Frequenzen.
frequenzen = np.fft.fftfreq(x.size, d=1/abtastrate)

# Sortiere die Frequenzen in aufsteigender Reihenfolge.
frequenzen = np.fft.fftshift(frequenzen)
x_ft = np.fft.fftshift(x_ft)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Frequenz [Hz]')
ax.set_ylabel('Amplitude')
ax.set_xlim(frequenzbereich_plot)
ax.grid()

# Plotte die Fourier-Transformierte.
ax.plot(frequenzen, np.imag(x_ft), 'ro-', label='Imaginärteil')
ax.plot(frequenzen, np.real(x_ft), 'b.-', label='Realteil')
ax.legend()

# Zeige den Plot an.
plt.show()
