"""Fourier-Transformation einer gedämpften Sinusschwingung."""

import numpy as np
import matplotlib.pyplot as plt

# Zeitdauer des Signals [s] und Abtastrate [1/s].
t_max = 0.2
abtastrate = 44100
# Signalfrequenz [Hz] und Abklingkoeffizient [1/s]
f_signal = 500.0
delta = 30.0
# Im Plot dargestellter Frequenzbereich [Hz].
frequenzbereich_plot = [-1000, 1000]

# Erzeuge ein gedämpftes, sinusförmiges Signal.
t = np.arange(0, t_max, 1 / abtastrate)
x = np.sin(2 * np.pi * f_signal * t) * np.exp(-delta * t)

# Führe die Fourier-Transformation durch.
x_ft = np.fft.fft(x) / x.size

# Bestimme die zugehörigen Frequenzen.
frequenzen = np.fft.fftfreq(x.size, d=1 / abtastrate)

# Sortiere die Frequenzen in aufsteigender Reihenfolge.
frequenzen = np.fft.fftshift(frequenzen)
x_ft = np.fft.fftshift(x_ft)

# Erzeuge eine Figure und ein GridSpec-Objekt.
fig = plt.figure(figsize=(10, 5))
fig.set_tight_layout(True)
gridspec = fig.add_gridspec(2, 2)

# Erzeuge eine Axes und plotte den Zeitverlauf.
ax_zeitverlauf = fig.add_subplot(gridspec[:, 0])
ax_zeitverlauf.set_xlabel('$t$ [s]')
ax_zeitverlauf.set_ylabel('Auslenkung')
ax_zeitverlauf.grid()
ax_zeitverlauf.plot(t, x)

# Erzeuge eine Axes und plotte Real- und Imaginärteil der
# Fourier-Transformierten.
ax_freq = fig.add_subplot(gridspec[0, 1])
ax_freq.set_xlabel('Frequenz [Hz]')
ax_freq.set_ylabel('Amplitude')
ax_freq.set_xlim(frequenzbereich_plot)
ax_freq.grid()
ax_freq.plot(frequenzen, np.real(x_ft), 'r', label='Realteil')
ax_freq.plot(frequenzen, np.imag(x_ft), 'b', label='Imaginärteil')
ax_freq.legend(loc='upper right')

# Erzeuge eine Axes und plotte den Betrag der
# Fourier-Transformierten.
ax_freq_abs = fig.add_subplot(gridspec[1, 1])
ax_freq_abs.set_xlabel('Frequenz [Hz]')
ax_freq_abs.set_ylabel('Amplitude')
ax_freq_abs.set_xlim(frequenzbereich_plot)
ax_freq_abs.grid()
ax_freq_abs.plot(frequenzen, np.abs(x_ft), label='Betrag')
ax_freq_abs.legend()

# Zeige den Plot an.
plt.show()
