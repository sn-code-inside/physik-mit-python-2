"""Frequenzmodulierte Sinusschwingung."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice

# Zeitdauer des Signals [s] und Abtastrate [1/s].
t_max = 3.0
abtastrate = 44100
# Im Plot dargestellter Zeitbereich [s] und Frequenzbereich [Hz].
zeitbereich_plot = [0, 0.1]
frequenzbereich_plot = [0, 800]
# Parameter für die Frequenzmodulation:
# Mittenkreisfrequenz [1/s] und Frequenzhub [1/s].
omega_0 = 2 * np.pi * 400
omega_hub = 2 * np.pi * 200

# Kreisfrequenz der Modulation [1/s].
omega_mod = 2 * np.pi * 20


def s(t):
    """Signal als Funktion der Zeit."""
    return np.sin(omega_mod * t)


def y(t):
    """Frequenzmodulierter Träger."""
    phi = omega_0 * t + omega_hub / omega_mod * (
            1 - np.cos(omega_mod * t))
    return np.sin(phi)


# Erzeuge eine Zeitachse.
t = np.arange(0, t_max, 1 / abtastrate)

# Erzeuge eine Figure ein GridSpec-Objekt.
fig = plt.figure(figsize=(10, 5))
fig.set_tight_layout(True)
gridspec = fig.add_gridspec(2, 2)

# Plotte das zu modulierende Signal.
ax_signal = fig.add_subplot(gridspec[0, 0])
ax_signal.set_ylabel('$s(t)$')
ax_signal.tick_params(labelbottom=False)
ax_signal.set_xlim(zeitbereich_plot)
ax_signal.grid()
ax_signal.plot(t, s(t))

# Plotten den frequenzmodulierten Träger.
ax_modulation = fig.add_subplot(gridspec[1, 0], sharex=ax_signal)
ax_modulation.set_ylabel('$y(t)$')
ax_modulation.set_xlabel('$t$ [s]')
ax_modulation.grid()
ax_modulation.plot(t, y(t))

# Führe die Fourier-Transformation durch.
y_ft = np.fft.fft(y(t)) / t.size
y_ft = np.fft.fftshift(y_ft)
freq = np.fft.fftfreq(t.size, d=1 / abtastrate)
freq = np.fft.fftshift(freq)

# Plotte das Spektrum des amplitudenmodulierten Trägers.
ax_freq_abs = fig.add_subplot(gridspec[:, 1])
ax_freq_abs.set_xlabel('$f$ [Hz]')
ax_freq_abs.set_ylabel('Amplitude')
ax_freq_abs.set_xlim(frequenzbereich_plot)
ax_freq_abs.grid()
ax_freq_abs.plot(freq, np.abs(y_ft), '.-')

# Gib den frequenzmodulierten Träger als Audiodatei aus.
audiodaten = np.int16(y(t) / np.max(np.abs(y(t))) * 32767)
scipy.io.wavfile.write('output.wav', abtastrate, audiodaten)

# Gib das Signal als Sound aus.
sounddevice.play(audiodaten, abtastrate, blocking=True)

# Zeige den Plot an.
plt.show()
