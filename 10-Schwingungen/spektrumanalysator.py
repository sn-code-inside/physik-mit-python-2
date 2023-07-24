"""Ein einfacher Spektrum-Analysator."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import sounddevice

# Länge des Zeitfensters [s] und Abtastrate [1/s].
t_max = 0.5
abtastrate = 44100

# Im Plot dargestellter Frequenzbereich [Hz] und der
# Bereich der Amplituden [a.u.].
frequenzbereich_plot_ft = [100, 10000]
amplitudenbereich_plot_ft = [0, 0.03]

# Erzeuge eine passende Zeitachse.
t = np.arange(0, t_max, 1 / abtastrate)

# Bestimme die Frequenzen der Fourier-Transformation.
frequenzen = np.fft.fftfreq(t.size, d=1 / abtastrate)
frequenzen = np.fft.fftshift(frequenzen)

# Plot für das Zeitsignal.
fig = plt.figure(figsize=(10, 5))
fig.set_tight_layout(True)

# Plotte das Zeitsignal.
ax_zeitverlauf = fig.add_subplot(2, 1, 1)
ax_zeitverlauf.set_ylim(-1.0, 1.0)
ax_zeitverlauf.set_xlim(0, t_max)
ax_zeitverlauf.set_xlabel('$t$ [s]')
ax_zeitverlauf.set_ylabel('Zeitsignal [a.u.]')
ax_zeitverlauf.grid()
plot_zeit, = ax_zeitverlauf.plot([], [])

# Plotte das Frequenzspektrum.
ax_freq_abs = fig.add_subplot(2, 1, 2)
ax_freq_abs.set_xlim(frequenzbereich_plot_ft)
ax_freq_abs.set_ylim(amplitudenbereich_plot_ft)
ax_freq_abs.set_xlabel('$f$ [Hz]')
ax_freq_abs.set_ylabel('Amplitude [a.u.]')
ax_freq_abs.set_xscale('log')
ax_freq_abs.grid(True, 'both')
plot_freq, = ax_freq_abs.plot([], [])

# Erzeuge ein Array, das das aufgenommene Zeitsignal speichert.
audiodaten = np.zeros(t.size)


def audio_callback(indata, frames, time, status):
    """Verarbeite die neu verfügbaren Audiodaten."""
    global audiodaten

    # Gib im Fehlerfall eine Fehlermeldung aus.
    if status:
        print(status)

    # Kopiere die Audiodaten in das Array data.
    if frames < audiodaten.size:
        audiodaten[:] = np.roll(audiodaten, -frames)
        audiodaten[-frames:] = indata[:, 0]
    else:
        audiodaten[:] = indata[-audiodaten.size:, 0]


def update(frame):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktualisiere den Plot des Zeitsignals.
    plot_zeit.set_data(t, audiodaten)

    # Aktualisiere die Fourier-Transformierte.
    ft = np.fft.fft(audiodaten) / audiodaten.size
    ft = np.fft.fftshift(ft)
    plot_freq.set_data(frequenzen, np.abs(ft))

    return plot_zeit, plot_freq


# Erzeuge das Animationsobjekt.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)

# Starte die Audioaufnahme und zeige die Animation an.
with sounddevice.InputStream(abtastrate, channels=1,
                             callback=audio_callback):
    plt.show(block=True)
