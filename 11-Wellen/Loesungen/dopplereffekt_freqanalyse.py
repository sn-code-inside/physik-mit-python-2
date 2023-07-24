"""Doppler-Effekt: Darstellung des empfangenen Frequenzspektrums."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import sounddevice

# Simulationsdauer [s] und Abtastrate [1/s].
t_max = 10.0
abtastrate = 44100
# Frequenz, mit der die Quelle Schallwellen aussendet [Hz].
f_Q = 300.0
# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 340.0
# Lege die Startposition der Quelle und des Beobachters fest [m].
startort_Q = np.array([-150.0, 5.0])
startort_B = np.array([0.0, -5.0])
# Lege die Geschwindigkeit von Quelle und Beobachter fest [m/s].
v_Q = np.array([30.0, 0.0])
v_B = np.array([0.0, 0.0])
# Dargestellter Koordinatenbereich [m].
plotbereich_x = (-160, 160)
plotbereich_y = (-30, 30)
# Zeitdauer für das Fenster der Fourier-Transformation [s].
zeitfenster = 1.0

# Anzahl der Datenpunkte in der Fourier-Transformierten.
n_punkte_fft = int(zeitfenster * abtastrate)


def signal_quelle(t):
    """Ausgesendetes Signal als Funktion der Zeit."""
    signal = np.sin(2 * np.pi * f_Q * t)
    signal[t < 0] = 0.0
    return signal


# Erzeuge ein Array von Zeitpunkten und lege ein leeres Array
# für das vom Beobachter empfangene Signal an.
t = np.arange(0, t_max, 1 / abtastrate)
signal_beob = np.zeros(t.size)

# Berechne für jeden Zeitpunkt die Positionen von Quelle und
# Beobachter.
r_Q = startort_Q + v_Q * t.reshape(-1, 1)
r_B = startort_B + v_B * t.reshape(-1, 1)

# Berechne für jeden Zeitpunkt t, zu dem der Beobachter ein
# Signal auffängt, die Zeitdauer dt, die das Signal
# von der Quelle benötigt hat. Dazu ist eine quadratische
# Gleichung der Form
#             dt² - 2 a dt - b = 0
# mit den unten definierten Größen a und b zu lösen.
r = r_B - r_Q
a = np.sum(v_Q * r, axis=1) / (c ** 2 - v_Q @ v_Q)
b = np.sum(r ** 2, axis=1) / (c ** 2 - v_Q @ v_Q)

# Berechne für jeden Zeitpunkt die beiden Lösungen der
# quadratischen Gleichung.
dt1 = a + np.sqrt(a ** 2 + b)
dt2 = a - np.sqrt(a ** 2 + b)

# Berücksichtige das Signal der positiven Lösungen.
# Beachte, dass die Amplitude mit 1/r abfällt.
idx1 = dt1 > 0
abstand = c * dt1[idx1]
signal_beob[idx1] = signal_quelle(t[idx1] - dt1[idx1]) / abstand

idx2 = dt2 > 0
abstand = c * dt2[idx2]
signal_beob[idx2] += signal_quelle(t[idx2] - dt2[idx2]) / abstand

# Normiere das Signal auf den Wertebereich -1 ... +1.
if np.max(np.abs(signal_beob)) > 0:
    signal_beob = signal_beob / np.max(np.abs(signal_beob))

# Erzeuge eine Figure.
fig = plt.figure(figsize=(12, 6))
fig.set_tight_layout(True)

# Erzeuge eine Axes für die Animation von Quelle und Beobachter.
ax_ani = fig.add_subplot(2, 1, 1)
ax_ani.grid()
ax_ani.set_xlim(plotbereich_x)
ax_ani.set_ylim(plotbereich_y)
ax_ani.set_aspect('equal')
ax_ani.set_xlabel('$x$ [m]')
ax_ani.set_ylabel('$y$ [m]')

# Erzeuge eine Axes für das Frequenzspektrum.
ax_freq = fig.add_subplot(2, 1, 2)
ax_freq.grid()
ax_freq.set_xlim(260, 340)
ax_freq.set_ylim(0, 1)
ax_freq.set_xlabel('$f$ [Hz]')
ax_freq.set_ylabel('Amplitude [a.u.]')

# Erzeuge zwei Punktplots für Quelle und Beobachter.
plot_quelle, = ax_ani.plot([], [], 'o', color='black')
plot_beobachter, = ax_ani.plot([], [], 'o', color='blue')

# Erzeuge die Frequenzachse für die Fourier-Transformation.
frequenzen = np.fft.fftfreq(n_punkte_fft, d=1 / abtastrate)
frequenzen = np.fft.fftshift(frequenzen)

# Erzeuge eine Linienplot für die Fourier-Transformation.
plot_fft, = ax_freq.plot([], [])

# Startindex für die nächste Audioausgabe.
audio_index = 0


def audio_callback(outdata, frames, time, status):
    """Stelle neue Audiodaten zur Ausgabe bereit."""
    global audio_index

    # Gib im Fehlerfall eine Fehlermeldung aus.
    if status:
        print(status)

    # Extrahiere den benötigten Ausschnitt aus den Daten. Durch
    # das Slicing kann es passieren, dass 'audiodaten' weniger
    # Datenpunkte als die Anzahl der 'frames' enthält.
    audiodaten = signal_beob[audio_index: audio_index + frames]

    # Schreibe die Daten in das Ausgabe-Array.
    outdata[:audiodaten.size, 0] = audiodaten

    # Fülle das Ausgabe-Array ggf. mit Nullen auf.
    outdata[audiodaten.size:, 0] = 0.0

    # Erhöhe den Index um die Anzahl der verwendeten Datenpunkte.
    audio_index += audiodaten.size


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Ignoriere den von der Animation übergebenen Bildindex n und
    # verwende stattdessen den Index, der sich aus der
    # Audioausgabe ergibt, sodass Audio- und Videoausgabe synchron
    # erfolgen.
    n = min(audio_index, t.size - 1)

    # Aktualisiere die Positionen von Quelle und Beobachter.
    plot_quelle.set_data(r_Q[n])
    plot_beobachter.set_data(r_B[n])

    # Aktualisiere die Fourier-Transformation.
    if n > n_punkte_fft:
        signal_fenster = signal_beob[n - n_punkte_fft:n]
        ft = np.fft.fft(signal_fenster) / n_punkte_fft
        ft = np.fft.fftshift(ft)
        ft = np.abs(ft)
        ft /= np.max(ft)
        plot_fft.set_data(frequenzen, ft)

    return plot_quelle, plot_beobachter, plot_fft


# Erzeuge einen Ausgabestrom für die Audioausgabe.
stream = sounddevice.OutputStream(abtastrate, channels=1,
                                  callback=audio_callback)

# Erzeuge die Animation.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)

# Starte die Audioausgabe und die Animation.
with stream:
    plt.show(block=True)
