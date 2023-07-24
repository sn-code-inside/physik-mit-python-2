"""Doppler-Effekt. Erzeugen einer Videodatei mit Audiospur."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.io.wavfile
import os

# Simulationsdauer [s] und Abtastrate [1/s].
t_max = 10.0
abtastrate = 44100
# Bildrate (frames per second) für das Video [1/s].
fps = 30
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


def signal_quelle(t):
    """Ausgesendetes Signal als Funktion der Zeit."""
    signal = np.sin(2 * np.pi * f_Q * t)
    signal[t < 0] = 0.0
    return signal


# Erzeuge ein Array von Zeitpunkten und lege ein leeres Array
# für das empfangene Signal an.
t = np.arange(0, t_max, 1 / abtastrate)
signal_beob = np.zeros(t.size)

# Berechne die Anzahl der Bilder im Video
n_frames = int(t_max * fps)

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

# Skaliere das Signal so, dass es in den Wertebereich von
# ganzen 16-bit Zahlen passt (-32768 ... +32767) und wandle
# es anschließend in 16-bit-Integers um.
if np.max(np.abs(signal_beob)) > 0:
    signal_beob *= 32767 / np.max(np.abs(signal_beob))
signal_beob = np.int16(signal_beob)

# Gib das Signal als Audiodatei im wav-Format aus.
scipy.io.wavfile.write('audio.wav', abtastrate, signal_beob)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(plotbereich_x)
ax.set_ylim(plotbereich_y)
ax.set_aspect('equal')
ax.grid()

# Erzeuge zwei Punktplots für Quelle und Empfänger.
plot_quelle, = ax.plot([], [], 'o', color='black')
plot_beobachter, = ax.plot([], [], 'o', color='blue')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Berechne, welcher Audio-Index zu diesem Bild gehört.
    audio_index = int(n / fps * abtastrate)

    # Aktualisiere die Positionen von Quelle und Beobachter.
    plot_quelle.set_data(r_B[audio_index].reshape(-1, 1))
    plot_beobachter.set_data(r_Q[audio_index].reshape(-1, 1))

    return plot_quelle, plot_beobachter


# Erzeuge die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=30, blit=True)

# Wir speichern jetzt die Animation ab. Wir haben bereits
# dafür gesorgt, dass die Dauer der Animation und die Dauer
# der Audiodatei gleich groß sind. Das eigentliche Erzeugen
# der Videodatei übernimmt das Programm ffmpeg. Diesem
# Programm müssen wir mit der Option "-i audio.wav" sagen,
# dass es einen zusätzlichen Eingabekanal gibt. Leider werden
# durch die zusätzliche Eingabeoption die Ausgabeoptionen für
# das Videoformat überschrieben. Wir müssen daher mit
# "-codec:v h264 -pix_fmt yuv420p" den Video-Codec und das
# Farbformat festlegen. Als Nächstes geben wir mit "-codec:a
# aac" an, dass die Audiodaten im AAC-Codec gespeichert
# werden sollen. Die letzte Option "-crf 20" gibt die Qualität
# der Videodatei an. Je kleiner diese Zahl ist, desto besser
# ist die Qualität.
ani.save('output.mp4', fps=fps,
         extra_args=['-i', 'audio.wav',
                     '-codec:v', 'h264', '-pix_fmt', 'yuv420p',
                     '-codec:a', 'aac',
                     '-crf', '20'])

# Als Letztes löschen wir die Wave-Datei.
os.unlink('audio.wav')

# Alternativ kann man auch mit
#    ani.save('video.mp4', fps=fps)
# nur eine Videodatei ohne Ton erzeugen. Anschließend kann man
# dann auf einer Kommandozeile die Audiodatei und die
# Videodatei zusammenführen. Dazu kann man auch wieder
# ffmpeg benutzen. Ein möglicher Kommandozeilenbefehl, der
# die Datei audio.wav und die Datei video.mp4 in eine
# Videodatei mit Ton überführt lautet:
#
#    ffmpeg -i audio.wav -i video.mp4 -codec:v copy -codec:a aac output.mp4
