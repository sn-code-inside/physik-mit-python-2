﻿"""Doppler-Effekt: Synchrone Audio- und Videoausgabe."""

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

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(12, 6))
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(plotbereich_x)
ax.set_ylim(plotbereich_y)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')
ax.grid()

# Erzeuge zwei Punktplots für Quelle und Beobachter.
plot_quelle, = ax.plot([], [], 'o', color='black')
plot_beobachter, = ax.plot([], [], 'o', color='blue')

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

    return plot_quelle, plot_beobachter


# Erzeuge einen Ausgabestrom für die Audioausgabe.
audiostream = sounddevice.OutputStream(abtastrate, channels=1,
                                       callback=audio_callback)

# Erzeuge die Animation.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)

# Starte die Audioausgabe und die Animation.
with audiostream:
    plt.show(block=True)
