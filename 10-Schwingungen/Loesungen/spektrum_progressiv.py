"""Frequenzspektrum der Schwingung mit progressiver Feder."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Zeitdauer des Signals [s] und Abtastrate [1/s].
t_max = 1.0
abtastrate = 44100
# Anfängliche Federhärte [N/m].
D0 = 400.0
# Konstante für den nichtlinearen Term [m].
alpha = 5e-3
# Masse [kg].
m = 1e-3
# Reibungskoeffizient [kg/s].
b = 0.01
# Anfangsauslenkung [m].
x0 = 20e-3
# Anfangsgeschwindigkeit [m/s].
v0 = 0


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    x, v = np.split(u, 2)
    F_feder = -D0 * alpha * np.expm1(np.abs(x)/alpha) * np.sign(x)
    F = F_feder - b * v
    return np.concatenate([v, F / m])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.array([x0, v0])

# Löse die Differentialgleichung an den durch die Abtastrate
# vorgegebenen Zeitpunkten.
t = np.arange(0, t_max, 1 / abtastrate)
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-6,
                                   t_eval=t)
x, v = result.y

# Führe die Fourier-Transformation durch.
x_ft = np.fft.fft(x) / x.size

# Bestimme die zugehörigen Frequenzen.
freq = np.fft.fftfreq(x.size, d=1 / abtastrate)

# Sortiere die Frequenzen in aufsteigender Reihenfolge.
freq = np.fft.fftshift(freq)
x_ft = np.fft.fftshift(x_ft)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(12, 4))
fig.set_tight_layout(True)

# Erzeuge eine Axes und plotte den Zeitverlauf.
ax_freq = fig.add_subplot(1, 2, 1)
ax_freq.set_xlabel('$t$ [s]')
ax_freq.set_ylabel('Auslenkung [m]')
ax_freq.grid()
ax_freq.plot(t, x)

# Erzeuge eine Axes und plotte den Betrag der
# Fourier-Transformierten.
ax_freq_abs = fig.add_subplot(1, 2, 2)
ax_freq_abs.set_xlabel('Frequenz [Hz]')
ax_freq_abs.set_ylabel('Amplitude')
ax_freq_abs.set_xlim(-1000, 1000)
ax_freq_abs.grid()
ax_freq_abs.plot(freq, np.abs(x_ft))

# Zeige den Plot an.
plt.show()
