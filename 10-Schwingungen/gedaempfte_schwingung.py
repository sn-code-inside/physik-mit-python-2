"""Simulation und Audioausgabe einer gedämpften Schwingung."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.io.wavfile
import sounddevice

# Zeitdauer der Simulation [s].
t_max = 3.0
# Abtastrate für die Tonwiedergabe [1/s].
abtastrate = 44100
# Federkonstante [N/m].
D = 7643.02
# Masse [kg].
m = 1e-3
# Reibungskoeffizient [kg/s].
b = 0.005
# Anfangsauslenkung [m].
x0 = 1e-3
# Anfangsgeschwindigkeit [m/s].
v0 = 0


def dgl(t, u):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    x, v = np.split(u, 2)
    F = - D * x - b * v
    return np.concatenate([v, F / m])


# Anfangszustand.
u0 = np.array([x0, v0])

# Löse die Differentialgleichung an den durch die Abtastrate
# vorgegebenen Zeitpunkten.
t = np.arange(0, t_max, 1 / abtastrate)
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, t_eval=t)
x, v = result.y

# Skaliere das Signal so, dass es in den Wertebereich von
# ganzen 16-bit Zahlen passt (-32768 ... +32767) und wandle
# es anschließend in 16-bit-Integers um.
audiodaten = np.int16(x / np.max(np.abs(x)) * 32767)

# Gib das Signal als Audiodatei im wav-Format aus.
scipy.io.wavfile.write('output.wav', abtastrate, audiodaten)

# Gib das Signal als Sound aus.
sounddevice.play(audiodaten, abtastrate, blocking=True)

# Erzeuge eine Figure und eine Axes und plotte den
# Zeitverlauf der Auslenkung.
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$t$ [s]')
ax.set_ylabel('$x$ [mm]')
ax.plot(t, 1e3 * x)

# Erzeuge eine Ausschnittvergrößerung.
ax_inset = ax.inset_axes([0.55, 0.67, 0.4, 0.25])
ax_inset.set_xlabel('$t$ [s]')
ax_inset.set_ylabel('$x$ [mm]')
ax_inset.set_xlim(0.5, 0.52)
ax_inset.set_ylim(-0.4, 0.4)
ax_inset.plot(t, 1e3 * x)

# Zeige den Plot an.
plt.show()
