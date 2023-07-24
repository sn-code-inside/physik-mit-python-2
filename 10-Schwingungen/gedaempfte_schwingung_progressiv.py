"""Gedämpfte Schwingung mit einer progressiven Feder."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.io.wavfile
import sounddevice

# Zeitdauer der Simulation [s].
t_max = 1.0
# Abtastrate für die Tonwiedergabe [1/s].
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
ax.plot(t, x / 1e-3)

# Erzeuge eine Ausschnittvergrößerung.
ax_inset1 = ax.inset_axes([0.55, 0.70, 0.4, 0.25])
ax_inset1.plot(t, x / 1e-3)
ax_inset1.set_xlim(0.0, 0.02)
ax_inset1.set_ylim(-20.0, 20.0)
ax_inset1.set_xlabel('$t$ [s]')
ax_inset1.set_ylabel('$x$ [mm]')

# Erzeuge eine zweite Ausschnittvergrößerung.
ax_inset2 = ax.inset_axes([0.55, 0.17, 0.4, 0.25])
ax_inset2.plot(t, x / 1e-3)
ax_inset2.set_xlim(0.8, 0.82)
ax_inset2.set_ylim(-0.8, 0.8)
ax_inset2.set_xlabel('$t$ [s]')
ax_inset2.set_ylabel('$x$ [mm]')

# Zeige den Plot an.
plt.show()
