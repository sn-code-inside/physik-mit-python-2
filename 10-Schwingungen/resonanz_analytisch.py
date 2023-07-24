"""Halblogarithmische Darstellung einer Resonanzkurve."""

import numpy as np
import matplotlib.pyplot as plt

# Eigenkreisfrequenz des Systems [1/s].
omega0 = 1.0
# Abklingkoeffizienten [1/s].
abklingkoeffizienten = np.array([0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
# Anregungsfrequenzen [Hz].
omegas = np.logspace(-1, 1, 500)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(10, 5))
fig.set_tight_layout(True)

# Erzeuge eine Axes für die Amplitude.
ax_amplitude = fig.add_subplot(1, 2, 1)
ax_amplitude.set_xscale('log')
ax_amplitude.set_xlabel('$\\omega / \\omega_0$')
ax_amplitude.set_ylabel('Amplitude')
ax_amplitude.set_ylim(0, 6)
ax_amplitude.grid()

# Erzeuge eine zweite Axes für die Phase.
ax_phase = fig.add_subplot(1, 2, 2)
ax_phase.set_xscale('log')
ax_phase.set_xlabel('$\\omega / \\omega_0$')
ax_phase.set_ylabel('Phasenverschiebung [rad]')
ax_phase.grid()

# Plotte den Amplituden- und Phasenverlauf als Funktion der
# Anregungskreisfrequenz.
for delta in abklingkoeffizienten:
    x = omega0 ** 2 / (
            omega0 ** 2 - omegas ** 2 + 2 * 1j * delta * omegas)
    labeltext = f'$\\delta$={delta/omega0:.1f}'
    ax_amplitude.plot(omegas / omega0, np.abs(x), label=labeltext)
    ax_phase.plot(omegas / omega0, -np.angle(x), label=labeltext)

# Erzeuge nur für die erste Axes eine Legende.
ax_amplitude.legend()

# Zeige die Figure an.
plt.show()
