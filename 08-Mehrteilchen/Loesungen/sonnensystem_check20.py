"""Überprüfung der Sonnensystem-Simulation.

Das Programm liest die Daten des Programms `sonnensystem_sim.py`
aus der Datei `ephemeriden.npz` ein und stellt die Positionen der
Himmelskörper nach einer Zeitdauer von 20 Jahren im Vergleich zu den
in der Horizon-Datenbank verfügbaren Positionen dar.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# Lies die Simulationsdaten ein.
dat = np.load('ephemeriden.npz')
datum_t0 = datetime.datetime.fromtimestamp(float(dat['datum_t0']))
dt, namen = dat['dt'], dat['namen']
AE, m, t, r, v = dat['AE'], dat['m'], dat['t'], dat['r'],  dat['v']

# Farben für die Darstellung der Planetenbahnen.
farben = ['gold', 'darkcyan', 'orange', 'blue', 'red', 'brown',
          'olive', 'green', 'slateblue', 'black', 'gray']

# Lege das Datum des Vergleichszeitpunkts fest auf den
# 01.01.2042 um 00:00 Uhr UTC.
datum_t1 = datetime.datetime(2042, 1, 1)

# Positionen [m] der Himmelskörper am Vergleichszeitpunkt.
# Quelle: https://ssd.jpl.nasa.gov/horizons/
r0 = AE * np.array([
    [+5.2956744196e-03, +3.5373208774e-03, -1.7762478953e-04],
    [+3.4171812310e-01, +6.6674828797e-02, -2.5859152637e-02],
    [+1.2952737444e-01, -7.1308420376e-01, -1.7221663653e-02],
    [-1.6725418218e-01, +9.7162931245e-01, -2.6967253277e-04],
    [-8.3466427219e-01, +1.4053916377e+00, +4.9780714866e-02],
    [-3.7511541461e+00, -3.8945192283e+00, +1.0014419206e-01],
    [-8.5101103922e+00, -4.7376987182e+00, +4.2105896416e-01],
    [-1.1814417811e+01, +1.4219380653e+01, +2.0575041937e-01],
    [+2.3957311705e+01, +1.7720980098e+01, -9.1705873548e-01],
    [+3.4620664381e+00, +2.1083640542e+00, -4.1231334328e-01],
    [-6.4657155680e-01, +4.2319235633e-01, +2.2970970727e-01]])

# Ziehe die Schwerpunktsposition von den Orten ab.
r0 -= m @ r0 / np.sum(m)

# Erzeuge eine Figure und eine 3D-Axes.
fig = plt.figure(figsize=(9, 6))
fig.tight_layout()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('$x$ [AE]')
ax.set_ylabel('$y$ [AE]')
ax.set_zlabel('$z$ [AE]')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.grid()

# Plotte für jeden Himmelskörper die Bahnkurve und füge die
# Legende hinzu. Wir reduzieren die Linienstärke, damit man die
# aktuelle Position der Himmelskörper besser erkennen kann.
for ort, name, farbe in zip(r, namen, farben):
    ax.plot(ort[0] / AE, ort[1] / AE, ort[2] / AE, '-',
            label=name, color=farbe, linewidth=0.2)

# Suche den Index in den Simulationsdaten, der am nächsten am
# gesuchten Zeitpunkt liegt.
delta_t = (datum_t1 - datum_t0).total_seconds()
index_zeit = np.argmin(np.abs(t - delta_t))

# Gib eine Tabelle der Positionsabweichungen aus.
for i in range(len(namen)):
    d = np.linalg.norm(r[i, :, index_zeit] - r0[i])
    print(f'{namen[i]:10s}: {d/AE:.5f} AE')

# Plotte die simulierten Positionen der Himmelskörper mit
# offenen Kreisen und füge die Beschriftung hinzu.
for ort, name, farbe in zip(r, namen, farben):
    ax.plot([ort[0, index_zeit] / AE],
            [ort[1, index_zeit] / AE],
            [ort[2, index_zeit] / AE], 'o',
            fillstyle='none', color=farbe)
ax.legend()

# Plotte die Positionen der Himmelskörper aus der Horizons-
# Datenbank mit offenen Rauten (engl. diamond).
for ort, farbe in zip(r0, farben):
    ax.plot([ort[0] / AE],
            [ort[1] / AE],
            [ort[2] / AE], 'D',
            fillstyle='none', color=farbe)

# Zeige den Plot an.
plt.show()
