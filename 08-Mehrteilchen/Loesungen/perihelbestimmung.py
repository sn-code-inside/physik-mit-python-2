"""Periheldurchgänge von Himmelskörpern.

Das Programm berechnet mithilfe der von `sonnensystem_sim.py`
erzeugten Daten, den Abstand des Kometen 9P/Tempel 1 von der Erde
bzw. von der Sonne und gibt den Zeitpunkt des Perihels des
Kometen an.
"""

import numpy as np
import datetime
import matplotlib.pyplot as plt

# Lies die Simulationsdaten ein.
dat = np.load('ephemeriden.npz')
tag, AE = dat['tag'], dat['AE']
datum_t0 = datetime.datetime.fromtimestamp(float(dat['datum_t0']))
namen = dat['namen']
t, r = dat['t'], dat['r']

# Suche aus dem Array `namen` die entsprechenden Indizes heraus.
index_komet = np.where(namen == 'Tempel 1')[0][0]
index_erde = np.where(namen == 'Erde')[0][0]
index_sonne = np.where(namen == 'Sonne')[0][0]

# Berechne den Abstand zwischen Kometen und Erde sowie zwischen
# Kometen und Sonne.
d_erde = np.linalg.norm(r[index_komet] - r[index_erde], axis=0)
d_sonne = np.linalg.norm(r[index_komet] - r[index_sonne], axis=0)

# Gib die Zeitpunkte aus, bei denen der Abstand zwischen
# Komet und Sonne minimal ist.
zeitpunkte_perihel = []
for i_min in range(1, len(d_erde) - 1):
    if d_sonne[i_min - 1] > d_sonne[i_min] < d_sonne[i_min + 1]:
        # Bestimme das Kalenderdatum zum aktuellen Zeitpunkt
        # und gib dieses Zusammen mit der Anzahl der vergangenen
        # Tage aus.
        datum = datum_t0 + datetime.timedelta(seconds=t[i_min])
        print(f"Periheldurchgang nach {t[i_min]/tag:.0f} Tagen "
              f"am {datum:%d.%m.%Y %H:%M}")

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$t$ [Tage]')
ax.set_ylabel('$d$ [AE]')
ax.grid()

# Plotte die Abstände und erzeuge eine Legende.
ax.plot(t / tag, d_erde / AE,
        label=f'{namen[index_komet]} - Erde')
ax.plot(t / tag, d_sonne / AE,
        label=f'{namen[index_komet]} - Sonne')
ax.legend()

# Zeige die Grafik an.
plt.show()
