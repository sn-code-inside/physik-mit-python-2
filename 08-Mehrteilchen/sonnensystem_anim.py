"""Animierte Darstellung des Sonnensystems.

Das Programm liest die Daten des Programms sonnensystem_sim.py
aus der Datei ephemeriden.npz ein und stellt das Sonnensystem
animiert dar.
"""

import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import mpl_toolkits.mplot3d

# Lies die Simulationsdaten ein.
dat = np.load('ephemeriden.npz')
tag, jahr, AE, G = dat['tag'], dat['jahr'], dat['AE'], dat['G']
dt, namen = dat['dt'], dat['namen']
m, t, r, v = dat['m'], dat['t'], dat['r'],  dat['v']
datum_t0 = datetime.datetime.fromtimestamp(float(dat['datum_t0']))

# Stelle nur jeden n-ten simulierten Zeitschritt dar, damit die
# Animation nicht zu langsam wird.
schritt = int(5 * tag / dt)

# Farben für die Darstellung der Himmelskörper.
farben = ['gold', 'darkcyan', 'orange', 'blue', 'red', 'brown',
          'olive', 'green', 'slateblue', 'black', 'gray']

# Anzahl der Himmelskörper und Dimension des Raumes.
n_koerper, n_dim = r.shape[:2]

# Berechne die verschiedenen Energiebeiträge.
E_kin = 1/2 * m @ np.sum(v * v, axis=1)
E_pot = np.zeros(t.size)
for i in range(n_koerper):
    for j in range(i):
        dr = np.linalg.norm(r[i] - r[j], axis=0)
        E_pot -= G * m[i] * m[j] / dr
E = E_pot + E_kin

# Berechne den Gesamtimpuls.
impuls = m @ v.swapaxes(0, 1)

# Berechne die Position des Schwerpunktes.
schwerpunkt = m @ r.swapaxes(0, 1) / np.sum(m)

# Berechne den Drehimpuls.
drehimpuls = m @ np.cross(r, v, axis=1).swapaxes(0, 1)

# Erzeuge eine Figure für die Plots der Erhaltungsgrößen.
fig_plots = plt.figure(figsize=(12, 6))
fig_plots.set_tight_layout(True)

# Erzeuge eine Axes und plotte die Energie.
ax_energ = fig_plots.add_subplot(2, 2, 1)
ax_energ.set_title('Energie')
ax_energ.set_xlabel('$t$ [d]')
ax_energ.set_ylabel('$E$ [J]')
ax_energ.grid()
ax_energ.plot(t / tag, E, label='$E$')

# Erzeuge eine Axes und plotte den Impuls.
ax_impuls = fig_plots.add_subplot(2, 2, 2)
ax_impuls.set_title('Impuls')
ax_impuls.set_xlabel('$t$ [d]')
ax_impuls.set_ylabel('$p$ [kg m / s]')
ax_impuls.grid()
ax_impuls.plot(t / tag, impuls[0, :], '-r', label='$p_x$')
ax_impuls.plot(t / tag, impuls[1, :], '-b', label='$p_y$')
ax_impuls.plot(t / tag, impuls[2, :], '-k', label='$p_z$')
ax_impuls.legend()

# Erzeuge eine Axes und plotte den Drehimpuls.
ax_drehimpuls = fig_plots.add_subplot(2, 2, 3)
ax_drehimpuls.set_title('Drehimpuls')
ax_drehimpuls.set_xlabel('$t$ [d]')
ax_drehimpuls.set_ylabel('$L$ [kg m² / s]')
ax_drehimpuls.grid()
ax_drehimpuls.plot(t / tag, drehimpuls[0, :], '-r', label='$L_x$')
ax_drehimpuls.plot(t / tag, drehimpuls[1, :], '-b', label='$L_y$')
ax_drehimpuls.plot(t / tag, drehimpuls[2, :], '-k', label='$L_z$')
ax_drehimpuls.legend()

# Erzeuge eine Axes und plotte die Schwerpunktskoordinaten.
ax_schwerpunkt = fig_plots.add_subplot(2, 2, 4)
ax_schwerpunkt.set_title('Schwerpunkt')
ax_schwerpunkt.set_xlabel('$t$ [d]')
ax_schwerpunkt.set_ylabel('$r_s$ [m]')
ax_schwerpunkt.grid()
ax_schwerpunkt.plot(t / tag, schwerpunkt[0, :], '-r',
                    label='$r_{s,x}$')
ax_schwerpunkt.plot(t / tag, schwerpunkt[1, :], '-b',
                    label='$r_{s,y}$')
ax_schwerpunkt.plot(t / tag, schwerpunkt[2, :], '-k',
                    label='$r_{s,z}$')
ax_schwerpunkt.legend()

# Erzeuge eine Figure und eine 3D-Axes für die Bahnkurven.
fig_bahn = plt.figure(figsize=(9, 6))
ax_bahn = fig_bahn.add_subplot(1, 1, 1, projection='3d')
ax_bahn.set_xlabel('$x$ [AE]')
ax_bahn.set_ylabel('$y$ [AE]')
ax_bahn.set_zlabel('$z$ [AE]')
ax_bahn.set_xlim([-3, 3])
ax_bahn.set_ylim([-3, 3])
ax_bahn.set_zlim([-3, 3])
ax_bahn.grid()

# Plotte für jeden Himmelskörper die Bahnkurve und füge die
# Legende hinzu. Dabei plotten wir nur jede n_schritt-ten
# Zeitschritt.
for ort, name, farbe in zip(r, namen, farben):
    ort = ort[:, ::schritt] / AE
    ax_bahn.plot(ort[0], ort[1], ort[2], '-', label=name,
                 color=farbe)

# Erzeuge für jeden Himmelskörper einen Punktplot in der
# entsprechenden Farbe und speichere diesen in der Liste.
plots_himmelskoerper = []
for farbe in farben:
    plot, = ax_bahn.plot([], [], [], 'o', color=farbe)
    plots_himmelskoerper.append(plot)
ax_bahn.legend(loc='lower left')

# Füge ein Textfeld für die Anzeige der verstrichenen Zeit hinzu.
text_zeit = fig_bahn.text(0.5, 0.95, '')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    for plot, ort in zip(plots_himmelskoerper, r):
        plot.set_data_3d(ort[:, n].reshape(-1, 1) / AE)

    # Berechne das aktuelle Datum und stelle den Zeitpunkt dar.
    datum = datum_t0 + datetime.timedelta(seconds=t[n])
    text_zeit.set_text(f'{t[n] / jahr:.2f} Jahre: {datum:%d.%m.%Y}')
    return plots_himmelskoerper + [text_zeit]


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig_bahn, update,
                                  frames=range(0, t.size, schritt),
                                  interval=30)
plt.show()
