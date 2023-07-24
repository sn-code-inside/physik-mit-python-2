"""Animation von verkürzten Zykloiden.

Die Bahnkurve eines Punktes wird dargestellt, der an einem
rollenden Rad befestigt ist.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Radius des Rades [m].
radius_rad = 0.35
# Abstand des betrachteten Punktes von der Drehachse [m].
radius_punkt = 0.35
# Geschwindigkeit [m/s].
v_rad = 5.0
# Anzahl der Radumdrehungen.
anzahl_umdrehungen = 2
# Anzahl der Zeitschritte pro Umlauf.
n_schritte = 200

# Winkelgeschwindigkeit [1/s].
omega = v_rad / radius_rad

# Berechne die Umlaufdauer [s].
T = 2 * np.pi / omega

# Lege die Simulationsdauer fest.
t_max = T * anzahl_umdrehungen

# Lege die Zeitschrittweite fest.
dt = T / n_schritte

# Erzeuge ein Array von Zeitpunkten.
t = np.arange(0, t_max, dt)

# Lege einen Skalierungsfaktoren für die Vektorpfeile
# der Beschleunigung und die Geschwindigkeit fest.
scal_a = 1 / omega ** 2
scal_v = 1 / omega

# Berechne die Position des Massenpunktes für jeden Zeitpunkt.
r = np.empty((t.size, 2))
r[:, 0] = radius_punkt * np.cos(-omega * t - np.pi / 2) + v_rad * t
r[:, 1] = radius_punkt * np.sin(-omega * t - np.pi / 2) + radius_rad

# Berechne den Geschwindigkeits- und Beschleunigungsvektor.
v = (r[1:] - r[:-1]) / dt
a = (v[1:] - v[:-1]) / dt

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-0.5 * radius_rad, np.max(r[:, 0]) + 0.5 * radius_rad)
ax.set_ylim(-0.5 * radius_rad, 2.5 * radius_rad)
ax.set_aspect('equal')
ax.grid()

# Plotte die Kreisbahn.
plot_bahn, = ax.plot(r[:, 0], r[:, 1])

# Erzeuge einen Punkt, der die Position der Masse darstellt.
plot_punkt, = ax.plot([], [], 'o', color='blue')

# Erzeuge zwei Textfelder für die Anzeige des aktuellen
# Geschwindigkeits- und Beschleunigungsbetrags.
text_v = ax.text(0.1, 0.2, '',
                 color='red', transform=ax.transAxes)
text_a = ax.text(0.6, 0.2, '', color='black',
                 transform=ax.transAxes)

# Erzeuge Pfeile für die Geschwindigkeit und die Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6,
                                      head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='black',
                                      arrowstyle=style)

# Füge die Grafikobjekte zur Axes hinzu.
ax.add_patch(pfeil_v)
ax.add_patch(pfeil_a)

# Erzeuge einen Kreis, der das Rad darstellt und füge ihn zu Axes
# hinzu. Das zusätzliche Argument visible=False sorgt dafür, dass
# der Kreis erst angezeigt wird, wenn die update-Funktion
# aufgerufen wird.
kreis_rad = mpl.patches.Ellipse((0, 0),
                                width=2*radius_rad,
                                height=2*radius_rad,
                                fill=False, color='black',
                                linewidth=0.5, visible=False)
ax.add_patch(kreis_rad)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Setze den Start- und Endpunkt des Geschwindigkeitspfeils
    # und zeige den Geschwindigkeitsbetrag an.
    if n < len(v):
        pfeil_v.set_positions(r[n], r[n] + scal_v * v[n])
        text_v.set_text(f'$v$ = {np.linalg.norm(v[n]):.1f} m/s')

    # Setze den Start- und Endpunkt des Beschleunigungspfeils
    # und zeige den Beschleunigungsbetrag an.
    if n < len(a):
        pfeil_a.set_positions(r[n], r[n] + scal_a * a[n])
        text_a.set_text(f'$a$ = {np.linalg.norm(a[n]):.1f} m/s²')

    # Aktualisiere die Position des Punktes.
    plot_punkt.set_data(r[n])

    # Aktualisiere die Postion des Rades und mache es sichtbar.
    kreis_rad.set_center((v_rad * n * dt, radius_rad))
    kreis_rad.set_visible(True)

    return kreis_rad, plot_punkt, pfeil_v, pfeil_a, text_a, text_v


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
