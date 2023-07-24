"""Animation zur gleichförmigen Kreisbewegung."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Radius der Kreisbahn [m].
radius = 3.0
# Umlaufdauer [s].
T = 12.0
# Zeitschrittweite [s].
dt = 0.02
# Winkelgeschwindigkeit [1/s].
omega = 2 * np.pi / T

# Gib das analytische Ergebnis aus.
print(f'Bahngeschwindigkeit:       {radius * omega:.3f} m/s')
print(f'Zentripetalbeschleunigung: {radius * omega ** 2:.3f} m/s²')

# Erzeuge ein Array von Zeitpunkten für einen Umlauf.
t = np.arange(0, T, dt)

# Berechne die Position des Massenpunktes für jeden Zeitpunkt.
r = np.empty((t.size, 2))
r[:, 0] = radius * np.cos(omega * t)
r[:, 1] = radius * np.sin(omega * t)

# Berechne den Geschwindigkeits- und Beschleunigungsvektor.
v = (r[1:] - r[:-1]) / dt
a = (v[1:] - v[:-1]) / dt

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-1.2 * radius, 1.2 * radius)
ax.set_ylim(-1.2 * radius, 1.2 * radius)
ax.set_aspect('equal')
ax.grid()

# Plotte die Kreisbahn.
plot_bahn, = ax.plot(r[:, 0], r[:, 1])

# Erzeuge einen Punktplot, der die Position der Masse darstellt.
plot_punkt, = ax.plot([], [], 'o', color='blue')

# Erzeuge zwei Textfelder für die Anzeige des aktuellen
# Geschwindigkeits- und Beschleunigungsbetrags.
text_v = ax.text(0, 0.2, '', color='red')
text_a = ax.text(0, -0.2, '', color='black')

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


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Setze den Start- und Endpunkt des Geschwindigkeitspfeils
    # und zeige den Geschwindigkeitsbetrag an.
    if n < len(v):
        pfeil_v.set_positions(r[n], r[n] + v[n])
        text_v.set_text(f'$v$ = {np.linalg.norm(v[n]):.3f} m/s')

    # Setze den Start- und Endpunkt des Beschleunigungspfeils
    # und zeige den Beschleunigungsbetrag an.
    if n < len(a):
        pfeil_a.set_positions(r[n], r[n] + a[n])
        text_a.set_text(f'$a$ = {np.linalg.norm(a[n]):.3f} m/s²')

    # Aktualisiere die Position des Punktes.
    plot_punkt.set_data(r[n].reshape(-1, 1))

    return plot_punkt, pfeil_v, pfeil_a, text_a, text_v


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
