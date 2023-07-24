"""Animierte Darstellung einer ebenen Welle."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Erzeuge die x-Werte von 0 bis 20 in 500 Schritten.
x = np.linspace(0, 20, 500)

# Definiere die Kreisfrequenz omega, die Wellenzahl k und
# die Zeitschrittweite delta_t.
omega = 1.0
k = 1.0
delta_t = 0.04

# Erzeuge die Figure und das Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('Ort x')
ax.set_ylabel('u(x, t)')
ax.grid()

# Erzeuge einen leeren Plot und ein leeres Textfeld.
plot, = ax.plot([], [])
text = ax.text(0.5, 1.05, '')


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Berechne die Funktionswerte u zum Zeitpunkt t.
    t = n * delta_t
    u = np.cos(k * x - omega * t)

    # Aktualisiere den Plot und das Textfeld.
    plot.set_data(x, u)
    text.set_text(f't = {t:5.1f}')

    # Gib ein Tupel mit den Grafikelementen zurück, die neu
    # dargestellt werden müssen.
    return plot, text


# Erzeuge das Animationsobjekt.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)

# Starte die Animation.
plt.show()
