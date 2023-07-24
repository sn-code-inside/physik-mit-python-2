"""Animation der Reihendefinition der Takagifunktion."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Anzahl der Punkte, die dargestellt werden sollen.
n_punkte = 2000

# Erzeuge ein Array mit den x-Werten.
x = np.linspace(0, 1, n_punkte)

# Erzeuge die Figure und das Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Approximation der Takagifunktion')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()

# Erzeuge einen Plot und ein leeres Textfeld.
plot, = ax.plot([], [])
text = ax.text(0.05, 0.9, '')


def s(x):
    """Berechne den Abstand von x zur nächsten ganzen Zahl."""
    return np.abs(x - np.round(x))


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Berechne die ersten n Summanden der Takagifunktion.
    y = np.zeros(n_punkte)
    for k in range(n):
        y += s(2**k * x) / 2**k

    # Aktualisiere den Plot und das Textfeld.
    plot.set_data(x, y)
    text.set_text(f'n = {n}')

    # Gib ein Tupel mit den Grafikelementen zurück, die neu
    # dargestellt werden müssen.
    return plot, text


# Erzeuge das Animationsobjekt.
ani = mpl.animation.FuncAnimation(fig, update,
                                  frames=range(51),
                                  interval=300, blit=True)

# Starte die Animation.
plt.show()
