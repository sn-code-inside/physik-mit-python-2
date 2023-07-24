"""Simulation zweier gekoppelter Masse-Feder-Schwinger."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import schraubenfeder

# Simulationszeit und Zeitschrittweite [s].
t_max = 42.0
dt = 0.01
# Masse der Körper [kg].
m = 1.0
# Federkonstante der äußeren Federn [N/m].
D = 35.808088530029416
# Federkonstante der Kopplungsfeder [N/m].
D_k = 1.8351645371640082
# Anfangslänge der äußeren Federn [m].
ruhelaenge = 0.3
# Anfangslänge der Kopplungsfeder [m].
ruhelaenge_k = 0.5
# Radius der Federn in der Darstellung [m].
federradius = 0.02
# Anfangsauslenkungen [m].
x0 = np.array([0.1, 0.0])
# Anfangsgeschwindigkeiten [m/s].
v0 = np.array([0.0, 0.0])


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    x1, x2, v1, v2 = u
    F1 = -D * x1 - D_k * (x1 - x2)
    F2 = -D * x2 - D_k * (x2 - x1)
    return np.array([v1, v2, F1 / m, F2 / m])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((x0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
x1, x2, v1, v2 = result.y

# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Erzeuge einen Plot der Auslenkung als Funktion der Zeit.
ax_zeitverlauf = fig.add_subplot(2, 1, 2)
ax_zeitverlauf.set_xlabel('$t$ [s]')
ax_zeitverlauf.set_ylabel('$x$ [m]')
ax_zeitverlauf.set_xlim(0, t_max)
ax_zeitverlauf.grid()
ax_zeitverlauf.plot(t, x1, 'b-', label='$x_1$')
ax_zeitverlauf.plot(t, x2, 'r-', label='$x_2$')
ax_zeitverlauf.legend()

# Erzeuge einen Plot zur animierten Darstellung.
ax_anim = fig.add_subplot(2, 1, 1)
ax_anim.set_xlim(-ruhelaenge / 20,
                 2 * ruhelaenge + ruhelaenge_k + ruhelaenge / 20)
ax_anim.set_aspect('equal')
ax_anim.set_xlabel('$x$ [m]')

# Mache die überflüssigen Achsenmarkierungen unsichtbar.
ax_anim.spines['top'].set_visible(False)
ax_anim.spines['left'].set_visible(False)
ax_anim.spines['right'].set_visible(False)
ax_anim.get_yaxis().set_visible(False)

# Linie zur Darstellung der aktuellen Zeit.
linie_t, = ax_zeitverlauf.plot([], [], '-k', linewidth=3, zorder=5)

# Lege die x-Position der linken und rechten Befestigung fest.
x_links = 0
x_rechts = 2 * ruhelaenge + ruhelaenge_k

# Definiere die Höhe des Plots relativ zur horizontalen Ausdehnung.
y_max = 0.08 * (x_rechts - x_links)

# Zeichne die beiden Befestigungen.
ax_anim.plot([x_links, x_links], [-y_max, y_max],
             'k-', linewidth=3, zorder=6)
ax_anim.plot([x_rechts, x_rechts], [-y_max, y_max],
             'k-', linewidth=3, zorder=6)

# Zeichne die Ruhelagen der beiden Massen ein.
ax_anim.plot([x_links + ruhelaenge, x_links + ruhelaenge],
             [-y_max, y_max],
             '--', color='gray', zorder=3)
ax_anim.plot([x_rechts - ruhelaenge, x_rechts - ruhelaenge],
             [-y_max, y_max],
             '--', color='gray', zorder=3)

# Die beiden Massenpunkte.
plot_masse1, = ax_anim.plot([], [], 'bo', zorder=5)
plot_masse2, = ax_anim.plot([], [], 'ro', zorder=5)

# Die drei Federn.
plot_feder_links, = ax_anim.plot([], [], 'k-', zorder=4)
plot_feder_rechts, = ax_anim.plot([], [], 'k-', zorder=4)
plot_feder_mitte, = ax_anim.plot([], [], 'g-', zorder=4)

# Textfeld zur Anzeige des aktuellen Zeitpunkts.
text = ax_anim.text(0.5, 0.8, '', transform=ax_anim.transAxes)


def update(n):
    """Aktualisiere die Grafik zum n-ten Zeitschritt."""
    # Aktuelle Position der Massen und der Befestigungspunkte.
    r_masse1 = np.array([x_links + ruhelaenge + x1[n], 0])
    r_masse2 = np.array([x_rechts - ruhelaenge + x2[n], 0])
    r_befest_links = np.array([x_links, 0.0])
    r_befest_rechts = np.array([x_rechts, 0.0])

    # Aktualisiere die Position der Massen.
    plot_masse1.set_data(r_masse1)
    plot_masse2.set_data(r_masse2)

    # Aktualisiere die linke Feder.
    plotdaten = schraubenfeder.data(r_befest_links, r_masse1,
                                    n_wdg=10, r0=federradius,
                                    a=federradius,
                                    l0=ruhelaenge)
    plot_feder_links.set_data(plotdaten)

    # Aktualisiere die mittlere Kopplungsfeder.
    plotdaten = schraubenfeder.data(r_masse1, r_masse2,
                                    n_wdg=10, r0=federradius,
                                    a=federradius,
                                    l0=ruhelaenge_k)
    plot_feder_mitte.set_data(plotdaten)

    # Aktualisiere die rechte Feder.
    plotdaten = schraubenfeder.data(r_masse2, r_befest_rechts,
                                    n_wdg=10, r0=federradius,
                                    a=federradius,
                                    l0=ruhelaenge)
    plot_feder_rechts.set_data(plotdaten)

    # Aktualisiere die Zeitanzeige.
    text.set_text(f'$t$ = {t[n]:5.1f} s')

    # Aktualisiere die Markierung der aktuellen Zeit.
    linie_t.set_data([[t[n], t[n]], ax_zeitverlauf.get_ylim()])

    return (plot_masse1, plot_masse2,
            plot_feder_links, plot_feder_rechts, plot_feder_mitte,
            text, linie_t)


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
