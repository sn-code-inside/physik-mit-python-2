"""Simulation des Ausrollens eines Fahrzeugs mit solve_ivp.

Die Verwendung der Option dense_output wird demonstriert.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Zeitdauer, die simuliert werden soll [s].
t_max = 200
# Masse des Fahrzeugs [kg].
m = 10.0
# Reibungskoeffizient [kg / m].
b = 2.5
# Anfangsort [m].
x0 = 0
# Anfangsgeschwindigkeit [m/s].
v0 = 10.0


def F(v):
    """Berechne die Kraft als Funktion der Geschwindigkeit v."""
    return - b * v * np.abs(v)


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    x, v = u
    return np.array([v, F(v) / m])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.array([x0, v0])

# Löse die Bewegungsgleichung von t = 0 bis t = t_max.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0,
                                   dense_output=True)

# Gib die Statusmeldung aus und verteile das Ergebnis an den
# tatsächlich gewählten Stützstellen auf entsprechende Arrays.
print(result.message)
t_stuetz = result.t
x_stuetz, v_stuetz = result.y

# Erzeuge ein Array mit fein aufgelösten Zeitschritten und werte
# die Interpolation, die für die Lösung der Differential-
# gleichung verwendet wurde, an diesen Zeitpunkten aus.
t_interp = np.linspace(0, t_max, 1000)
x_interp, v_interp = result.sol(t_interp)

# Berechne die analytische Lösung für die gleichen Zeitschritte.
x_analyt = m / b * np.log(1 + v0 * b / m * t_interp)
v_analyt = v0 / (1 + v0 * b / m * t_interp)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)

# Plotte das Geschwindigkeits-Zeit-Diagramm.
ax_geschw = fig.add_subplot(1, 2, 1)
ax_geschw.set_xlabel('$t$ [s]')
ax_geschw.set_ylabel('$v$ [m/s]')
ax_geschw.grid()
ax_geschw.plot(t_interp, v_analyt, '-b', label='analytisch')
ax_geschw.plot(t_stuetz, v_stuetz, '.r',
               label='simuliert (Stützstellen)')
ax_geschw.plot(t_interp, v_interp, '-r',
               label='simuliert (Interpolation)')
ax_geschw.legend()

# Plotte das Orts-Zeit-Diagramm.
ax_ort = fig.add_subplot(1, 2, 2)
ax_ort.set_xlabel('$t$ [s]')
ax_ort.set_ylabel('$x$ [m]')
ax_ort.grid()
ax_ort.plot(t_interp, x_analyt, '-b', label='analytisch')
ax_ort.plot(t_stuetz, x_stuetz, '.r',
            label='simuliert (Stützstellen)')
ax_ort.plot(t_interp, x_interp, '-r',
            label='simuliert (Interpolation)')
ax_ort.legend()

# Zeige die Grafik an.
plt.show()
