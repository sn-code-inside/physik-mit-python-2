"""Simulation des Stratosphärensprungs.

Für die Luftdichte werden tabellierte Messdaten geeignet
interpoliert.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

# Masse des Körpers [kg].
m = 90.0
# Erdbeschleunigung [m/s²].
g = 9.81
# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.47
# Anfangshöhe [m].
y0 = 39.045e3
# Anfangsgeschwindigkeit [m/s].
v0 = 0.0

# Messwerte: Luftdichte [kg/m³] in Abhängigkeit von der Höhe [m].
messwerte_h = 1e3 * np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                              11.02, 15, 20.06, 25, 32.16, 40])
messwerte_rho = np.array([1.225, 1.112, 1.007, 0.909, 0.819, 0.736,
                          0.660, 0.590, 0.526, 0.467, 0.414, 0.364,
                          0.195, 0.0880, 0.0401, 0.0132, 0.004])

# Erzeuge eine Interpolationsfunktion für die Luftdichte.
fill = (messwerte_rho[0], messwerte_rho[-1])
rho = scipy.interpolate.interp1d(messwerte_h, messwerte_rho,
                                 kind='cubic', bounds_error=False,
                                 fill_value=fill)


def F(y, v):
    """Bestimme die auf den Springer wirkende Kraft.

    Args:
        y (float):
            Aktuelle Höhe [m].
        v (float):
            Aktuelle Geschwindigkeit [m/s].

    Returns:
        float: Kraft [N].
    """
    Fg = -m * g
    Fr = -0.5 * rho(y) * cwA * v * np.abs(v)
    return Fg + Fr


def dgl(t, u):
    """Berechne die rechte Seite der Differentialgleichung."""
    y, v = u
    return np.array([v, F(y, v) / m])


def aufprall(t, u):
    """Ereignisfunktion: Detektiere das Erreichen des Erdbodens."""
    y, v = u
    return y


# Bende die Simulation beim Auftreten des Ereignisses.
aufprall.terminal = True

# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.array([y0, v0])

# Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde.
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                   events=aufprall,
                                   dense_output=True)
t_stuetz = result.t
y_stuetz, v_stuetz = result.y

# Berechne die Interpolation auf einem feinen Raster.
t_interp = np.linspace(0, np.max(t_stuetz), 1000)
y_interp, v_interp = result.sol(t_interp)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)

# Plotte das Geschwindigkeits-Zeit-Diagramm.
ax_geschw = fig.add_subplot(1, 2, 1)
ax_geschw.set_xlabel('$t$ [s]')
ax_geschw.set_ylabel('$v$ [m/s]')
ax_geschw.grid()
ax_geschw.plot(t_stuetz, v_stuetz, '.b')
ax_geschw.plot(t_interp, v_interp, '-b')

# Plotte das Orts-Zeit-Diagramm.
ax_ort = fig.add_subplot(1, 2, 2)
ax_ort.set_xlabel('$t$ [s]')
ax_ort.set_ylabel('$y$ [m]')
ax_ort.grid()
ax_ort.plot(t_stuetz, y_stuetz, '.b')
ax_ort.plot(t_interp, y_interp, '-b')

# Zeige die Grafik an.
plt.show()
