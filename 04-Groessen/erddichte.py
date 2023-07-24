"""Berechnung der mittleren Dichte der Erde."""

import math

# Mittlerer Erdradius [m].
R = 6371e3
# Masse der Erde [kg].
m = 5.972e24
# Berechne das Volumen.
V = 4 / 3 * math.pi * R**3
# Berechne die Dichte.
rho = m / V
# Gib das Ergebnis aus.
print(f'Die mittlere Erddichte beträgt {rho/1e3:.3f} g / cm³.')
