"""Definition einer abgeleiteten Klasse."""

import math
from klasse1 import Koerper


class Planet(Koerper):
    """Ein Planet.

    Args:
        name (str):
            Name des Planeten
        masse (float):
            Masse des Planeten [kg].
        radius (float):
            Radius des Planeten [m].
    """

    def __init__(self, name, masse, radius):
        super().__init__(name, masse)
        self.radius = radius
        """float: Der Radius des Planeten [m]."""

    def volumen(self):
        """Berechne das Volumen des Planeten."""
        return 4 / 3 * math.pi * self.radius ** 3

    def dichte(self):
        """Berechne die Dichte des Planeten."""
        return self.masse / self.volumen()


if __name__ == '__main__':
    jupiter = Planet('Jupiter', 1.89813e27,  6.9911e7)
    rho = jupiter.dichte()
    print(jupiter)
    print(f'Dichte: {rho/1e3:.2f} g/cm³')
