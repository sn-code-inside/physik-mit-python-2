"""Definition der Klasse Koerper mit weiteren Methoden."""


class Koerper:
    """Ein Himmelskörper.

    Args:
        name (str):
            Name des Himmelskörpers.
        masse (float):
            Masse des Himmelskörpers [kg].
    """

    def __init__(self, name, masse):
        self.name = name
        """str: Der Name des Himmelskörpers."""
        self.masse = masse
        """float: Die Masse des Himmelskörpers [kg]."""

    def erdmassen(self):
        """Berechne die Masse in Vielfachen der Erdmasse."""
        return self.masse / 5.9722e24

    def __str__(self):
        """Erzeuge eine Beschreibung des Körpers als String."""
        return f'Körper {self.name}: m = {self.masse} kg'


if __name__ == '__main__':
    planet = Koerper('Jupiter', 1.89813e27)
    print(planet.name)
    print(planet.masse)

    print(planet)
    print(f'Der Planet {planet.name} hat eine Masse von '
          f'{planet.erdmassen():.1f} Erdmassen')
