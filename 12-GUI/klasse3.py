"""Überschreiben einer Methode."""

from klasse2 import Planet


class Planet2(Planet):
    """Ein Planet mit schönerer Stringausgabe."""

    def __str__(self):
        """Gib eine Beschreibung des Planeten als String zurück."""
        return f'{self.name}: m = {self.erdmassen():.2f} Erdmassen'


if __name__ == '__main__':
    planet = Planet2('Jupiter', 1.89813e27,  6.9911e7)
    print(planet)
