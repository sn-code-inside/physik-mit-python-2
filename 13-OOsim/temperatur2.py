"""Klasse zur Darstellung von Temperaturen mit Properties."""


class Temperatur:
    def __init__(self, kelvin=293.15):
        self.kelvin = kelvin

    @property
    def celsius(self):
        return self.kelvin - 273.15

    @celsius.setter
    def celsius(self, temperatur_celsius):
        self.kelvin = temperatur_celsius + 273.15
