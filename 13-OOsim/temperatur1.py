"""Klasse zur Darstellung von Temperaturen."""


class Temperatur:
    def __init__(self, kelvin=293.15):
        self.kelvin = kelvin

    def get_celsius(self):
        return self.kelvin - 273.15

    def set_celsius(self, temperatur_celsius):
        self.kelvin = temperatur_celsius + 273.15
