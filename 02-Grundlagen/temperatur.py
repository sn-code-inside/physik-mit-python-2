"""Rechne Temperaturangaben von °C in °F um."""


def fahrenheit_von_celsius(grad_c):
    """Wandle eine Temperaturangabe von °C in °F um.

    Die Berechnung erfolgt auf Grundlage der beiden Fixpunkte
    der Temperaturskalen: 0 °C = 32 °F und 100 °C = 212 °F

    Args:
        grad_c (float):
            Temperaturangabe in °C.

    Returns:
        float: Temperaturangabe in °F.
    """
    return grad_c * 1.8 + 32


eingabe = input('Geben Sie eine Temperatur in °C ein: ')
temp_c = float(eingabe)
temp_f = fahrenheit_von_celsius(temp_c)
print(f'{temp_c} °C entsprechen {temp_f} °F.')
