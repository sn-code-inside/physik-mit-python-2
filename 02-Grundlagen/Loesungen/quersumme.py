"""Berechnung von Quersummen."""


def quersumme(zahl):
    """Berechne die Quersumme des ganzzahligen Betrags der Zahl."""
    # Betrachte nur den ganzzahligen Betrag der Zahl.
    zahl = int(abs(zahl))

    # Um die Quersumme einer Zahl zu berechnen, muss man über die
    # Ziffern der Zahl im Dezimalsystem summieren. Um
    # nacheinander die Ziffern einer Zahl zu erhalten, kann man
    # die Zahl immer wieder durch 10 teilen und dabei eine
    # ganzzahlige Division mit Rest durchführen. Der Rest bei
    # einer Division durch 10 ist jeweils die 'Einerstelle' der
    # Zahl.
    ergebnis = 0
    while zahl > 0:
        zahl, rest = zahl // 10, zahl % 10
        ergebnis += rest
    return ergebnis


# Teste die Funktion anhand einiger willkürlich gewählter Zahlen.
testzahlen = [43512123, 123412335, 1235, 1235.5, 0, 1, -12]
for x in testzahlen:
    print(f'Die Quersumme von {x} ist {quersumme(x)}')
