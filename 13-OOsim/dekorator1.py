"""Funktionsweise eines Dekorators."""


# Definiere einen Dekorator.
def logge_aufruf(f):
    """Gib Aufrufe der Funktion f auf dem Bildschirm aus."""
    def innere_funktion(x):
        print(f'Funktionsaufruf von "{f.__name__}({x})".')
        y = f(x)
        print(f'Ergebnis von "{f.__name__}({x})": {y}')
        return y
    return innere_funktion


# Definiere eine gewöhnliche Funktion.
def wurzel(x):
    return x ** (1/2)


# Dekoriere die Funktion mit dem Dekorator.
wurzel_kommentiert = logge_aufruf(wurzel)

# Rufe die dekorierte Funktion auf.
wurzel_von_zwei = wurzel_kommentiert(2)
