"""Funktionsweise eines Dekorators."""


# Definiere einen Dekorator.
def logge_aufruf(f):
    """Gib Aufrufe der Funktion f auf dem Bildschirm aus."""
    def innere_funktion(x):
        print(f'Funktionsaufruf "{f.__name__}({x})".')
        y = f(x)
        print(f'Funktion "{f.__name__}" wurde beendet.')
        return y
    return innere_funktion


# Definiere eine gewöhnliche Funktion mit Dekorator.
@logge_aufruf
def wurzel(x):
    return x ** (1/2)


# Rufe die dekorierte Funktion auf.
wurzel_von_zwei = wurzel(2)
