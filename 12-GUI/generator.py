"""Wirkungsweise von Generatoren."""


def generator():
    """Generiere die Folge 3, 7, 5."""
    yield 3
    yield 7
    yield 5


for k in generator():
    print(k)
