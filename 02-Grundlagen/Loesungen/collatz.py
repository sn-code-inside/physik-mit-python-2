"""Das Collatz-Problem."""


def collatz(n):
    """Berechne die Collatz-Folge mit Startzahl n.

    Args:
        n (int): Die Startzahl n >= 1.

    Returns:
        list: Liste der Elemente der Folge bis zur Zahl 1.
    """
    # Definiere die Liste der Folgenglieder und initialisiere
    # diese mit der Startzahl.
    collatzfolge = [n]

    # Führe die Iteration durch. Es ist dabei wichtig für das
    # Teilen durch 2 den Operator // zu verwenden, weil sonst
    # eine Gleitkommazahl zurückgegeben wird und es zu
    # Rundungsfehlern kommt.
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        collatzfolge.append(n)

    return collatzfolge


def laengste_collatz_folge(n):
    """Suche die längste Collatz-Folge mit Startzahl bis n.

    Args:
        n (int): Größte Startzahl, bis zu der die Folgen
                 untersucht werden.

    Returns:
        list: Die längste Collatz-Folge, die gefunden wurde.
    """
    laengste_folge = []

    for startzahl in range(1, n + 1):
        folge = collatz(startzahl)
        if len(folge) > len(laengste_folge):
            laengste_folge = folge

    return laengste_folge


# Suche die längste Collatz-Folge bis zu einer Startzahl
# von einer Einhunderttausend.
max_startzahl = 100000
folge = laengste_collatz_folge(max_startzahl)
print(f"Die längste Collatz-Folge im Bereich bis {max_startzahl} "
      f"beginnt mit der Startzahl {folge[0]} und benötigt "
      f"{len(folge)} Schritte, bis die Zahl 1 erreicht wird.")
