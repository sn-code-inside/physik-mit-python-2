"""Simulation des Ziegenproblems."""

import random


def ziegenproblem(wahl_aendern, druckausgabe=False):
    """Simuliere eine Spielrunde des Ziegenproblems.

    Args:
        wahl_aendern (bool): Legt fest, ob der Kandidat seine
                             Wahl nach dem Öffnen eines Tores
                             immer ändert (True) oder immer
                             beibehält (False).
        druckausgabe (bool): Legt fest, ob der Spielablauf als
                             Text ausgegeben wird.
    Returns:
        bool: Der Kandidat hat gewonnen (True) oder erhält den
              Trostpreis (False).
    """
    # Erstelle ein Tuple mit den Torbezeichnungen.
    tore = ('Tor 1', 'Tor 2', 'Tor 3')

    # Platziere den Gewinn hinter einem zufällig gewählten Tor.
    tor_gewinn = random.choice(tore)

    # Lasse den Kandidaten ein zufälliges Tor öffnen.
    tor_wahl1 = random.choice(tore)

    # Erstelle ein Tuple mit den Toren, die prinzipiell geöffnet
    # werden können.
    tore_oeffenbar = tuple(set(tore) - {tor_gewinn, tor_wahl1})

    # Wähle aus diesem Tuple zufällig ein Tor, das geöffnet wird.
    tor_offen = random.choice(tore_oeffenbar)

    # Lasse sich den Kandidaten ggf. umentscheiden.
    if wahl_aendern:
        tor_wahl2, = set(tore) - {tor_wahl1, tor_offen}
    else:
        tor_wahl2 = tor_wahl1

    # Bestimme, ob der Kandidat gewonnen hat.
    hat_gewonnen = tor_wahl2 == tor_gewinn

    # Gib ggf. den Spielablauf aus.
    if druckausgabe:
        print(f'Der Gewinn befindet sich hinter {tor_gewinn}.')
        print(f'Der Kandidat wählt {tor_wahl1}.')
        print(f'Das {tor_offen} wird geöffnet.')
        print(f'Der Kandidat wählt {tor_wahl2}.')
        if hat_gewonnen:
            print('Der Kandidat hat gewonnen.')
        else:
            print('Der Kandidat erhält den Trostpreis.')

    return hat_gewonnen


# Mache einen Testlauf mit jeder der beiden Strategien und
# gib dabei den Spielablauf aus:
print('Strategie: Torwahl immer beibehalten')
ziegenproblem(wahl_aendern=False, druckausgabe=True)
print()
print('Strategie: Tor immer wechseln:')
ziegenproblem(wahl_aendern=True, druckausgabe=True)
print()

# Führe jeweils n Spiele mit jeder der beiden Strategien durch
# und zähle, wie häufig der Kandidat gewinnt.
anzahl_spiele = 100000
anzahl_gewinne1 = 0
anzahl_gewinne2 = 0
for i in range(anzahl_spiele):
    if ziegenproblem(wahl_aendern=False):
        anzahl_gewinne1 += 1
    if ziegenproblem(wahl_aendern=True):
        anzahl_gewinne2 += 1

# Bestimme die relative Häufigkeit, mit der bei den beiden
# Strategien gewonnen wurde.
gewinnhaeufigkeit1 = anzahl_gewinne1 / anzahl_spiele
gewinnhaeufigkeit2 = anzahl_gewinne2 / anzahl_spiele

# Gib das Ergebnis als Text aus.
print(f'Anzahl der Spiele: {anzahl_spiele}')
print('Relative Häufigkeit der Gewinne mit der Strategie:')
print(f'  - Torwahl immer beibehalten:'
      f' {100 * gewinnhaeufigkeit1:.1f} %')
print(f'  - Tor immer wechseln:'
      f'        {100 * gewinnhaeufigkeit2:.1f} %')
