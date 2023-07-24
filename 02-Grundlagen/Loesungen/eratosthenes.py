"""Das Sieb des Eratosthenes."""

# Zahl bis zu der nach Primzahlen gesucht werden soll.
obere_grenze = 100

# Erzeuge eine Liste die angibt, welche Zahlen als Primzahlen
# infrage kommen. Zu Beginn sind das alle Zahlen außer 0 und 1.
# Wir nehmen hier bewusst die 0 mit in die Liste auf, damit der
# Index in der Liste mit der entsprechenden Zahl übereinstimmt.
primzahlkandidaten = [False, False] + (obere_grenze - 2) * [True]

# Die erste Primzahl ist also die Zahl 2.
primzahl = 2
while primzahl ** 2 < obere_grenze:
    # Streiche alle Vielfachen der Primzahl aus der Liste der
    # Kandidaten. Wir müssen dabei keine Vielfachen mit einer
    # kleineren Zahl beachten, da diese sicher schon vorher aus
    # der Kandidatenliste entfernt worden sind.
    for index in range(primzahl ** 2, obere_grenze, primzahl):
        primzahlkandidaten[index] = False
    # Suche die nächste Zahl in der Liste, die noch nicht
    # herausgestrichen wurde.
    primzahl = primzahlkandidaten.index(True, primzahl + 1)

# Erzeuge nun eine Liste aller gefundenen Primzahlen.
primzahlen = []
for zahl, ist_prim in enumerate(primzahlkandidaten):
    if ist_prim:
        primzahlen.append(zahl)

# Gib die Liste der Primzahlen aus.
print(primzahlen)
