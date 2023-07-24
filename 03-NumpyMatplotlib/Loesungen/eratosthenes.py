"""Das Sieb des Eratosthenes."""

import math
import numpy as np

# Zahl bis zu der nach Primzahlen gesucht werden soll.
obere_grenze = 100

# Erzeuge ein Array vom Datentyp bool das angibt, welche Zahlen
# als Primzahlen infrage kommen. Zu Beginn sind das alle Zahlen
# außer 0 und 1. Wir nehmen hier bewusst die 0 mit in dem
# Array auf, damit der Array-Index mit der entsprechenden
# Zahl übereinstimmt.
primzahlkandidaten = np.ones(obere_grenze, dtype=bool)

# Markiere 0 und 1 als keine Primzahlen.
primzahlkandidaten[:2] = False

# Streiche nacheinander die Vielfachen von ganzen Zahlen aus dem
# Array der Kandidaten. Wir müssen dabei keine Vielfachen mit
# einer kleineren Zahl beachten, da diese sicher schon vorher aus
# der Kandidatenliste entfernt worden sind.
for i in range(2, int(math.sqrt(obere_grenze) + 1)):
    if primzahlkandidaten[i]:
        primzahlkandidaten[i * i::i] = False

# Primzahlen sind die Zahlen, die am Ende noch als Kandidaten
# markiert sind. Alternativ zu der hier vorgestellten Lösung könnte
# man hier auch die Funktion np.where wie folgt benutzen:
#    primzahlen, = np.where(primzahlkandidaten)
zahlen = np.arange(obere_grenze)
primzahlen = zahlen[primzahlkandidaten]

# Gib die Primzahlen aus.
print(primzahlen)
