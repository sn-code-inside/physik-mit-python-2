"""Schleife über eine Liste mit enumerate."""

primzahlen = [2, 3, 5, 7, 11, 13, 17, 19, 23]
for i, p in enumerate(primzahlen):
    print(f'Die {i+1}-te Primzahl ist {p}.')
