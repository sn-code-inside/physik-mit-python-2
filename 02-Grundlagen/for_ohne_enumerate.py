"""Schleife über eine Liste mit Index."""

primzahlen = [2, 3, 5, 7, 11, 13, 17, 19, 23]
for i in range(len(primzahlen)):
    p = primzahlen[i]
    print(f'Die {i+1}-te Primzahl ist {p}.')
