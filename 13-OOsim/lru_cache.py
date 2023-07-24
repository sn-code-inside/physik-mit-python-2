"""Funktionsweise des Dekorators lru_cache."""

import functools
import random
import time


@functools.lru_cache(maxsize=10)
def f(x):
    print('       Die Berechnung dauert ganz schön lange')
    time.sleep(1)
    return 2 * x


for i in range(100):
    x = random.randint(1, 15)
    print(f'f({x})')
    y = f(x)
    print(f'Ergebnis: {y}')
