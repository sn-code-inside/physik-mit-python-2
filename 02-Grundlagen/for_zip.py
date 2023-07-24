"""For-Schleife über zwei Listen mit zip."""

liste1 = [2, 4, 6, 8, 10]
liste2 = [3, 5, 7, 9, 11]
for a, b in zip(liste1, liste2):
    print(f'{a:3d}    {b:3d}')
