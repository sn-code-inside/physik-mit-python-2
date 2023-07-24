"""For-Schleife über zwei Listen mit Indizierung."""

liste1 = [2, 4, 6, 8, 10]
liste2 = [3, 5, 7, 9, 11]
for i in range(len(liste1)):
    a = liste1[i]
    b = liste2[i]
    print(f'{a:3d}    {b:3d}')
