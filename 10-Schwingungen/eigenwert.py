"""Bestimmung von Eigenwerten und -vektoren mit NumPy."""

import numpy as np

# Definiere die Matrix.
omega_0 = 1.0
omega_k = 0.1
matrix = np.array([[omega_0 ** 2 + omega_k ** 2, -omega_k ** 2],
                   [-omega_k ** 2, omega_0 ** 2 + omega_k ** 2]])

# Bestimme die Eigenwerte und die Eigenvektoren.
eigenwerte, eigenvektoren = np.linalg.eig(matrix)

# Der i-te Eigenvektor ist durch eigenvektoren[:, i] gegeben.
# Damit die Funktion zip die Eigenwerte und Eigenvektoren richtig
# zuordnet, muss eigenvektoren transponiert werden.
for eigenwert, eigenvekt in zip(eigenwerte, eigenvektoren.T):
    print(f'Eigenwert {eigenwert:5f}: '
          f'Eigenvektor ({eigenvekt[0]: .3}, {eigenvekt[1]: .3})')
