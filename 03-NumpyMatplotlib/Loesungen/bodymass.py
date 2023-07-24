"""Körpermasse in Abhängigkeit von der Körpergröße.

Da Programm stellt die Körpermasse in Abhängigkeit von der
Körpergröße für verschiedene 'Normmaße' dar. Diese sind:
    - das Normalgewicht (Körpergröße in kg minus 100),
    - das Idealgewicht (Normalgewicht minus 10 %),
    - der Body-Mass-Index von 20 kg/m² (Grenze zum Untergewicht
      bei erwachsenen Männern),
    - der Body-Mass-Index von 25 kg/m² (Grenze zum Übergewicht
      bei erwachsenen Männern),
    - der Body-Mass-Index von 22,5 kg/m² (mittlerer Wert zwischen
      den beiden Grenzwerte für Männer).
"""

import numpy as np
import matplotlib.pyplot as plt

# Bereich der betrachteten Körpergrößen [m].
koerpergroessen = np.linspace(1.70, 2.10, 500)

# Werte für den Body-Mass-Index [kg/m²].
body_mass_indizes = np.array([25, 22.5, 20], dtype=float)

# Klassische Formeln für Normal- und Idealgewicht [kg].
normalgewichte = koerpergroessen * 100 - 100
idealgewichte = normalgewichte * 0.9

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Körpergröße [m]')
ax.set_ylabel('Körpermasse [kg]')
ax.grid()

# Plotte das Normal- und Idealgewicht.
ax.plot(koerpergroessen, normalgewichte, label='Normalgewicht')
ax.plot(koerpergroessen, idealgewichte, label='Idealgewicht')

# Plotte das Körpergewicht zu den festgelegten Body-Mass-Indizes.
for bmi in body_mass_indizes:
    koerpergewichte = bmi * koerpergroessen ** 2
    ax.plot(koerpergroessen, koerpergewichte,
            linestyle='--', label=f'BMI = {bmi} kg/m²')

# Erzeuge eine Legende und zeige die Grafik an.
ax.legend()
plt.show()
