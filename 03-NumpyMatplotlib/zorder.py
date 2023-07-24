"""Demonstration der zorder in Matplotlib."""

import numpy as np
import matplotlib.pyplot as plt

x_grob = np.array([0, 2, 4, 8])
x_fein = np.linspace(0, 10, 500)

fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x_grob, x_grob ** 2, 'ro')
ax1.plot(x_fein, x_fein ** 2, 'b-', linewidth=2)

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x_grob, x_grob ** 2, 'ro', zorder=2)
ax2.plot(x_fein, x_fein ** 2, 'b-', linewidth=2, zorder=1)

plt.show()
