"""Gib eine Multiplikationstabelle für das kleine 1x1 aus."""

import numpy as np

faktoren = np.arange(1, 10)
einmaleins = faktoren * faktoren.reshape(-1, 1)
print(einmaleins)
