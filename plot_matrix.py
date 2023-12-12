#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
###################################################################################################

if len(sys.argv) != 2:
	print("Usage: " + str(sys.argv[0]) + " matrix_file")
	exit()
matName = sys.argv[1]

mat = np.genfromtxt(matName, dtype=complex)
mat = np.abs(mat)

countNonZero = 0
for i in mat:
	for ij in i:
		if ij != 0:
			countNonZero += 1
print(f"Non zero elements count: {countNonZero}")

vmax = 2
plt.imshow(mat, cmap="Greys", vmin=0, vmax=vmax)
plt.title(fr"{matName}, $vmax = {vmax}$")

plt.grid(zorder=50)

plt.colorbar()
plt.tight_layout()
plt.savefig(matName+"_plot.pdf")