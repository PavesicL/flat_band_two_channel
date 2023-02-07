#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
###################################################################################################

n = 16
matName = f"matrix_n{n}"

mat = np.genfromtxt(matName, dtype=complex)
mat = np.abs(mat)

countNonZero = 0
for i in mat:
	for ij in i:
		if ij != 0:
			countNonZero += 1
print(f"Non zero elements count: {countNonZero}")

if 1:
	vmax = 2
	plt.imshow(mat, cmap="Greys", vmin=0, vmax=vmax)
	plt.title(fr"$n = {n}$, $vmax = {vmax}$")

else:
	antiDiagonal = [ mat[i][-i] for i in range(1, len(mat))]
	xx = [i for i in range(len(mat)-1)]

	print(antiDiagonal)

	plt.plot(xx, antiDiagonal)

print(len(mat), len(mat)/14)
if 1:
	ticks = [-0.5 + i * 14 for i in range(1+len(mat)//14)]
	print(ticks)
	plt.xticks(ticks)
	plt.yticks(ticks)

plt.grid(zorder=50)

#plt.title(fr"$n = {n}$")
#plt.colorbar()
plt.tight_layout()
plt.show()