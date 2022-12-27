#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
###################################################################################################

n = 31
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
	vmax = 0.1
	plt.imshow(mat, cmap="Greys", vmin=0, vmax=vmax)
	plt.title(fr"$n = {n}$, $vmax = {vmax}$")

else:
	antiDiagonal = [ mat[i][-i] for i in range(1, len(mat))]
	xx = [i for i in range(len(mat)-1)]

	print(antiDiagonal)

	plt.plot(xx, antiDiagonal)


if 0:
	if n%2 == 1:
		k=3
		ticks = [k, ]
		while k < len(mat) - 3 - 8 - 6:
			k += 8
			ticks.append(k)	
			k += 6
			ticks.append(k)

		ticks.append(k+8)

	else:
		k = 1
		ticks = [k,]
		k+= 6
		ticks.append(k)
		while k < len(mat) - 17:
			k += 7
			ticks.append(k)
			k += 7
			ticks.append(k)
		ticks.append(k+6)
		ticks.append(k+6+6)
		
	plt.xticks(ticks)
	plt.yticks(ticks)

plt.grid(zorder=50)

#plt.title(fr"$n = {n}$")
#plt.colorbar()
plt.tight_layout()
plt.show()