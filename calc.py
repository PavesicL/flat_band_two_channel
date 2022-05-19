#!/usr/bin/env python3


from helper import *
from observables_calculation import *
from matrix import generate_total_matrix, reorder_matrix_dM, reorder_matrix_phi, fourier_transform_matrix
import sys
from math import pi
import time
from numpy.linalg import eigh, eig
###################################################################################################

if len(sys.argv) != 2:
	print(f"Usage: {sys.argv[0]} inputFile")
	exit()
inputFile = sys.argv[1]

p = parse_params(inputFile)
print(p)

subspace_list = p.subspace_list
print(f"Computing in subspaces: {subspace_list}")

###################################################################################################
resDict = {}
for n in subspace_list:
	print(f"In the subspace with {n} particles.\n")

	if  n%2==0:
		subspaceName = "singlet"
	else:
		subspaceName = "doublet"

	mat, bas = generate_total_matrix(subspaceName, n, p)
	print(f"\nGenerated matrix, size: {len(bas)}\n")

	if p.reorder_matrix_dM:
		mat, bas = reorder_matrix_dM(mat, bas)

	if p.phase_fourier_transform:
		mat, basis_transformation_matrix, phi_list = fourier_transform_matrix(mat, bas, p)
		if 1:
			mat, basis_transformation_matrix = reorder_matrix_phi(mat, basis_transformation_matrix, phi_list)

	else:
		basis_transformation_matrix = np.identity(len(bas))	

	if p.save_matrix:
		np.savetxt( f"matrix_n{n}", mat)
		print("Saved matrix.")

	if p.verbose:
		HH = np.transpose(np.conjugate(mat))
		print(f"Is H hermitian? {np.allclose(mat, HH)}")
		print("Diagonalizing ...\n")

	def print_ith_vec(i):
		print(f"In current basis, the {i}-th vector is:")
		one = np.zeros(len(bas))
		one[i] += 1
		res = np.matmul(basis_transformation_matrix, one)
		for j, r in enumerate(res):
			if r != 0:
				print(bas[j], abs(r), r)
		print("\n")

	#print_ith_vec(0)	
	#print_ith_vec(8)
	#print_ith_vec(1)
	#print_ith_vec(9)

	#print(f"ALSO: {mat[0][8]}")		
	#print(f"ALSO: {mat[1][9]}")		

	start = time.time()
	val, vec = eigh(mat)
	vec = vec.T #eigh() returns eigenvectors as columns of a matrix, but we want vec[i] to be i-th eigenvector.
	end = time.time()
	
	if p.verbose:
		print(f"Finished, t = {round(end-start, 2)} s")

	#save the results into a dictionary of dictionaries	
	resDict[n] = { "basis" : bas, "energies" : val + p.U/2, "eigenstates" : vec}

	import cmath
	def print_eigenstate(j):
		Pvec = np.matmul(P, vec[j])

		print(f"vec {j}")

		E = 0
		for i in range(len(bas)):
			E += abs(Pvec[i])**2 * bas[i].energy()

			if  abs(Pvec[i]) > 0.01:
				pass
				#print("AAA", bas[i], bas[i].energy())

		print(f"E = {E}")
		for i in range(len(bas)):
			if abs(Pvec[i]) > 0.01:
				print(bas[i], abs(Pvec[i]), "\t", round(cmath.phase(Pvec[i])/np.pi, 3)) 
		print()		

	P = basis_transformation_matrix
	invP = np.transpose(np.conjugate(basis_transformation_matrix))
	iid = np.matmul(P, invP)	
	print("CHECK: ", np.allclose(iid, np.identity(len(P)), atol=1e-14) )
	print("EIGENSTATES:")

	print_eigenstate(0)
	print_eigenstate(1)
	print_eigenstate(2)
	print_eigenstate(3)
	#print_eigenstate(4)
	#print_eigenstate(5)




d = process_and_print_results(resDict, p)
	




#TO DO
#save_results(resDict)

