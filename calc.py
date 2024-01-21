#!/usr/bin/env python3

from helper import * #fix this at some point pls (Luka, Jan 2023)
from observables_calculation import *
from matrix import generate_total_matrix, reorder_matrix_dM, fourier_transform_matrix
import sys
from math import pi
import time
from numpy.linalg import eigh
import h5py
from joblib import Parallel, delayed

###################################################################################################

def main():
	if len(sys.argv) != 2:
		print(f"Usage: {sys.argv[0]} inputFile")
		exit()
	inputFile = sys.argv[1]
	h5file = h5py.File("solution.h5", "w")

	p = parse_params(inputFile)
	print(p)

	print(f"Computing in subspaces: {p.subspace_list}")

	#compute for each subspace
	num_processes = len(p.subspace_list) if p.parallel else 1
	results = Parallel(n_jobs = num_processes)(delayed(diagonalize_subspace)(n, p) for n in p.subspace_list)

	#now merge all dictionaries into a big one
	results_dict = {}
	for r in results:
		results_dict = {**results_dict, **r}

	process_save_and_print_results(results_dict, h5file, p)

###################################################################################################

def diagonalize_subspace(n, p):
	"""
	Solves the problem in one subspace.
	Returns a dictionary with energies, eigenvalues and the basis.
	"""
	start = time.time()

	print(f"In the subspace with {n} particles.\n")
	results_dict = {}

	if  n%2==0:
		subspaceName = "singlet"
		Sz = 0 #this Sz only goes to the path in the hdf5 file as (n, Sz)!
	else:
		if p.doublet_both_Sz:
			subspaceName = "doublet_both_Sz"
			Sz = "all" #if both Sz=+1/2 and Sz=-1/2 states are calculated in the same matrix
		else:
			subspaceName = "doublet"
			Sz = 1/2
	sector = (n, Sz)

	mat, bas = generate_total_matrix(subspaceName, n, p)
	print(f"\nGenerated matrix, size: {len(bas)}\n")

	if p.reorder_matrix_dM:
		mat, bas = reorder_matrix_dM(mat, bas)
		print("Reordered into dM blocks.")

	if p.phase_fourier_transform:
		mat, basis_transformation_matrix, phi_basis = fourier_transform_matrix(mat, bas, p)
		print("Fourier transformed.")
	else:
		basis_transformation_matrix = np.identity(len(bas))

	if p.save_matrix:
		np.savetxt( f"matrix_n{n}", mat)
#		mat.real.astype('float64').tofile( f"matrix_bin_re_n{n}" )
#		mat.imag.astype('float64').tofile( f"matrix_bin_im_n{n}" )
#		mat.astype('complex128').tofile( f"matrix_bin_n{n}" )
		print("Saved matrix.")

	if p.verbose:
		HH = np.transpose(np.conjugate(mat))
		print(f"Is H hermitian? {np.allclose(mat, HH)}")

		invP = np.transpose(np.conjugate(basis_transformation_matrix))
		prod = np.matmul(invP, basis_transformation_matrix)
		print(f"Is the transform unitary? {np.allclose(prod, np.identity(len(basis_transformation_matrix)))}")

	###############################################################################################

	print("Diagonalizing ...\n")
	val, vec = eigh(mat)
	vec = vec.T #eigh() returns eigenvectors as columns of a matrix, but we want vec[i] to be i-th eigenvector.

	#cut off the number of states to save
	if not p.save_all_states:
		vec = vec[:p.num_states_to_save]
		val = val[:p.num_states_to_save]

	###############################################################################################
	#save the results into a dictionary of dictionaries
	results_dict[sector] = { "energies" : val + p.U/2, "dM_basis" : bas }

	if p.phase_fourier_transform:
		#The results are in the FT basis, to get the dM basis back transform all vectors back.

		#transform eigenvectors back into the original basis
		dMvec = [ np.matmul(basis_transformation_matrix, v) for v in vec]

		results_dict[sector]["dM_eigenstates"] = dMvec
		results_dict[sector]["phi_eigenstates"] = vec
		results_dict[sector]["phi_basis"] = phi_basis

	else:
		#if FT was not performed, just save the eigenvalues as they are
		results_dict[sector]["dM_eigenstates"] = vec

	###############################################################################################

	end = time.time()
	print(f"Sector {n} finished, t = {round(end-start, 2)} s")

	return results_dict

###################################################################################################

if __name__ == "__main__":
	main()
