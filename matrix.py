#!/usr/bin/env python3

from helper import STATE, BASIS_STATE, PHI_STATE, UP, DOWN, ZERO, UPDN, delta
from parse_matrices import parse_hopping_matrix
import numpy as np
import cmath
import os

import sys

path_to_my_second_quantization = os.getenv("MY_SECOND_QUANTIZATION_PATH")
if not path_to_my_second_quantization:
	raise Exception("System variable 'MY_SECOND_QUANTIZATION_PATH' not defined! Set the path, probably with:\nexport MY_SECOND_QUANTIZATION_PATH=/Volumes/f1login.ijs.si/git_repos/my_second_quantization\nOR\nexport MY_SECOND_QUANTIZATION_PATH=/home/pavesic/git_repos/my_second_quantization")

sys.path.insert(1, path_to_my_second_quantization)
import operators as op
from operators import BASIS_STATE as COMP_B_STATE
from operators import STATE as COMP_STATE
from bitwise_ops import generate_Sz_basis
###################################################################################################
#GENERAL BASIS GENERATION

def doublet_basis_states(mL, mR, p):
	"""
	DO NOT CHANGE THE ORDER IN THIS LIST! IT HAS TO BE THE SAME AS IN THE MATHEMATICA NOTEBOOK, BECAUSE MATRIX ELEMENTS ARE PARSED FROM THERE!

	Returns a set of doublet basis states with given mL and mR, Sz=+1/2.
	"""

	# 1 qp
	psi_100 = STATE(0, (1.0, BASIS_STATE(UP, mL, ZERO, mR, ZERO, p)) )
	psi_010 = STATE(1, (1.0, BASIS_STATE(ZERO, mL, UP, mR, ZERO, p)) )
	psi_001 = STATE(2, (1.0, BASIS_STATE(ZERO, mL, ZERO, mR, UP, p)) )

	# 3 qp
	psi_210 = STATE(3, (1.0, BASIS_STATE(UPDN, mL, UP, mR, ZERO, p)) )
	psi_201 = STATE(4, (1.0, BASIS_STATE(UPDN, mL, ZERO, mR, UP, p)) )
	psi_120 = STATE(5, (1.0, BASIS_STATE(UP, mL, UPDN, mR, ZERO, p)) )
	psi_021 = STATE(6, (1.0, BASIS_STATE(ZERO, mL, UPDN, mR, UP, p)) )
	psi_012 = STATE(7, (1.0, BASIS_STATE(ZERO, mL, UP, mR, UPDN, p)) )
	psi_102 = STATE(8, (1.0, BASIS_STATE(UP, mL, ZERO, mR, UPDN, p)) )
	
	psi_S12 = STATE(9, 	(1/np.sqrt(2),  BASIS_STATE(DOWN, mL, UP, mR, UP, p)), 
						(-1/np.sqrt(2), BASIS_STATE(UP, mL, DOWN, mR, UP, p)) )
	psi_3qp = STATE(10,	(1/np.sqrt(6),  BASIS_STATE(DOWN, mL, UP, mR, UP, p) ),
						(1/np.sqrt(6),  BASIS_STATE(UP, mL, DOWN, mR, UP, p) ),
						(-2/np.sqrt(6), BASIS_STATE(UP, mL, UP, mR, DOWN, p) ),
						)
	# 5 qp
	psi_221 = STATE(11, (1.0, BASIS_STATE(UPDN, mL, UPDN, mR, UP, p)) )
	psi_212 = STATE(12, (1.0, BASIS_STATE(UPDN, mL, UP, mR, UPDN, p)) )
	psi_122 = STATE(13, (1.0, BASIS_STATE(UP, mL, UPDN, mR, UPDN, p)) )

	allStates = [psi_100, psi_010, psi_001, psi_210, psi_201, psi_120, psi_021, psi_012, psi_102, psi_S12, psi_3qp, psi_221, psi_212, psi_122]
	return allStates

def doublet_basis_states_both_Sz(mL, mR, p):
	"""
	DO NOT CHANGE THE ORDER IN THIS LIST! IT HAS TO BE THE SAME AS IN THE MATHEMATICA NOTEBOOK, BECAUSE MATRIX ELEMENTS ARE PARSED FROM THERE!

	Returns a set of doublet basis states with given mL and mR, Sz=+1/2 and Sz=-1/2.
	"""

	# Sz = +1/2
	# 1 qp
	psi_100 = STATE(0, (1.0, BASIS_STATE(UP, mL, ZERO, mR, ZERO, p)) )
	psi_010 = STATE(1, (1.0, BASIS_STATE(ZERO, mL, UP, mR, ZERO, p)) )
	psi_001 = STATE(2, (1.0, BASIS_STATE(ZERO, mL, ZERO, mR, UP, p)) )

	# 3 qp
	psi_210 = STATE(3, (1.0, BASIS_STATE(UPDN, mL, UP, mR, ZERO, p)) )
	psi_201 = STATE(4, (1.0, BASIS_STATE(UPDN, mL, ZERO, mR, UP, p)) )
	psi_120 = STATE(5, (1.0, BASIS_STATE(UP, mL, UPDN, mR, ZERO, p)) )
	psi_021 = STATE(6, (1.0, BASIS_STATE(ZERO, mL, UPDN, mR, UP, p)) )
	psi_012 = STATE(7, (1.0, BASIS_STATE(ZERO, mL, UP, mR, UPDN, p)) )
	psi_102 = STATE(8, (1.0, BASIS_STATE(UP, mL, ZERO, mR, UPDN, p)) )
	
	psi_S12 = STATE(9, 	(1/np.sqrt(2),  BASIS_STATE(DOWN, mL, UP, mR, UP, p)), 
						(-1/np.sqrt(2), BASIS_STATE(UP, mL, DOWN, mR, UP, p)) )
	psi_3qp = STATE(10,	(1/np.sqrt(6),  BASIS_STATE(DOWN, mL, UP, mR, UP, p) ),
						(1/np.sqrt(6),  BASIS_STATE(UP, mL, DOWN, mR, UP, p) ),
						(-2/np.sqrt(6), BASIS_STATE(UP, mL, UP, mR, DOWN, p) ),
						)
	# 5 qp
	psi_221 = STATE(11, (1.0, BASIS_STATE(UPDN, mL, UPDN, mR, UP, p)) )
	psi_212 = STATE(12, (1.0, BASIS_STATE(UPDN, mL, UP, mR, UPDN, p)) )
	psi_122 = STATE(13, (1.0, BASIS_STATE(UP, mL, UPDN, mR, UPDN, p)) )

	# Sz = -1/2
	# 1 qp
	psi_100m = STATE(0, (1.0, BASIS_STATE(DOWN, mL, ZERO, mR, ZERO, p)) )
	psi_010m = STATE(1, (1.0, BASIS_STATE(ZERO, mL, DOWN, mR, ZERO, p)) )
	psi_001m = STATE(2, (1.0, BASIS_STATE(ZERO, mL, ZERO, mR, DOWN, p)) )

	# 3 qp
	psi_210m = STATE(3, (1.0, BASIS_STATE(UPDN, mL, DOWN, mR, ZERO, p)) )
	psi_201m = STATE(4, (1.0, BASIS_STATE(UPDN, mL, ZERO, mR, DOWN, p)) )
	psi_120m = STATE(5, (1.0, BASIS_STATE(DOWN, mL, UPDN, mR, ZERO, p)) )
	psi_021m = STATE(6, (1.0, BASIS_STATE(ZERO, mL, UPDN, mR, DOWN, p)) )
	psi_012m = STATE(7, (1.0, BASIS_STATE(ZERO, mL, DOWN, mR, UPDN, p)) )
	psi_102m = STATE(8, (1.0, BASIS_STATE(DOWN, mL, ZERO, mR, UPDN, p)) )
	
	psi_S12m = STATE(9, 	(1/np.sqrt(2),  BASIS_STATE(DOWN, mL, UP, mR, DOWN, p)), 
						(-1/np.sqrt(2), BASIS_STATE(UP, mL, DOWN, mR, DOWN, p)) )
	psi_3qpm = STATE(10,	(1/np.sqrt(6),  BASIS_STATE(UP, mL, DOWN, mR, DOWN, p) ),
						(1/np.sqrt(6),  BASIS_STATE(DOWN, mL, UP, mR, UP, p) ),
						(-2/np.sqrt(6), BASIS_STATE(DOWN, mL, DOWN, mR, UP, p) ),
						)
	# 5 qp
	psi_221m = STATE(11, (1.0, BASIS_STATE(UPDN, mL, UPDN, mR, DOWN, p)) )
	psi_212m = STATE(12, (1.0, BASIS_STATE(UPDN, mL, DOWN, mR, UPDN, p)) )
	psi_122m = STATE(13, (1.0, BASIS_STATE(DOWN, mL, UPDN, mR, UPDN, p)) )

	allStates = [psi_100, psi_010, psi_001, psi_210, psi_201, psi_120, psi_021, psi_012, psi_102, psi_S12, psi_3qp, psi_221, psi_212, psi_122,
				 psi_100m, psi_010m, psi_001m, psi_210m, psi_201m, psi_120m, psi_021m, psi_012m, psi_102m, psi_S12m, psi_3qpm, psi_221m, psi_212m, psi_122m]
	return allStates

def singlet_basis_states(mL, mR, p):
	"""
	DO NOT CHANGE THE ORDER IN THIS LIST! IT HAS TO BE THE SAME AS IN THE MATHEMATICA NOTEBOOK, BECAUSE MATRIX ELEMENTS ARE PARSED FROM THERE!

	Returns a set of singlet basis states with given mL and mR.
	"""

	# 0 qp
	phi_0 = STATE(0, (1.0, BASIS_STATE(ZERO, mL, ZERO, mR, ZERO, p)) )

	# 2 qp
	phi_002 = STATE(1, (1.0, BASIS_STATE(ZERO, mL, ZERO, mR, UPDN, p)) )
	phi_020 = STATE(2, (1.0, BASIS_STATE(ZERO, mL, UPDN, mR, ZERO, p)) )
	phi_200 = STATE(3, (1.0, BASIS_STATE(UPDN, mL, ZERO, mR, ZERO, p)) )

	phi_s12 = STATE(4, 	(1/np.sqrt(2),  BASIS_STATE(UP, mL, DOWN, mR, ZERO, p)),
						(-1/np.sqrt(2), BASIS_STATE(DOWN, mL, UP, mR, ZERO, p))
						)
	phi_s13 = STATE(5, 	(1/np.sqrt(2),  BASIS_STATE(UP, mL, ZERO, mR, DOWN, p)),
						(-1/np.sqrt(2), BASIS_STATE(DOWN, mL, ZERO, mR, UP, p))
						)
	phi_s23 = STATE(6,	(1/np.sqrt(2),  BASIS_STATE(ZERO, mL, UP, mR, DOWN, p)),
						(-1/np.sqrt(2), BASIS_STATE(ZERO, mL, DOWN, mR, UP, p))
						)
	# 4 qp
	phi_022 = STATE(7, (1.0, BASIS_STATE(ZERO, mL, UPDN, mR, UPDN, p)) )
	phi_202 = STATE(8, (1.0, BASIS_STATE(UPDN, mL, ZERO, mR, UPDN, p)) )
	phi_220 = STATE(9, (1.0, BASIS_STATE(UPDN, mL, UPDN, mR, ZERO, p)) )

	phi_s4_12 = STATE(10, 	(1/np.sqrt(2),  BASIS_STATE(DOWN, mL, UP, mR, UPDN, p)),
							(-1/np.sqrt(2), BASIS_STATE(UP, mL, DOWN, mR, UPDN, p))
							)
	phi_s4_13 = STATE(11, 	(1/np.sqrt(2),  BASIS_STATE(DOWN, mL, UPDN, mR, UP, p)),
							(-1/np.sqrt(2), BASIS_STATE(UP, mL, UPDN, mR, DOWN, p))
							)
	phi_s4_23 = STATE(12, 	(1/np.sqrt(2),  BASIS_STATE(UPDN, mL, UP, mR, DOWN, p)),
							(-1/np.sqrt(2), BASIS_STATE(UPDN, mL, DOWN, mR, UP, p))
							)

	# 6 qp
	phi_6 = STATE(13, (1.0, BASIS_STATE(UPDN, mL, UPDN, mR, UPDN, p)) )

	allStates = [phi_0, phi_002, phi_020, phi_200, phi_s12, phi_s13, phi_s23, phi_022, phi_202, phi_220, phi_s4_12, phi_s4_13, phi_s4_23, phi_6]
	return allStates

###################################################################################################
#FUNCTIONS FOR THE GENERATION OF THE MATRIX
bias = "left"

def get_excluded_dMs(bias: str, n: int):
	"""
	Generate a list of dM values to be excluded. 
	An odd number of values has to be excluded. For odd n, it has to be 3, for even; 5.
	For a span of -m, -m+1, ..., m, either exclude the first one and two last or the opposite.
	bias can be "left" or "right". "left" means exclude one value on the left and two on the right, 
	while "right" means the opposite. 
	"""
	if n%2 == 1:
		val = (n-1)/2
		if bias == "left":
			return [-val, val, val-1]
		elif bias == "right":
			return [-val, -val+1, val]
	elif n%2 == 0:
		val = n/2
		if bias == "left":
			return [-val, -val+1, val, val-1, val-2]
		elif bias == "right":
			return [-val, -val+1, -val+2, val, val-1]

def get_max_min_dMs(bias: str, n: int):
	if n%2 == 1:
		val = (n-1)/2
		if bias == "left":
			smallest_dM, largest_dM = -val + 1, val - 2
		elif bias == "right":
			smallest_dM, largest_dM = -val + 2, val - 1
	elif n%2 == 0:
		val = n/2
		if bias == "left":
			smallest_dM, largest_dM = -val + 2, val - 3
		elif bias == "right":
			smallest_dM, largest_dM = -val + 3, val - 2

	return smallest_dM, largest_dM				

def generate_full_basis(subspace, n, p):
	"""
	Generates the full basis for a given problem, as well as the indeces which relate the states to their indeces in the general basis.
	This is neccessary to relate the basis state with the parsed hopping matrix elements.
	"""
	basis, indexList = [], []

	# Set the name of the function used to generate the basis states for each mL, mR combination.
	if subspace == "singlet":
		general_basis = singlet_basis_states
	if subspace == "doublet":
		general_basis = doublet_basis_states
	if subspace == "doublet_both_Sz":
		general_basis = doublet_basis_states_both_Sz
			
	for mL in range(p.LL+1):
		for mR in range(p.LL+1):
			if p.restrict_basis_to_make_periodic:
				# Throw away all states that would form uncomplete dM blocks. To make the matrix perfectly periodic with additional periodic hopping blocks, 
				# we have to throw away three blocks! (There is always an odd number of available dM blocks, eq: -3, -2, -1, 0, 1, 2, 3.)
				# Here calculate the largest possible dMmax and throw away states with dM = -dMmax, dM = +dMmax and dM = +dMmax - 1.
				restrict_dMs = get_excluded_dMs(bias, n)
				if mL - mR in restrict_dMs:
					continue #skips this iteration of the loop
			for i, state in enumerate(general_basis(mL, mR, p)):
				if check_state(state, n, p):
					basis.append(state)
					indexList.append(i)

	return basis, indexList

def generate_hopping_matrix(subspace, full_basis, index_list, n, p):
	"""
	Generates the full hopping matrix. 
	For each (mL, mR) takes a general basis, and finds the matrix elements for all states for all (nL, nR).
	"""
	if p.turn_off_hopping_finite_size_effects:
		matName = subspace + "_no_finite_size_effects.dat"
	else:
		matName = subspace + "_mat.dat"	

	file_path =  os.path.abspath(os.path.dirname(__file__))	#this is the absolute path of the script! (eg. /home/pavesic/git_repos/flat_band_two_channel on spinon) 

	general_hopping_matrix, _ = parse_hopping_matrix(file_path + "/matrices/" + matName)

	H = np.zeros((len(full_basis), len(full_basis)), dtype=np.cdouble)
	for i, si in enumerate(full_basis):
		for j, sj in enumerate(full_basis):
			i_ind = index_list[i]
			j_ind = index_list[j]

			val = general_hopping_matrix[i_ind][j_ind](mL=si.mL, mR=si.mR, nL=sj.mL, nR=sj.mR, vL=p.v_L, vR=p.v_R, tsc=p.tsc, l=p.LL)

			if np.isnan(val): # we get nan in expressions like mL/sqrt(mL), when mL=0. But this is actually 0.
				val = 0.0
			H[i, j] += val

			if p.add_periodic_hopping_blocks:
				# This has to be used with p.restrict_basis_to_make_periodic. 
				# If dMmax = n//2, the states will have dM = -dMmax+1, +2, +3, ..., +dMmax-2. 
				# You have to add hopping between the extreme ones, ie. -dMmax+1 and +dMmax-2
				smallest_dM, largest_dM = get_max_min_dMs(bias, n)
				val = 0	
				if si.dM == smallest_dM and sj.dM == largest_dM:
					# make the values like they are for dM -> dM+1 artificially! 
					val += general_hopping_matrix[i_ind][j_ind](mL=si.mL, mR=si.mR, nL=si.mL-1, nR=si.mR, vL=p.v_L, vR=p.v_R, tsc=p.tsc, l=p.LL)
					val += general_hopping_matrix[i_ind][j_ind](mL=si.mL, mR=si.mR, nL=si.mL, nR=si.mR+1, vL=p.v_L, vR=p.v_R, tsc=p.tsc, l=p.LL)
				elif si.dM == largest_dM and sj.dM == smallest_dM:
					# make the values like they are for dM -> dM-1 artificially! 
					val += general_hopping_matrix[i_ind][j_ind](mL=si.mL, mR=si.mR, nL=si.mL+1, nR=si.mR, vL=p.v_L, vR=p.v_R, tsc=p.tsc, l=p.LL)
					val += general_hopping_matrix[i_ind][j_ind](mL=si.mL, mR=si.mR, nL=si.mL, nR=si.mR-1, vL=p.v_L, vR=p.v_R, tsc=p.tsc, l=p.LL)
				H[i, j] += val
	return H

#def pair_hopping_element(tpair: float, phiext: float, mL: int, mR: int, nL: int, nR: int) -> float :
def pair_hopping_element(tpair: float, phiext: float, si: STATE, sj: STATE) -> float :
	mL, mR = si.mL, si.mR
	nL, nR = sj.mL, sj.mR
	QPi, QPj = si.QP_state, sj.QP_state
	return -1 * tpair * delta(QPi, QPj) * ( np.exp(1j * phiext) * delta(mL, nL+1) * delta(mR, nR-1) + np.exp(- 1j * phiext) * delta(mL, nL-1) * delta(mR, nR+1) )

def add_sc_pair_hopping(H, basis, n, p):
	"""
	This is pair hopping between the two SCs and is just pair exhange between the two reservoirs.
	It should not affect quasiparticles or the QD configuration. 
	Its matrix element between states |qp, mL, mR> and |qp, nL, nR> is proportional to: 
		delta(mL, nL+1) delta(mR, nR-1) + delta(mL, nL-1) delta(mR, nR+1) 
	"""
	for i, si in enumerate(basis):
		for j, sj in enumerate(basis):
			mL, mR = si.mL, si.mR
			nL, nR = sj.mL, sj.mR
			#this has to have a minus in order for the ground state to have phi=0!
			H[i, j] += pair_hopping_element(p.tpair, p.phiext, si, sj)
			if p.add_periodic_hopping_blocks:
				#HERE DO LIKE ABOVE FOR REAL HOPPING!!
				smallest_dM, largest_dM = get_max_min_dMs(bias, n)
				val = 0	
				if si.dM == smallest_dM and sj.dM == largest_dM:
					# make the values like they are for dM -> dM+1 artificially! 
					val += pair_hopping_element(p.tpair, p.phiext, si, sj)
					val += pair_hopping_element(p.tpair, p.phiext, si, sj)
				elif si.dM == largest_dM and sj.dM == smallest_dM:
					# make the values like they are for dM -> dM-1 artificially! 
					val += pair_hopping_element(p.tpair, p.phiext, si, sj)
					val += pair_hopping_element(p.tpair, p.phiext, si, sj)
	return H

def add_diagonal_elements(H, basis):
	for i, state in enumerate(basis):
		H[i, i] += state.energy()
	return H	

def generate_total_matrix(subspace, n, p):
	"""
	Generates the full Hamiltonian as a sum of hopping and diagonal terms.
	"""
	full_basis, index_list = generate_full_basis(subspace, n, p)

	H = generate_hopping_matrix(subspace, full_basis, index_list, n, p)
	H = add_sc_pair_hopping(H, full_basis, n, p)
	H = add_diagonal_elements(H, full_basis)
	return H, full_basis

def reorder_matrix_dM(mat, bas):
	"""
	Orders the basis states and the matrix in such a way that the states with equal dm = mL - mR are together.
	This makes the block structure of the hamiltonian more apparent.
	Ordering parameters:
	dm, number of unpaired particles (n-mL-mR),
	"""

	print("Sorting matrix by dM.")

	sortingRule = [i for i in range(len(bas))]
	zippedObj = sorted( list(zip(sortingRule, bas)), key = lambda x : (x[1].dM, x[1].n - x[1].mL - x[1].mR,  ) )
	sortingRule, sortedBas = unzip(zippedObj)

	mat = reorder_matrix(mat, sortingRule)
	return mat, sortedBas

def reorder_matrix_phi(mat, basis_transformation_matrix, phi_list):
	"""
	Orders the matrix into blocks with equal phi.
	phi_list is a list, i-th element is phi of the i-th vector.
	"""

	print("Sorting matrix by phi.")
	raise Exception("REORDERING BY PHI DOES NOT WORK! THE BASIS IS WRONG!")

	sortingRule = [i for i in range(len(mat))]
	zippedObj = sorted( list(zip(sortingRule, phi_list)), key = lambda x : x[1] )
	sortingRule, sortedPhis = unzip(zippedObj)

	mat = reorder_matrix(mat, sortingRule)
	#basis_transformation_matrix = reorder_matrix(basis_transformation_matrix, sortingRule)

	basis_transformation_matrix = basis_transformation_matrix[:, sortingRule]
	#basis_transformation_matrix = basis_transformation_matrix[sortingRule, :]

	return mat, basis_transformation_matrix

def reorder_matrix(mat, rule):
	"""
	Orders the matrix and basis by the given rule.
	rule is a list of integers, rule[i] is where to move the i-th element to. 
	"""
	mat = mat[rule, :] [:, rule]	#numpy magic - reorders both columns and rows by the rule
	return mat

def unzip(zipped_object):	
	unzipped = list(zip(*zipped_object))
	return unzipped[0], unzipped[1]

###################################################################################################
#COMPUTATION BASIS - basis with unique basis states with (n, Sz). Used for calculations of properties.

def odd_computation_basis_OLD(mL, mR, p):
	"""
	A list of all unique BASIS_STATE with Sz=1/2.
	"""
	basis = []

	#1 qp
	basis.append( BASIS_STATE(UP, mL, ZERO, mR, ZERO, p) )
	basis.append( BASIS_STATE(ZERO, mL, UP, mR, ZERO, p) )
	basis.append( BASIS_STATE(ZERO, mL, ZERO, mR, UP, p) )
	
	#3 qp
	basis.append( BASIS_STATE(UP, mL, UP, mR, DOWN, p) )
	basis.append( BASIS_STATE(UP, mL, DOWN, mR, UP, p) )
	basis.append( BASIS_STATE(DOWN, mL, UP, mR, UP, p) )

	basis.append( BASIS_STATE(UPDN, mL, UP, mR, ZERO, p) )
	basis.append( BASIS_STATE(UPDN, mL, ZERO, mR, UP, p) )
	basis.append( BASIS_STATE(ZERO, mL, UPDN, mR, UP, p) )
	basis.append( BASIS_STATE(UP, mL, UPDN, mR, ZERO, p) )
	basis.append( BASIS_STATE(UP, mL, ZERO, mR, UPDN, p) )
	basis.append( BASIS_STATE(ZERO, mL, UP, mR, UPDN, p) )

	# 5 qp
	basis.append( BASIS_STATE(UPDN, mL, UPDN, mR, UP, p) )
	basis.append( BASIS_STATE(UPDN, mL, UP, mR, UPDN, p) )
	basis.append( BASIS_STATE(UP, mL, UPDN, mR, UPDN, p) )

	return basis

def even_computation_basis_OLD(mL, mR, p):
	"""
	A list of all unique BASIS_STATE with Sz=0.
	"""
	basis = []
	
	# 0 qp
	basis.append( BASIS_STATE(ZERO, mL, ZERO, mR, ZERO, p) )
	
	# 2 qp
	basis.append( BASIS_STATE(UP, mL, DOWN, mR, ZERO, p) )
	basis.append( BASIS_STATE(UP, mL, ZERO, mR, DOWN, p) )

	basis.append( BASIS_STATE(DOWN, mL, UP, mR, ZERO, p) )
	basis.append( BASIS_STATE(DOWN, mL, ZERO, mR, UP, p) )

	basis.append( BASIS_STATE(ZERO, mL, UP, mR, DOWN, p) )
	basis.append( BASIS_STATE(ZERO, mL, DOWN, mR, UP, p) )

	basis.append( BASIS_STATE(UPDN, mL, ZERO, mR, ZERO, p) )
	basis.append( BASIS_STATE(ZERO, mL, UPDN, mR, ZERO, p) )
	basis.append( BASIS_STATE(ZERO, mL, ZERO, mR, UPDN, p) )
	
	# 4 qp
	basis.append( BASIS_STATE(UPDN, mL, UPDN, mR, ZERO, p) )
	basis.append( BASIS_STATE(UPDN, mL, ZERO, mR, UPDN, p) )
	basis.append( BASIS_STATE(ZERO, mL, UPDN, mR, UPDN, p) )

	basis.append( BASIS_STATE(UPDN, mL, UP, mR, DOWN, p) )
	basis.append( BASIS_STATE(UPDN, mL, DOWN, mR, UP, p) )

	basis.append( BASIS_STATE(UP, mL, UPDN, mR, DOWN, p) )
	basis.append( BASIS_STATE(DOWN, mL, UPDN, mR, UP, p) )

	basis.append( BASIS_STATE(UP, mL, DOWN, mR, UPDN, p) )
	basis.append( BASIS_STATE(DOWN, mL, UP, mR, UPDN, p) )

	# 6 qp
	basis.append( BASIS_STATE(UPDN, mL, UPDN, mR, UPDN, p) )

	return basis

def computation_basis(Sz, mL, mR):
	"""
	Generates op.BASIS_STATES with a given Sz on three sites (imp, L, R) 
	and mL and mR as additional quantum numbers. 
	"""
	basis_states = generate_Sz_basis(Sz=Sz, N=3)
	basis = []
	for m in basis_states:
		basis.append( COMP_B_STATE( bitstring = m, mL = mL, mR = mR ) )
	basis = sorted(basis)
	basis = np.array(basis, dtype=object)	
	return basis

def generate_computation_basis(n, Sz, p):
	"""
	Generates a list of all basis states with a given n.
	"""
	#First generate a list of all Szs. 
	#This should be generalised if the code is extended to do more Szs in the future.
	if Sz == "all":
		Szlist = [-1/2, 1/2]
	else:
		Szlist = [Sz]

	basis = []
	for Sz in Szlist:
		for mL in range(0, n): #this loop is overkill but whatever
			for mR in range(0, n):
				this_basis = computation_basis(Sz, mL, mR)
				for state in this_basis:
					if state.n + 2 * (mL + mR) == n:
						basis.append(state)

	basis = np.array( sorted(basis), dtype=op.BASIS_STATE)
	return basis

def dM_basis_to_calc_basis(dM_basis_state):
	"""
	Writes a basis state in the dM basis into BASIS_STATE type from my_second_quantization, 
	where mL and mR are quantum numbers and the QP_STATE is a bitstring.
	"""
	bitstring = dM_basis_state.QP_state.bitstring()
	return COMP_B_STATE( bitstring = bitstring, mL = dM_basis_state.L.M, mR = dM_basis_state.R.M)

def write_vector_in_computation_basis(eigenstate, dM_basis, computation_basis):
	"""
	Rewrites a vector from the spin basis to the computation basis. 
	
	Iterates over all STATEs in the vector and for each BASIS_STATE adds up the amplitude to the corresponding point in the computation basis vector.
	"""
	comp_vector = np.zeros(len(computation_basis), dtype=complex)

	for i, dM_state in enumerate(dM_basis):
		state_amp = eigenstate[i]
		for amp, dM_basis_state in dM_state.amplitudes_and_basis_states:
			corresponding_calc_basis_state = dM_basis_to_calc_basis(dM_basis_state)
			ndx = my_find_ndx(corresponding_calc_basis_state, computation_basis) #finds the index of the basis state in the computation basis list
			if ndx == None:
				raise Exception(f"THIS BASIS STATE WAS NOT FOUND IN THE COMPUTATION BASIS! {dM_basis_state}")
			
			comp_vector[ndx] += amp * state_amp
	comp_state = COMP_STATE(vector = comp_vector, basis = computation_basis, N = 3)
	return comp_state

###################################################################################################
#FUNCTIONS FOR FOURIER TRANSFORM

def fourier_transform_basis(basis, p):
	"""
	Fourier transform the deltaM quantum number into phi, by
	|phi, QP> = sum_dM e^{i dM.phi} |dM, QP>
	where QP is a label for the state of all quasiparticles.
	"""

	#the COLUMNS of this matrix are the new vectors written in the old basis.
	basis_transformation_matrix = np.zeros( (len(basis), len(basis)), dtype=np.cdouble)

	#find all present QP configurations
	QPstates = [ state.QP_state for state in basis]
	QPstates = my_unique(QPstates)

	phi_basis = [] # contains phi for each consecutive state. Used to reorder the matrix into blocks with equal phi. 
	total_count = 0
	for QP in QPstates:
		#find all dMs for this QP configuration!

		deltaMs = [ state.dM for state in basis if state.QP_state == QP]
		deltaMs = np.unique(deltaMs)
		numdMs = len(deltaMs)
		m_MAX = max(deltaMs)
		m_MIN = min(deltaMs)
		
		#phis = [ 2 * np.pi * i / (m_MAX +1) for i in range(len(deltaMs))]
		phis = [ 2 * np.pi * i / numdMs for i in range(len(deltaMs))]
		# now for each phi construct the vector |phi, QP>, by adding contributions e^(i phi dM) to the position of the eigenvector in the basis
		for phi in phis:
			for i, state in enumerate(basis):
				if state.QP_state == QP:
					#print((1/np.sqrt(m_MAX + 1)) * cmath.exp( 1j * phi * 0.5 * (state.dM + m_MAX) ), m_MAX, state.dM)
					#print("A", state.dM - m_MIN)
					basis_transformation_matrix[i, total_count] += (1/np.sqrt(numdMs)) * cmath.exp( 1j * phi * 0.5 * (state.dM - m_MIN) )
				
			phi_basis.append( PHI_STATE(phi, QP) )

			total_count += 1 #This counts which transformed vector |phi, QP> we are creating.
	if p.verbose:
		P = basis_transformation_matrix
		invP = np.transpose(np.conjugate(basis_transformation_matrix))
		prod = np.matmul(P, invP)
		det = abs(np.linalg.det(P))
	
		print("FT determinant (has to be 1) = ", det)	
		print("Is it unitary? ", np.allclose(np.identity(len(basis_transformation_matrix)), prod ))

	return basis_transformation_matrix, phi_basis

def fourier_transform_matrix(matrix, basis, p):
	"""
	Fourier transforms the Hamiltonian into blocks with well defined |phi>.
	"""
	P, phi_basis = fourier_transform_basis(basis, p)
	#invP = np.linalg.inv(P)
	invP = np.transpose(np.conjugate(P))
	mat = np.matmul(np.matmul(invP, matrix), P)
	
	return mat, P, phi_basis

###################################################################################################

def check_state(state, n, p):
	"""
	For each basis state check if:
		its charge is equal to n
		number of occupied levels is not larger than the number of all levels
		number of cooper pairs is not negative
	"""
	if p.use_all_states:
		return check_conditions( state.n == n, state.mL >= 0, state.mR >= 0)
		#return check_conditions( state.n == n, state.mL >= 0, state.mR >= 0, state.nqp == 1)
	else:
		all_conds = [[bstate.n == n, bstate.L.occupiedLevels <= p.LL, bstate.R.occupiedLevels <= p.LL, bstate.L.M >= 0, bstate.R.M >= 0] for bstate in state.basis_states]
		flat_list = [item for sublist in all_conds for item in sublist] #this flattens the list
		if False in flat_list:
			return False
		else:
			return True

def check_conditions(*conditions):
	"""
	Return True if all conditions are True.
	"""
	for cond in conditions:
		if not cond:
			return False
	return True

def my_unique(ll):
	tmpL = []
	for elem in ll:
		if elem in tmpL:
			continue
		else:
			tmpL.append(elem)
	return tmpL

def my_find_ndx(element, compareList):
	"""
	Intended to work like np.where but also for the BASIS_STATE class. 
	Finds the index of the element in the compareList.
	"""
	for i, el in enumerate(compareList):
		if element == el:
			return i
	return None