#!/usr/bin/env python3

from helper import *
from parse_matrices import parse_hopping_matrix
import numpy as np
import cmath
import os
###################################################################################################
#GENERAL BASIS GENERATION

def doublet_basis_states(mL, mR, p):
	"""
	DO NOT CHANGE THE ORDER IN THIS LIST! IT HAS TO BE THE SAME AS IN THE MATHEMATICA NOTEBOOK, BECAUSE MATRIX ELEMENTS ARE PARSED FROM THERE!

	Returns a set of doublet basis states with given mL and mR.
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

def generate_full_basis(subspace, n, p):
	"""
	Generates the full basis for a given problem, as well as the indeces which relate the states to their indeces in the general basis.
	This is neccessary to relate the basis state with the parsed hopping matrix elements.
	"""
	basis, indexList = [], []

	if subspace == "singlet":
		general_basis = singlet_basis_states
	if subspace == "doublet":
		general_basis = doublet_basis_states

	for mL in range(p.LL+1):
		for mR in range(p.LL+1):
			for i, state in enumerate(general_basis(mL, mR, p)):
				if check_state(state, n, p):
					basis.append(state)
					indexList.append(i)

	return basis, indexList

def generate_hopping_matrix(subspace, n, p):
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
	full_basis, indexList = generate_full_basis(subspace, n, p)

	H = np.zeros((len(full_basis), len(full_basis)), dtype=np.cdouble)
	for i, si in enumerate(full_basis):
		for j, sj in enumerate(full_basis):
			i_ind = indexList[i]
			j_ind = indexList[j]

			val = general_hopping_matrix[i_ind][j_ind](mL=si.mL, mR=si.mR, nL=sj.mL, nR=sj.mR, vL=p.v_L, vR=p.v_R, l=p.LL)

			if np.isnan(val): # we get nan in expressions like mL/sqrt(mL), when mL=0. But this is actually 0.
				val = 0.0
			H[i, j] += val

			if p.add_periodic_hopping_blocks:
				p_val = 0
				Mmaxi = si.mL + si.mR
				Mmaxj = sj.mL + sj.mR

				if si.dM == Mmaxi and sj.dM == -Mmaxj:
					p_val = general_hopping_matrix[i_ind][j_ind](mL=Mmaxi-1, mR=1, nL=Mmaxj, nR=0, vL=p.v_L, vR=p.v_R, l=p.LL) * cmath.exp(1j * 0 * np.pi)
					
					H[i, j] += p_val
					H[j, i] += np.conj(p_val)

				elif si.dM == -Mmaxi and sj.dM == Mmaxj:
					p_val = general_hopping_matrix[i_ind][j_ind](mL=Mmaxi-1, mR=1, nL=Mmaxj, nR=0, vL=p.v_L, vR=p.v_R, l=p.LL) * cmath.exp(1j * 0 * np.pi)

					H[i, j] += p_val
					H[j, i] += np.conj(p_val)
	
				"""
				#this is the hooping in the other direction but does not work for some reason. But we know that it should be hermitian.
				elif si.dM == -Mmaxi and sj.dM == Mmaxj:
					p_val = general_hopping_matrix[i_ind][j_ind](mL=1, mR=Mmaxi-1, nL=0, nR=Mmaxj, vL=p.v_L, vR=p.v_R, l=p.LL)
				"""
	return H, full_basis

def add_diagonal_elements(H, basis):

	for i, state in enumerate(basis):
		H[i, i] += state.energy()
	return H	

def generate_total_matrix(subspace, n, p):
	"""
	Generates the full Hamiltonian as a sum of hopping and diagonal terms.
	"""
	H, basis = generate_hopping_matrix(subspace, n, p)
	H = add_diagonal_elements(H, basis)
	return H, basis

def reorder_matrix_dM(mat, bas):
	"""
	Orders the basis states and the matrix in such a way that the states with equal dm = mL - mR are together.
	This makes the block structure of the hamiltonian more apparent.
	The first ordering parameter is dm, the second is the number of unpaired particles, calculated as n - mL - mR
	"""

	print("Sorting matrix by dM.")

	sortingRule = [i for i in range(len(bas))]
	zippedObj = sorted( list(zip(sortingRule, bas)), key = lambda x : (x[1].dM, x[1].n - x[1].mL - x[1].mR ) )
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

def fourier_transform_basis(basis, p):
	"""
	Fourier transform the deltaM quantum number into phi, by
	|phi, QP> = sum_dM e^i dM phi |dM, QP>
	where QP denotes the state of all quasiparticles.
	"""

	#the COLUMNS of this matrix are the new vectors written in the old basis.
	basis_transformation_matrix = np.zeros( (len(basis), len(basis)), dtype=np.cdouble)

	#find all present QP configurations
	QPstates = [ state.QP_state for state in basis]
	QPstates = my_unique(QPstates)

	all_phi_list = [] # contains phi for each consecutive state. Used to reorder the matrix into blocks with equal phi. 
	total_count = 0
	for QP in QPstates:
		#find all dMs for this QP configuration!
		deltaMs = [ state.dM for state in basis if state.QP_state == QP]
		deltaMs = np.unique(deltaMs)
		m_MAX = max(deltaMs)

		phis = [ 2 * np.pi * i / (m_MAX +1) for i in range(len(deltaMs))]
		# now for each phi construct the vector |phi, QP>, by adding contributions e^(i phi dM) to the position of the eigenvector in the basis
		for phi in phis:
			for i, state in enumerate(basis):
				if state.QP_state == QP:					
					basis_transformation_matrix[i, total_count] += (1/np.sqrt(m_MAX + 1)) * cmath.exp( 1j * phi * 0.5 * (state.dM + m_MAX) )
					
			all_phi_list.append(phi)
			total_count += 1 #This counts which transformed vector |phi, QP> we are creating.
	if p.verbose:
		P = basis_transformation_matrix
		invP = np.transpose(np.conjugate(basis_transformation_matrix))
		prod = np.matmul(P, invP)
		det = abs(np.linalg.det(P))
	
		print("FT determinant (has to be 1) = ", det)		
		print("Is it unitary? ", np.allclose(np.identity(len(basis_transformation_matrix)), prod ))

	return basis_transformation_matrix, all_phi_list

def fourier_transform_matrix(matrix, basis, p):
	"""
	Fourier transforms the Hamiltonian into blocks with well defined |phi>.
	"""
	P, phi_list = fourier_transform_basis(basis, p)
	#invP = np.linalg.inv(P)
	invP = np.transpose(np.conjugate(P))
	mat = np.matmul(np.matmul(invP, matrix), P)
	
	return mat, P, phi_list

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
	else:
		return check_conditions( [[bstate.n() == n, bstate.L.occupiedLevels() <= p.LL, bstate.R.occupiedLevels() <= p.LL, bstate.L.M >= 0, bstate.R.M >= 0] for bstate in state.basis_states] )

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




