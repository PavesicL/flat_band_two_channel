#!/usr/bin/env python3

import numpy as np
from helper import *

###################################################################################################
# PRINT AN EIGENSTATE


def print_energies(n_dict, p):
	print("\nEnergies: ")
	
	maxE = min(len(n_dict["energies"]), p.print_energies)

	Estr = ""
	for i in range(maxE):
		E = n_dict["energies"][i]
		Estr += f"{E}, "
	Estr = Estr[:-2]	
	print(Estr)


def print_state(eigenvector, basis, prec):

	printList = []
	for i, amplitude in enumerate(eigenvector):
		if abs(amplitude) > prec:
			printList.append([amplitude, basis[i]])
	printList = sorted(printList, key = lambda x : -abs(x[0]))
	for amp, bas in printList:
		print(f"{amp}	{bas}")		


###################################################################################################
# OCCUPANCY CALCULATION

def calculate_occupancy(eigenvector, basis):
	"""
	Calculates the occupancy in all parts of the system for a given eigenvector.
	"""
	nimp, nL, nR = 0, 0, 0
	for i, amplitude in enumerate(eigenvector):
		if amplitude != 0:
			nimp += amplitude**2 * basis[i].nimp
			nL += amplitude**2 * basis[i].nL
			nR += amplitude**2 * basis[i].nR
	return nimp, nL, nR		

def print_all_occupancies(states, basis):
	ns = ""
	listNs = []
	for state in states:
		nimp, nL, nR = calculate_occupancy(state, basis)			
		ns += f"({nimp}, {nL}, {nR}) "
		listNs.append([nimp, nL, nR])
	print(f"occupation: {ns}")	
	return listNs

###################################################################################################
# PHASE CALCULATION

def calculate_psi_N_overlap(N, eigenvector, basis, p):
	"""
	Computes <psi|N>, where |N> is a state with N Cooper pairs in the left channel.
	"""
	psiN = 0
	for i, state in enumerate(basis):
		amplitude = eigenvector[i]
		for amp_bs in state.amplitudes_and_basis_states:
			amp_b, bstate = amp_bs
			psiN += amplitude * amp_b * delta(N, bstate.L.M)
	
	return psiN		

def calculate_phase(eigenvector, basis, p):
	"""
	Calculation of phase and phase^2. 
	<phase> = 2 * sum_N( <psi|N> <N+1|psi> ) 
	<phase^2> = sum_N( <psi|N><N+2|psi> + <psi|N><N|psi> + <psi|N+1><N+1|psi> + <psi|N+1><N-1|psi> )
	"""
	phi = 0
	phi2 = 0
	for N in range(p.LL+1):
		psiN = calculate_psi_N_overlap( N%p.LL, eigenvector, basis, p)
		psiNp1 = calculate_psi_N_overlap( (N+1)%p.LL, eigenvector, basis, p)
		psiNp2 = calculate_psi_N_overlap( (N+2)%p.LL, eigenvector, basis, p)
		psiNm1 = calculate_psi_N_overlap( (N-1)%p.LL, eigenvector, basis, p)

		phi  += 2 * psiN * psiNp1
		phi2 += psiN*psiNp2 + psiN*psiN + psiNp1*psiNp1 + psiNp1*psiNm1

	return phi, phi2, phi**2 - phi2

def print_all_phases(states, basis, p):
	"""
	Prints phase and phase fluctuations for all eigenstates. 
	"""
	listPhi, listPhiFluct = [], []
	phiString, phiFluctString = "Phi: ", "deltaPhi: "
	for state in states:
		phi, phi2, phi_fluct = calculate_phase(state, basis, p)
		phiString += f"{phi}, "
		phiFluctString += f"{phi_fluct}, "
		listPhi.append(phi)
		listPhiFluct.append(phi_fluct)
	print(phiString)
	print(phiFluctString)
	return listPhi, listPhiFluct	

###################################################################################################
# PRINTING RESULTS

def process_and_print_results(d, p):
	"""
	Prints results and saves them to the dictionary d. 
	"""
	for n in d:
		n_dict = d[n]
		energies, eigenstates, basis = n_dict["energies"], n_dict["eigenstates"], n_dict["basis"]
	
		print(f"RESULTS FOR n = {n}:")

		print_energies(n_dict, p)

		if p.print_states > 0:
			print("\nEigenvectors:")

			for i in range(p.print_states):
				print(f"i = {i}")
				print_state(eigenstates[i], basis, p.print_states_precision)
				print()

		if p.calc_occupancies:
			occs = print_all_occupancies(eigenstates, basis)
			d[n]["occupancies"] = occs

		if p.calc_phase:
			phase, phase_fluct = print_all_phases(eigenstates, basis, p)
			d[n]["phase"] = phase
			d[n]["phase_fluctuations"] = phase_fluct

	return d	