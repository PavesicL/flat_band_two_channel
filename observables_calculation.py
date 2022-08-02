#!/usr/bin/env python3

import numpy as np
import h5py
from helper import *
import cmath
import os
from parse_matrices import parse_phi_matrix 
###################################################################################################
# h5py functions

def h5dump(file, saveString, values):
	"""
	saveString has to end with / !!!
	"""
	saveString = saveString[:-1] # I AM SURE THIS USED TO WORK, BUT NOW THIS LINE IS NECCESSARY?? (JUNE 2022)
	file.create_dataset( saveString, data=values)

###################################################################################################
# PRINT ENERGIES

def print_and_save_energies(sector, n_dict, h5file, p):
	print("Energies: ")
	n, Sz = sector
	
	maxE = min(len(n_dict["energies"]), p.print_energies)

	Estr = ""
	for i, E in enumerate(n_dict["energies"]):
		h5dump(h5file, f"{n}/{Sz}/{i}/E/", E)

		Estr += f"{round(E, p.print_precision)}, "
	Estr = Estr[:-2]	
	print(Estr)

###################################################################################################
# PRINT AN EIGENSTATE

def print_states(eigenstates, basis, label, p):
	if p.print_states > 0:
		print(f"\nEigenvectors, {label}:")
		for i in range(p.print_states):
			
			print(f"i = {i}")
			print_state(eigenstates[i], basis, p)
			print()

def print_state(eigenvector, basis, p):
	printList = []
	for i, amplitude in enumerate(eigenvector):
		if abs(amplitude) > p.print_states_precision:
			size, phi = cmath.polar(amplitude)
			printList.append([size, phi/np.pi, basis[i]])
	printList = sorted(printList, key = lambda x : -abs(x[0]))
	#printList = sorted(printList, key = lambda x : x[2].dM)
	for amp, phi, bas in printList:
		print(f"{amp}	{round(phi, p.print_precision)}	{bas}")		

###################################################################################################
# OCCUPANCY CALCULATION

def calculate_occupancy(eigenvector, basis):
	"""
	Calculates the occupancy in all parts of the system for a given eigenvector.
	"""
	nimp, nL, nR = 0, 0, 0
	for i, amplitude in enumerate(eigenvector):
		if amplitude != 0:
			nimp += abs(amplitude)**2 * basis[i].nimp
			nL += abs(amplitude)**2 * basis[i].nL
			nR += abs(amplitude)**2 * basis[i].nR
	return nimp, nL, nR	

def print_and_save_all_occupancies(sector, h5file, states, basis, p):
	n, Sz = sector

	ns = ""
	for i, state in enumerate(states):
		nimp, nL, nR = calculate_occupancy(state, basis)			
		h5dump(h5file, f"{n}/{Sz}/{i}/nimp/", nimp)
		h5dump(h5file, f"{n}/{Sz}/{i}/nL/", nL)
		h5dump(h5file, f"{n}/{Sz}/{i}/nR/", nR)

		ns += f"({round(nimp, p.print_precision)}, {round(nL, p.print_precision)}, {round(nR, p.print_precision)}) "
	print(f"occupation: {ns}")	

###################################################################################################
# deltaM CALCULATION

def calculate_delta_M(eigenvector, basis):
	"""
	<delta M> 
	"""
	dM = 0
	for i, amplitude in enumerate(eigenvector):
		if amplitude != 0:
			dM += abs(amplitude)**2 * basis[i].dM
	return dM	

def calculate_delta_M2(eigenvector, basis):
	"""
	<delta M^2> 
	"""
	dM2 = 0
	for i, amplitude in enumerate(eigenvector):
		if amplitude != 0:
			dM2 += abs(amplitude)**2 * (basis[i].dM**2)
	return dM2	

def calculate_delta_M_amplitudes(n, eigenvector, basis):
	"""
	Returns the list of abs(amplitude) for each dM. 
	So summing over QP states, what is the amplitude of each dM in the eigenvector.
	"""
	dMmax = n//2

	res = np.zeros(1 + 2*dMmax)
	for i, amplitude in enumerate(eigenvector):
		state = basis[i]
		for dM in range(-dMmax,	dMmax+1):
			if state.dM == dM:
				res[dM + dMmax] += abs(amplitude)**2 # for dM == -dMmax, this will go to res[0]
	return res			

def print_and_save_dMs(sector, h5file, states, basis, p):
	n, Sz = sector

	dMs, dM2s = "", ""
	for i, state in enumerate(states):
		dM = calculate_delta_M(state, basis)			
		dM2 = calculate_delta_M2(state, basis)			
		amplitudes = calculate_delta_M_amplitudes(n, state, basis)	

		h5dump(h5file, f"{n}/{Sz}/{i}/dM/", dM)
		h5dump(h5file, f"{n}/{Sz}/{i}/dM2/", dM2)
		h5dump(h5file, f"{n}/{Sz}/{i}/dM_amplitudes/", amplitudes)

		dMs += f"{round(dM, p.print_precision)} "
		dM2s += f"{round(dM2, p.print_precision)} "
	print(f"dM: {dMs}")	
	print(f"dM2: {dM2s}")	

###################################################################################################
# PHASE CALCULATION

def calculate_phase(eigenvector, basis):
	"""
	This is equivalent to Eq. (3) form https://arxiv.org/pdf/cond-mat/0305361.pdf
	"""
	e_to_iphi = 0 + 0 * 1j
	for i, a_i in enumerate(eigenvector):
		for j, a_j in enumerate(eigenvector):
			if basis[i].QP_state == basis[j].QP_state:

				e_to_iphi +=  a_i.conjugate() * a_j * delta(basis[i].mL, basis[j].mL + 1 ) * delta( basis[i].mR, basis[j].mR - 1)
				#e_to_iphi += 0.5 * a_i.conjugate() * a_j * delta(basis[i].mL, basis[j].mL - 1 ) * delta( basis[i].mR, basis[j].mR + 1)

	size, phi = cmath.polar(e_to_iphi)
	return size, phi

def print_and_save_all_phases(sector, h5file, states, basis, p):
	n, Sz = sector

	sizes, phis = "", ""
	for i, state in enumerate(states):
		size, phi = calculate_phase(state, basis)

		h5dump(h5file, f"{n}/{Sz}/{i}/phi/", phi)
		h5dump(h5file, f"{n}/{Sz}/{i}/phi_size/", size)

		phis += f"{round(phi/np.pi, p.print_precision)} "
		sizes += f"{round(size, p.print_precision)} "
	print(f"phi/pi: {phis}")	
	print(f"phi size: {sizes}")

###################################################################################################
# QP PHASE CALCULATION

def calculate_QP_phase(Sz, eigenvector, basis):
	"""
	Reads out the matrix element of the operator f^dag_L,down f^dag_L,up f_R,up f_R,down and computes its expected value. 
	"""
	if Sz == 0:
		subspace = "singlet"
	elif Sz ==1/2:
		subspace = "doublet"

	matName = subspace + "_phi_mat_no_finite_size.dat"
	file_path =  os.path.abspath(os.path.dirname(__file__))	#this is the absolute path of the script! (eg. /home/pavesic/git_repos/flat_band_two_channel on spinon)

	operator_matrix = parse_phi_matrix(file_path + "/matrices/" + matName)

	e_to_iphi = 0 + 0 * 1j
	for i, a_i in enumerate(eigenvector):
		for j, a_j in enumerate(eigenvector):
			type_index_i, type_index_j = basis[i].type_index, basis[j].type_index

			matElement = operator_matrix[type_index_i][type_index_j]

			e_to_iphi += a_i.conjugate() * a_j * matElement( basis[i].mL, basis[i].mR, basis[j].mL, basis[j].mR)
	
	size, phi = cmath.polar(e_to_iphi)
	return size, phi

def print_and_save_all_QP_phases(sector, h5file, states, basis, p):
	n, Sz = sector

	sizes, phis = "", ""
	for i, state in enumerate(states):
		size, phi = calculate_QP_phase(Sz, state, basis)

		h5dump(h5file, f"{n}/{Sz}/{i}/QP_phi/", phi)
		h5dump(h5file, f"{n}/{Sz}/{i}/QP_phi_size/", size)

		phis += f"{round(phi/np.pi, p.print_precision)} "
		sizes += f"{round(size, p.print_precision)} "
	print(f"QP phi/pi: {phis}")	
	print(f"QP phi size: {sizes}")

###################################################################################################
# ABSOLUTE PHI DIRECTLY FROM BASIS VECTORS

def calculate_abs_phi(eigenvector, phi_basis):
	"""
	The eigenstates are typically (always?) a superposition of |phi> + |-phi>. The average phase as computed
	using various operators is then 0, or ill defined. Here, |phi| is computed by reflecting all phi > pi
	values across the real line (into the first two quadrants), and then taking the average, weighted by amplitudes. 
	"""
	avg, std = 0, 0
	for i, amp in enumerate(eigenvector):
		phi = phi_basis[i].phi

		if phi > np.pi:
			#reflect into the first two quadrants
			phi = 2*np.pi - phi

		avg += abs(amp)**2 * phi
		std += abs(amp)**2 * (phi**2)

	return avg, std	

def print_and_save_abs_phis(sector, h5file, states, phi_basis, p):
	n, Sz = sector

	abs_phis, abs_phi2s = "", ""
	for i, state in enumerate(states):
		phi, phi2 = calculate_abs_phi(state, phi_basis)			

		h5dump(h5file, f"{n}/{Sz}/{i}/abs_phi/", phi)
		h5dump(h5file, f"{n}/{Sz}/{i}/abs_phi2/", phi2)

		abs_phis += f"{round(phi/np.pi, p.print_precision)} "
		abs_phi2s += f"{round(phi2, p.print_precision)} "
	print(f"abs phi/pi: {abs_phis}")	
	print(f"abs phi^2: {abs_phi2s}")	

###################################################################################################
# SAVE THE AMPLITUDES OF ALL EIGENVECTORS IN THE phi BASIS

def get_phis_and_amps(state, phi_basis):
	"""
	The state is written in the phi basis. Accumulate a vector of 
	all phis in the basis and corresponding amplitudes in the given vector.
	"""

	phis, nqps, nqpSCs = [], [], []
	amps = []
	for i, phi_state in enumerate(phi_basis):
		phis.append( phi_state.phi )
		nqps.append( phi_state.QP_state.nqp )
		nqpSCs.append( phi_state.QP_state.nqpSC )
		amps.append( abs( state[i] )**2 )
		
	return phis, nqps, nqpSCs, amps

def print_and_save_phi_amplitudes(sector, h5file, states, phi_basis, p):
	n, Sz = sector

	for i, state in enumerate(states):
		phis, amps = get_phis_and_amps(state, phi_basis)	

		h5dump(h5file, f"{n}/{Sz}/{i}/all_phis/", phis)
		h5dump(h5file, f"{n}/{Sz}/{i}/all_nqps/", nqps)
		h5dump(h5file, f"{n}/{Sz}/{i}/all_nqpSCs/", nqpSCs)
		h5dump(h5file, f"{n}/{Sz}/{i}/phi_amps/", amps)

###################################################################################################
# NUMBER OF QUASIPARTICLES

def calculate_nqp(eigenvector, basis):

	nQP = 0
	for i, amplitude in enumerate(eigenvector):
		if amplitude != 0:
			nQP += abs(amplitude)**2 * (basis[i].nqp_no_imp**2)
	return nQP

def print_and_save_nqp(sector, h5file, states, basis, p):
	n, Sz = sector

	nqps = ""
	for i, state in enumerate(states):
		nqp = calculate_nqp(state, basis)			
		
		h5dump(h5file, f"{n}/{Sz}/{i}/nqp/", nqp)

		nqps += f"{round(nqp, p.print_precision)} "
	print(f"nqp: {nqps}")	

###################################################################################################
# PRINTING RESULTS

def process_save_and_print_results(d, h5file, p):
	"""
	Prints results and saves them to the hdf5 file. 
	"""
	for sector in d:
		n, Sz = sector

		n_dict = d[sector]
		energies = n_dict["energies"]
		dM_eigenstates, dM_basis = n_dict["dM_eigenstates"], n_dict["dM_basis"]
		
		if p.phase_fourier_transform:
			phi_eigenstates, phi_basis = n_dict["phi_eigenstates"], n_dict["phi_basis"]

		print()
		print("###################################################################################################")
		print(f"RESULTS FOR n = {n}, Sz = {Sz}:")
		print_and_save_energies(sector, n_dict, h5file, p)
		print_states(dM_eigenstates, dM_basis, "dM basis", p)
		
		if p.phase_fourier_transform:
			print_states(phi_eigenstates, phi_basis, "phi basis", p)

		if p.calc_occupancies:
			print_and_save_all_occupancies(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_dMs:
			print_and_save_dMs(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_phase:
			print_and_save_all_phases(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_QP_phase:
			print_and_save_all_QP_phases(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_abs_phase:
			print_and_save_abs_phis(sector, h5file, phi_eigenstates, phi_basis, p)
		if p.save_phi_amplitudes:
			print_and_save_phi_amplitudes(sector, h5file, phi_eigenstates, phi_basis, p)
		if p.calc_nqp:
			print_and_save_nqp(sector, h5file, dM_eigenstates, dM_basis, p)