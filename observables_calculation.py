#!/usr/bin/env python3

import numpy as np
import h5py
from helper import *
import cmath

###################################################################################################
# h5py functions

def h5dump(file, saveString, values):
	"""
	saveString has to end with / !!!
	"""
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

		Estr += f"{round(E, 5)}, "
	Estr = Estr[:-2]	
	print(Estr)

###################################################################################################
# PRINT AN EIGENSTATE

def print_states(eigenstates, basis, p):
	if p.print_states > 0:
		print("\nEigenvectors:")
		for i in range(p.print_states):
			
			print(f"i = {i}")
			print_state(eigenstates[i], basis, p.print_states_precision)
			print()

def print_state(eigenvector, basis, prec):
	printList = []
	for i, amplitude in enumerate(eigenvector):
		if abs(amplitude) > prec:
			size, phi = cmath.polar(amplitude)
			printList.append([size, phi/np.pi, basis[i]])
	#printList = sorted(printList, key = lambda x : -abs(x[0]))
	printList = sorted(printList, key = lambda x : x[2].dM)
	for amp, phi, bas in printList:
		print(f"{amp}	{round(phi, 3)}	{bas}")		

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

def print_and_save_all_occupancies(sector, h5file, states, basis):
	n, Sz = sector

	ns = ""
	for i, state in enumerate(states):
		nimp, nL, nR = calculate_occupancy(state, basis)			
		h5dump(h5file, f"{n}/{Sz}/{i}/nimp/", nimp)
		h5dump(h5file, f"{n}/{Sz}/{i}/nL/", nL)
		h5dump(h5file, f"{n}/{Sz}/{i}/nR/", nR)

		ns += f"({round(nimp, 4)}, {round(nL, 4)}, {round(nR, 4)}) "
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
	"""
	dMmax = n//2

	res = np.zeros(1 + 2*dMmax)
	for i, amplitude in enumerate(eigenvector):
		state = basis[i]
		for dM in range(-dMmax,	dMmax+1):
			if state.dM == dM:
				res[dM + dMmax] += abs(amplitude)**2 # for dM == -dMmax, this will go to res[0]
	return res			

def print_and_save_dMs(sector, h5file, states, basis):
	n, Sz = sector

	dMs, dM2s = "", ""
	for i, state in enumerate(states):
		dM = calculate_delta_M(state, basis)			
		dM2 = calculate_delta_M2(state, basis)			
		amplitudes = calculate_delta_M_amplitudes(n, state, basis)	

		h5dump(h5file, f"{n}/{Sz}/{i}/dM/", dM)
		h5dump(h5file, f"{n}/{Sz}/{i}/dM2/", dM2)
		h5dump(h5file, f"{n}/{Sz}/{i}/dM_amplitudes/", amplitudes)

		dMs += f"{round(dM, 4)} "
		dM2s += f"{round(dM2, 4)} "
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

def print_and_save_all_phases(sector, h5file, states, basis):
	n, Sz = sector

	sizes, phis = "", ""
	for i, state in enumerate(states):
		size, phi = calculate_phase(state, basis)

		h5dump(h5file, f"{n}/{Sz}/{i}/phi/", phi)
		h5dump(h5file, f"{n}/{Sz}/{i}/phi_size/", size)

		phis += f"{round(phi/np.pi, 4)} "
		sizes += f"{round(size, 4)} "
	print(f"phi/pi: {phis}")	
	print(f"phi size: {sizes}")

###################################################################################################
# NUMBER OF QUASIPARTICLES

def calculate_nqp(eigenvector, basis):

	nQP = 0
	for i, amplitude in enumerate(eigenvector):
		if amplitude != 0:
			nQP += abs(amplitude)**2 * (basis[i].nqp_no_imp**2)
	return nQP

def print_and_save_nqp(sector, h5file, states, basis):
	n, Sz = sector

	nqps = ""
	for i, state in enumerate(states):
		nqp = calculate_nqp(state, basis)			
		
		h5dump(h5file, f"{n}/{Sz}/{i}/nqp/", nqp)

		nqps += f"{round(nqp, 4)} "
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
		energies, eigenstates, basis = n_dict["energies"], n_dict["eigenstates"], n_dict["basis"]

		print()
		print("###################################################################################################")
		print(f"RESULTS FOR n = {n}, Sz = {Sz}:")
		print_and_save_energies(sector, n_dict, h5file, p)
		print_states(eigenstates, basis, p)

		if p.calc_occupancies:
			print_and_save_all_occupancies(sector, h5file, eigenstates, basis)
		if p.calc_dMs:
			print_and_save_dMs(sector, h5file, eigenstates, basis)
		if p.calc_phase:
			print_and_save_all_phases(sector, h5file, eigenstates, basis)
		if p.calc_nqp:
			print_and_save_nqp(sector, h5file, eigenstates, basis)