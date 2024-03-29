#!/usr/bin/env python3

import numpy as np
import h5py
from helper import *
import cmath
import os
from parse_matrices import parse_phi_matrix
from matrix import generate_computation_basis, write_vector_in_computation_basis

import my_second_quantization.operators as op
import my_second_quantization.bitwise_ops as bo

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

		if not p.turn_off_all_finite_size_effects: #to compare this with Lanczos/DMRG
			E -= -p.U/2
		Estr += f"{round(E, p.print_precision)}, "
	Estr = Estr[:-2]
	print(Estr)

###################################################################################################
# PRINT AN EIGENSTATE

def print_states(eigenstates, basis, label, p):
	if p.number_of_states_to_print > 0:
		print(f"\nEigenvectors, {label}:")
		for i in range(p.number_of_states_to_print):

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
	# multiply so that the largest contribution has a real valued amplitude
	first_phi = printList[0][1]
	for amp, phi, bas in printList:
		print(f"{amp}	exp({round(phi - first_phi, 3)}/pi)	{bas}")

###################################################################################################
# OCCUPANCY CALCULATION

def calculate_occupancy(dM_eigenvector, dM_basis):
	"""
	Calculates the occupancy in all parts of the system for a given eigenvector.
	"""
	nimp, nL, nR = 0, 0, 0
	nqp = 0
	for i, amplitude in enumerate(dM_eigenvector):
		if amplitude != 0:
			nimp += abs(amplitude)**2 * dM_basis[i].nimp
			nL += abs(amplitude)**2 * dM_basis[i].nL
			nR += abs(amplitude)**2 * dM_basis[i].nR
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
# nQD OVERLAPS/MATRIX ELEMENTS

def calculate_nQD_matrix_element(psi1, psi2, dM_basis):
	"""
	Calculate the overlap <psi1|nQD|psi2>.
	"""
	# apply nQD to psi2
	nQD_psi2 = [ psi2[i] * dM_basis[i].nimp for i in range(len(dM_basis)) ]
	# compute the overlap
	res = np.dot( np.conjugate(psi1), nQD_psi2 )
	return res

def print_and_save_all_nQD_matrix_elements(sector, h5file, states, basis, p):
	n, Sz = sector

	nQDs = np.zeros( shape = (p.number_of_overlaps, p.number_of_overlaps), dtype="complex128")
	for i in range(p.number_of_overlaps):
		for j in range(p.number_of_overlaps):
			nQDs[i,j] += calculate_nQD_matrix_element(states[i], states[j], basis)

	h5dump(h5file, f"{n}/{Sz}/matrix_elements/nQD/", nQDs)

###################################################################################################
# nL OVERLAPS/MATRIX ELEMENTS

def calculate_nL_matrix_element(psi1, psi2, dM_basis):
	"""
	Calculate the overlap <psi1|nL|psi2>.
	"""
	# apply nL to psi2
	nL_psi2 = [ psi2[i] * dM_basis[i].nL for i in range(len(dM_basis)) ]
	# compute the overlap
	res = np.dot( np.conjugate(psi1), nL_psi2 )
	return res

def print_and_save_all_nL_matrix_elements(sector, h5file, states, basis, p):
	n, Sz = sector

	nLs = np.zeros( shape = (p.number_of_overlaps, p.number_of_overlaps), dtype="complex128")
	for i in range(p.number_of_overlaps):
		for j in range(p.number_of_overlaps):
			nLs[i,j] += calculate_nL_matrix_element(states[i], states[j], basis)

	h5dump(h5file, f"{n}/{Sz}/matrix_elements/nL/", nLs)

###################################################################################################
# nR OVERLAPS/MATRIX ELEMENTS

def calculate_nR_matrix_element(psi1, psi2, dM_basis):
	"""
	Calculate the overlap <psi1|nR|psi2>.
	"""
	# apply nR to psi2
	nR_psi2 = [ psi2[i] * dM_basis[i].nR for i in range(len(dM_basis)) ]
	# compute the overlap
	res = np.dot( np.conjugate(psi1), nR_psi2 )
	return res

def print_and_save_all_nR_matrix_elements(sector, h5file, states, basis, p):
	n, Sz = sector

	nRs = np.zeros( shape = (p.number_of_overlaps, p.number_of_overlaps), dtype="complex128")
	for i in range(p.number_of_overlaps):
		for j in range(p.number_of_overlaps):
			nRs[i,j] += calculate_nR_matrix_element(states[i], states[j], basis)

	h5dump(h5file, f"{n}/{Sz}/matrix_elements/nR/", nRs)
	
###################################################################################################
# identity OVERLAPS/MATRIX ELEMENTS

def calculate_ID_matrix_element(psi1, psi2, dM_basis):
	"""
	Calculate the overlap <psi1|psi2>.
	"""
	# compute the overlap
	res = np.dot( np.conjugate(psi1), psi2 )
	return res

def print_and_save_all_ID_matrix_elements(sector, h5file, states, basis, p):
	n, Sz = sector

	IDs = np.zeros( shape = (p.number_of_overlaps, p.number_of_overlaps), dtype="complex128")
	for i in range(p.number_of_overlaps):
		for j in range(p.number_of_overlaps):
			IDs[i,j] += calculate_ID_matrix_element(states[i], states[j], basis)

	h5dump(h5file, f"{n}/{Sz}/matrix_elements/ID/", IDs)
	
###################################################################################################
# total Sz CALCULATION

def calculate_Sz(dM_eigenvector, dM_basis):
	Sz = 0
	for i, amplitude in enumerate(dM_eigenvector):
		if amplitude != 0:
			Sz += abs(amplitude)**2 * dM_basis[i].Sz
	return Sz

def print_and_save_total_Sz(sector, h5file, states, basis, p):

	n, Sz = sector

	Szs = ""
	for i, state in enumerate(states):
		totSz = calculate_Sz(state, basis)
		h5dump(h5file, f"{n}/{Sz}/{i}/tot_Sz/", totSz)

		Szs += f"{round(totSz, p.print_precision)}, "

	print(f"Sz: {Szs}")
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

def print_and_save_dMs(sector, h5file, states, basis, p):
	n, Sz = sector

	dMs, dM2s = "", ""
	for i, state in enumerate(states):
		dM = calculate_delta_M(state, basis)
		dM2 = calculate_delta_M2(state, basis)

		h5dump(h5file, f"{n}/{Sz}/{i}/dM/", dM)
		h5dump(h5file, f"{n}/{Sz}/{i}/dM2/", dM2)

		dMs += f"{round(dM, p.print_precision)} "
		dM2s += f"{round(dM2, p.print_precision)} "
	print(f"dM: {dMs}")
	print(f"dM2: {dM2s}")

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

def calculate_abs_phi(eigenvector, phi_basis, p):
	"""
	The eigenstates are typically (always?) a superposition of |phi> + |-phi>. The average phase as computed
	using various operators is then 0, or ill defined. Here, |phi| is computed by reflecting all phi > pi
	values across the real line (into the first two quadrants), and then taking the average, weighted by amplitudes.
	"""
	avg, std = 0, 0
	for i, amp in enumerate(eigenvector):
		phi = phi_basis[i].phi
		phi -= p.phiext # compute this relative to phiext!

		if phi > np.pi:
			#reflect into the first two quadrants
			phi = 2*np.pi - phi

		#now express phi in the units of pi!!! Also put phiext back in.
		phi = ( phi + p.phiext ) / np.pi

		avg += abs(amp)**2 * phi
		std += abs(amp)**2 * (phi**2)

	return avg, std

def print_and_save_abs_phis(sector, h5file, states, phi_basis, p):
	n, Sz = sector

	abs_phis, abs_phi2s = "", ""
	fluctuations = ""
	for i, state in enumerate(states):
		phi, phi2 = calculate_abs_phi(state, phi_basis, p)
		fluct = abs( phi**2 - phi2 )

		h5dump(h5file, f"{n}/{Sz}/{i}/abs_phi/", phi)
		h5dump(h5file, f"{n}/{Sz}/{i}/abs_phi2/", phi2)

		abs_phis += f"{round(phi, p.print_precision)} "
		abs_phi2s += f"{round(phi2, p.print_precision)} "
		fluctuations += f"{round(fluct, p.print_precision)} "

	print(f"abs phi/pi: {abs_phis}")
	print(f"abs phi^2: {abs_phi2s}")
	print(f"fluctuations: {fluctuations}")

###################################################################################################
# COMPUTE <sin(phi)> AND <cos(phi)>

def calculate_sin_cos_phi(eigenvector, phi_basis):
	"""
	Calculate expected values of cos(phi) and sin(phi).
	"""
	c, s = 0, 0
	for i, amp in enumerate(eigenvector):
		phi = phi_basis[i].phi
		absamp = abs(amp)**2

		c += absamp	* np.cos(phi)
		s += absamp * np.sin(phi)

	return c, s

def print_and_save_sin_cos_phi(sector, h5file, states, phi_basis, p):
	n, Sz = sector

	cs, ss, atans = "", "", ""
	for i, state in enumerate(states):
		c, s = calculate_sin_cos_phi(state, phi_basis)
		atan2 = np.arctan2(s, c)

		h5dump(h5file, f"{n}/{Sz}/{i}/cos_phi/", c)
		h5dump(h5file, f"{n}/{Sz}/{i}/sin_phi/", s)
		h5dump(h5file, f"{n}/{Sz}/{i}/atan2_phi/", atan2)

		cs += f"{round(c, p.print_precision)} "
		ss += f"{round(s, p.print_precision)} "
		atans += f"{round(atan2/np.pi, p.print_precision)} "

	print(f"cos(phi): {cs}")
	print(f"sin(phi): {ss}")
	print(f"atan(s, c)/pi: {atans}")

###################################################################################################
# dM AMPLITUDES - SAVE THE AMPLITUDES OF ALL EIGENVECTORS IN THE dM BASIS

def get_delta_Ms_and_amps(dM_eigenvector, dM_basis):
	"""
	Returns the list of abs(amplitude) for each dM.
	So summing over QP states, what is the amplitude of each dM in the eigenvector.
	"""
	dMs, nqps, amps = [], [], []
	for i, state in enumerate(dM_basis):
		dMs.append( state.dM )
		nqps.append( state.nqp )
		amps.append( abs( dM_eigenvector[i] )**2 )
	return dMs, nqps, amps

def print_and_save_dM_amplitudes(sector, h5file, states, basis, p):
	n, Sz = sector
	for i, state in enumerate(states):
		dMs, nqps, amplitudes = get_delta_Ms_and_amps(state, basis)
		h5dump(h5file, f"{n}/{Sz}/{i}/dM_amplitudes/dMs/", dMs)
		h5dump(h5file, f"{n}/{Sz}/{i}/dM_amplitudes/nqps/", nqps)
		h5dump(h5file, f"{n}/{Sz}/{i}/dM_amplitudes/amplitudes/", amplitudes)
	print(f"{i} dM vectors saved")

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
		nqps.append( phi_state.nqp )
		nqpSCs.append( phi_state.nqp_SC )
		amps.append( abs( state[i] )**2 )

	return phis, nqps, nqpSCs, amps

def print_and_save_phi_amplitudes(sector, h5file, states, phi_basis, p):
	n, Sz = sector
	for i, state in enumerate(states):
		phis, nqps, nqpSCs, amplitudes = get_phis_and_amps(state, phi_basis)
		h5dump(h5file, f"{n}/{Sz}/{i}/phi_amplitudes/phis/", phis)
		h5dump(h5file, f"{n}/{Sz}/{i}/phi_amplitudes/nqps/", nqps)
		h5dump(h5file, f"{n}/{Sz}/{i}/phi_amplitudes/nqpSCs/", nqpSCs)
		h5dump(h5file, f"{n}/{Sz}/{i}/phi_amplitudes/amplitudes/", amplitudes)
	print(f"{i} phi vectors saved")

###################################################################################################

def trace_out_QD_amps(dM_eigenvector, dM_basis):
	"""
	Sum all amplitudes with the same dM.
	"""

	unique_dMs = []
	for state in dM_basis:
		unique_dMs.append(state.dM)
	unique_dMs = np.unique(unique_dMs)
	unique_amps = np.zeros(len(unique_dMs), dtype=complex)

	for i, state in enumerate(dM_basis):
		ndx = list(unique_dMs).index(state.dM)
		unique_amps[ndx] += dM_eigenvector[i]

	return unique_dMs, unique_amps

def get_amp(dm, dmdict):
	try:
		return dmdict[dm]
	except KeyError:
		return 0

def wigner(dm, phi, n, dmdict):
	res = 0
	for y in range(-n, n+1, 1):
		amp1 = get_amp(dm - y, dmdict)
		amp2 = get_amp(dm + y, dmdict)
		fac = np.exp( 1j * 2 * y * phi )

		res += np.conj(amp1) * amp2 * fac
	res /= np.pi
	return res

def check_if_all_real(vals, prec = 10**(-8)):
	vals = np.array(vals).flatten()
	for v in vals:
		if abs(v.imag) > prec:
			raise Exception(f"Value {v} is not real in the calculation of the Wigner distribution!")

def calc_wigner_dist(n, dM_eigenvector, dM_basis):

	unique_dMs, unique_amps = trace_out_QD_amps(dM_eigenvector, dM_basis)
	dmdict = dict(zip(unique_dMs, unique_amps))
	phis = np.linspace(0, 2 * pi, len(unique_dMs))

	res = [[wigner(dm, phi, n, dmdict) for dm in unique_dMs] for phi in phis]
	check_if_all_real(res)
	res = np.real(res)
	return res, unique_dMs, phis

def print_and_save_wigner_distribution(sector, h5file, states, basis, p):
	n, Sz = sector
	for i, state in enumerate(states):
		vals, dMs, phis = calc_wigner_dist(n, state, basis)
		h5dump(h5file, f"{n}/{Sz}/{i}/wigner_distribution/dMs/", dMs)
		h5dump(h5file, f"{n}/{Sz}/{i}/wigner_distribution/phis/", phis)
		h5dump(h5file, f"{n}/{Sz}/{i}/wigner_distribution/vals/", vals)
	print(f"{i} wigner distributions saved")


###################################################################################################
# NUMBER OF QUASIPARTICLES

def calculate_nqp(comp_eigenstate):
	"""
	Calculates the number of quasiparticles in the SC channels.
	"""

	nOp = lambda i, s : op.OPERATOR_STRING( op.OPERATOR( "n", i, s ) )

	nL = op.expectedValue( nOp(1, "UP"), comp_eigenstate ) + op.expectedValue( nOp(1, "DOWN"), comp_eigenstate )
	nR = op.expectedValue( nOp(2, "UP"), comp_eigenstate ) + op.expectedValue( nOp(2, "DOWN"), comp_eigenstate )
	return np.real( nL + nR )

def print_and_save_nqp(sector, h5file, states, p):
	n, Sz = sector

	nqps = ""
	for i, state in enumerate(states):
		nqp = calculate_nqp(state)

		h5dump(h5file, f"{n}/{Sz}/{i}/nqp/", nqp)

		nqps += f"{round(nqp, p.print_precision)} "
	print(f"nqp: {nqps}")

###################################################################################################
# SPIN CORRELATIONS

def calc_SS(comp_eigenstate, i, j):
	"""
	Spin correlations are S_i . S_j = Sz.Sz + 0.5 ( S+.S- + S-.S+ ).
	"""

	SzSzOp = op.OPERATOR_STRING( ("Sz", i), ("Sz", j) )
	#SpSmOp = op.OPERATOR_STRING( op.OPERATOR("cdag", i, "UP"), op.OPERATOR("c", i, "DOWN"), op.OPERATOR("c", i, "DOWN"), op.OPERATOR("c", i, "DOWN") )
	SpSmOp = op.OPERATOR_STRING( ("cdag", i, "UP"), ("c", i, "DOWN"), ("cdag", j, "DOWN"), ("c", j, "UP") )
	SmSpOp = op.OPERATOR_STRING( ("cdag", i, "DOWN"), ("c", i, "UP"), ("cdag", j, "UP"), ("c", j, "DOWN") )

	ss = op.expectedValue( SzSzOp, comp_eigenstate )
	spsm = op.expectedValue( SpSmOp, comp_eigenstate )
	smsp = op.expectedValue( SmSpOp, comp_eigenstate )

	return np.real(ss + 0.5 * ( spsm + smsp ))

def print_and_save_imp_spin_correlations(sector, h5file, computational_eigenstates, p):
	n, Sz = sector

	impimps, impLs, impRs = "", "", ""
	for i, state in enumerate(computational_eigenstates):
		impimp = calc_SS(state, 0, 0)
		LL = calc_SS(state, 1, 1)
		RR = calc_SS(state, 2, 2)

		impL = calc_SS(state, 0, 1)
		impR = calc_SS(state, 0, 2)
		LR = calc_SS(state, 1, 2)

		h5dump(h5file, f"{n}/{Sz}/{i}/SS/impimp/", impimp)
		h5dump(h5file, f"{n}/{Sz}/{i}/SS/LL/", LL)
		h5dump(h5file, f"{n}/{Sz}/{i}/SS/RR/", RR)
		h5dump(h5file, f"{n}/{Sz}/{i}/SS/impL/", impL)
		h5dump(h5file, f"{n}/{Sz}/{i}/SS/impR/", impR)
		h5dump(h5file, f"{n}/{Sz}/{i}/SS/LR/", LR)

		impimps += f"{round(impimp, p.print_precision)} "
		impLs += f"{round(impL, p.print_precision)} "
		impRs += f"{round(impR, p.print_precision)} "
	print(f"Simp.Simp: {impimps}")
	print(f"Simp.Sl: {impLs}")
	print(f"Simp.Sr: {impRs}")

###################################################################################################
# PARITY

def parity_transform_bitstring(bitstring):
	"""
	We know that this is a bitstring for 3 sites. The IMP level is left alone, L and R are switched.
	If L and R both have occupancy 1, get a -1 prefactor, otherwise +1.
	This is the most direct way to do this, but probably transparent and good enough.
	"""
	prefactor = 1
	new_bitstring = 0

	impUp = bo.bit(bitstring, 5)
	impDn = bo.bit(bitstring, 4)
	LUp = bo.bit(bitstring, 3)
	LDn = bo.bit(bitstring, 2)
	RUp = bo.bit(bitstring, 1)
	RDn = bo.bit(bitstring, 0)

	if LUp + LDn == 1 and RUp + RDn == 1:
		prefactor *= -1

	#now switch R and L
	new_bitstring += 1 * LDn + 2 * LUp
	new_bitstring += 4 * RDn + 8 * RUp
	new_bitstring += 16 * impDn + 32 * impUp

	return new_bitstring, prefactor

def apply_parity_op(state):
	"""
	Transforms the basis string and switches mL and mR.
	When both L and R have one particle, the vector gets a -1 prefactor.
	"""

	new_vector, new_basis = [], []
	for i, basis_state in enumerate(state.basis):


		new_bitstring, prefactor = parity_transform_bitstring( basis_state.bitstring )
		new_mL = basis_state.quantum_numbers["mR"]
		new_mR = basis_state.quantum_numbers["mL"]

		new_basis.append(op.BASIS_STATE( bitstring = new_bitstring, mL = new_mL, mR = new_mR ))
		new_vector.append( prefactor * state.vector[i] )

		"""
		if state.vector[i] != 0:
			print()
			print("basis state: ", basis_state)
			print("new basis state: ", op.BASIS_STATE( bitstring = new_bitstring, mL = new_mL, mR = new_mR ))
			print(state.vector[i], prefactor, prefactor * state.vector[i])
		"""

	new_vector, new_basis = zip(*sorted(zip(new_vector, new_basis), key= lambda x : x[1]))
	res = np.dot( np.conjugate(state.vector), new_vector )
	"""
	print(f"Parity is: {res}\n")
	print("INITIAL STATE:")
	print(state)
	print("Resulting state is:")
	print(op.STATE( vector = np.array(new_vector, dtype=complex), basis = np.array(new_basis, dtype=op.BASIS_STATE), N=state.N) )
	print("DONE\n\n")
	"""
	return op.STATE( vector = np.array(new_vector, dtype=complex), basis = np.array(new_basis, dtype=op.BASIS_STATE), N=state.N)

def calculate_parity(calc_state):
	"""
	The space parity operator transforms the creation operators in L to the ones in R and opposite.
	So a given string of cdag_L cdag_R -> cdag_R cdag_L.
	Then, these have to be reshuffled back into original IMP, L, R order.
	Pairs do not gain a prefactor, so mL and mR can simply be exchanged.
	The QP_STATE can gain a prefactor only if there is one particle in L AND one in R.
	This is the only way one gets a single commutation and thus a minus.
	"""
	P_state = apply_parity_op(calc_state)
	return np.real( np.dot( np.conjugate(calc_state.vector), P_state.vector ) )

def print_and_save_parity(sector, h5file, calc_states, p):
	n, Sz = sector

	Ps = ""
	Psum = 0
	for i, state in enumerate(calc_states):
		P = calculate_parity(state)
		h5dump(h5file, f"{n}/{Sz}/{i}/parity/", P)
		Ps += f"{round(P, p.print_precision)} "
		Psum += P
	print(f"parity: {Ps}")
	print(f"sum of all Ps (should be integer): {Psum}")

###################################################################################################

def calc_dEs(energies, p):
	difsString = ""
	for i in range(len(energies)-1):
		d = energies[i+1] - energies[i]
		difsString += f"{round(d, p.print_precision)} "
	return difsString

def print_dEs(sector, energies, p):
	"""
	Print the energy difference to the previous state.
	"""
	n, Sz = sector
	difsString = calc_dEs(energies, p)
	print(f"dEs: {difsString}")

###################################################################################################
# PRINTING RESULTS

def process_save_and_print_results(d : dict, h5file : str, p):
	"""
	Prints results and saves them to the hdf5 file.
	"""
	for sector in d:
		n, Sz = sector

		n_dict = d[sector]
		energies = n_dict["energies"]
		dM_eigenstates, dM_basis = n_dict["dM_eigenstates"], n_dict["dM_basis"]

		# THE COMPUTATIONAL BASIS IS USEFUL FOR EXTRACTING QUANTITIES WHICH ARE NOT GOOD QUANTUM NUMBERS
		computational_basis = generate_computation_basis(n, Sz, p)
		computational_eigenstates = [ write_vector_in_computation_basis(eigenstate, dM_basis, computational_basis) for eigenstate in dM_eigenstates ]

		if p.phase_fourier_transform:
			phi_eigenstates, phi_basis = n_dict["phi_eigenstates"], n_dict["phi_basis"]

		print()
		print("###################################################################################################")
		print(f"RESULTS FOR n = {n}, Sz = {Sz}:")
		print_and_save_energies(sector, n_dict, h5file, p)

		if p.print_states_dM:
			print_states(dM_eigenstates, dM_basis, "dM basis", p)
		if p.phase_fourier_transform and p.print_states_phi:
			print_states(phi_eigenstates, phi_basis, "phi basis", p)

		if p.calc_occupancies:
			print_and_save_all_occupancies(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_tot_Sz:
			print_and_save_total_Sz(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_dMs:
			print_and_save_dMs(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_QP_phase:
			print_and_save_all_QP_phases(sector, h5file, dM_eigenstates, dM_basis, p) #this depends on matrix elements written in the dM_basis
		if p.calc_abs_phase:
			print_and_save_abs_phis(sector, h5file, phi_eigenstates, phi_basis, p)
		if p.calc_sin_cos_phi:
			print_and_save_sin_cos_phi(sector, h5file, phi_eigenstates, phi_basis, p)
		if p.save_dM_amplitudes:
			print_and_save_dM_amplitudes(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.save_phi_amplitudes:
			print_and_save_phi_amplitudes(sector, h5file, phi_eigenstates, phi_basis, p)
		if p.calc_wigner_distribution:
			print_and_save_wigner_distribution(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_nqp:
			print_and_save_nqp(sector, h5file, computational_eigenstates, p)
		if p.calc_imp_spin_correlations:
			print_and_save_imp_spin_correlations(sector, h5file, computational_eigenstates, p)
		if p.calc_parity:
			print_and_save_parity(sector, h5file, computational_eigenstates, p)
		if p.calc_dEs:
			print_dEs(sector, energies, p)
		if p.calc_nQD_matrix_elements:
			print_and_save_all_nQD_matrix_elements(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_nL_matrix_elements:
			print_and_save_all_nL_matrix_elements(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_nR_matrix_elements:
			print_and_save_all_nR_matrix_elements(sector, h5file, dM_eigenstates, dM_basis, p)
		if p.calc_ID_matrix_elements:
			print_and_save_all_ID_matrix_elements(sector, h5file, dM_eigenstates, dM_basis, p)
