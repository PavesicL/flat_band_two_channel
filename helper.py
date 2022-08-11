#!/usr/bin/env python3

from enum import Enum, auto
from dataclasses import dataclass, fields
import numpy as np
from math import pi
import re
###################################################################################################
# CLASS CONTAINING PARAMETERS

#@dataclass(slots=True) - this works with Python 3.10
@dataclass()
class params:

	UNSPECIFIED_DEFAULT = object() # Empty object, used as a flag for where the default value depends on other attributes. These are handled with the set_default() method.

	N: int #system size, required!

	#imp
	U: float = 0.
	epsimp: float = UNSPECIFIED_DEFAULT

	#left channel
	alpha_L: float = 0.
	Ec_L: float = 0.
	n0_L:	float = 0.

	#right channel
	alpha_R: float = 0.
	Ec_R: float = 0.
	n0_R: float = 0.

	#hopping
	# Either specify gammas, which will transform into v later, or specify v only!
	# Gammas are not used in the calculation.
	gamma_L: float = 0.
	gamma_R: float = 0.
	v_L: float = UNSPECIFIED_DEFAULT
	v_R: float = UNSPECIFIED_DEFAULT

	#setting the calculations parameters
	nrange: int = 0
	nref: int = UNSPECIFIED_DEFAULT
	refisn0: bool = False
	verbose: bool = False

	#set calculations to perform and things to print
	print_energies: int = 10
	print_states: int = 0
	print_states_precision: float = 0.01
	print_precision: int = 5

	calc_occupancies: bool = True
	calc_dMs: bool = True
	calc_phase: bool = True
	calc_QP_phase: bool = True
	calc_abs_phase: bool = True
	save_dM_amplitudes: bool = False
	save_phi_amplitudes: bool = False
	calc_nqp: bool = True
	calc_imp_spin_correlations: bool = True
	calc_parity: bool = True
	
	save_all_states: bool = False
	num_states_to_save: int = 10
	parallel: bool = False

	#transformations
	save_matrix: bool = False
	reorder_matrix_dM: bool = True
	phase_fourier_transform: bool = False

	#approximations
	turn_off_all_finite_size_effects: bool = False
	turn_off_SC_finite_size_effects: bool = UNSPECIFIED_DEFAULT			# no finite size effects in SCs - the energy is just alpha * number of QPs
	turn_off_hopping_finite_size_effects: bool = UNSPECIFIED_DEFAULT	# all hoppings are taken in the limit of half filling and large L
	use_all_states: bool = UNSPECIFIED_DEFAULT							# does not throw away "unphysical" states, eg. a state with mL = L and a quasiparticle in the same channel is kept
	add_periodic_hopping_blocks: bool = UNSPECIFIED_DEFAULT				# adds hopping between m = L and m = 0 blocks

	def __post_init__(self):

		# All parameters are given as a dictionary of strings, parsed form the input file.
		# Here, cast them into their specified type using eval.
		for field in fields(self):
			if not self.unspecified_default(field.name):	#if it is not unspecified default - these are handled below
				value = getattr(self, field.name)
				if isinstance(value, str):		#if the given value is a string
					object.__setattr__(self, field.name, eval(value, {"false" : False, "true" : True}))

		
		# set the default values here!
		self.set_default("epsimp", -self.U/2)
		self.set_default("v_L", np.sqrt( 2 * self.gamma_L / pi))
		self.set_default("v_R", np.sqrt( 2 * self.gamma_R / pi))
		
		default_nref = 0
		if self.U != 0:
			default_nref += (0.5 - (self.epsimp/self.U))
		
		if self.Ec_L != 0:
			default_nref += self.n0_L
		else:
			default_nref += self.LL	

		if self.Ec_R != 0:
			default_nref += self.n0_R
		else:
			default_nref += self.LL	
	
		self.set_default("nref", int(default_nref))
		
		self.set_default("turn_off_SC_finite_size_effects", self.turn_off_all_finite_size_effects)
		self.set_default("turn_off_hopping_finite_size_effects", self.turn_off_all_finite_size_effects)
		self.set_default("use_all_states", self.turn_off_all_finite_size_effects)
		self.set_default("add_periodic_hopping_blocks", self.turn_off_all_finite_size_effects)

	@property
	def LL(self):
		"""
		Set LL - size of each SC channel.
		"""
		if self.N%2 == 0:
			raise Exception("N has to be odd!")
		else:
			return int((self.N - 1)//2)

	def unspecified_default(self, paramName):
		"""
		Check whether the parameter is an unspecified default value.
		"""
		if getattr(self, paramName) is self.UNSPECIFIED_DEFAULT:
			return True
		else:
			return False

	def set_default(self, paramName, defaultValue):
		"""
		Given the name of a parameter (string!), sets its default value if it is currently unpecified.
		"""
		if self.unspecified_default(paramName):
			object.__setattr__(self, paramName, defaultValue)

	@property
	def subspace_list(self):
		"""
		Generates the list of subspaces in which to perform the calculation.
		"""
		subspace_list = [self.nref,]
		for n in range(1, self.nrange+1):
			subspace_list.append(self.nref-n)			
			subspace_list.append(self.nref+n)			
		return sorted(subspace_list)

	def __str__(self):
		representation = ""
		for field in fields(self):
			value = getattr(self, field.name)
			representation += f"{field.name} = {value}\n"
		return representation	
		
###################################################################################################
# PARSING THE PARAMETERS FROM FILE

def parse_input_to_dict(inputFile):
	"""
	Parses all parameters in the inputFile into a dictionary. All values are still strings!
	"""
	d = {}
	with open(inputFile, "r") as file:
		for line in file:
			line = line.strip()

			if len(line) > 0:	
				if not line[0] == "#": #comment
					a = re.search(r"(.*?)=(.*)", line)
					if a:
						name = a.group(1).strip()
						val = a.group(2).strip()
						d[name] = val
	return d			

def parse_params(inputFile):
	"""
	Generates the params object p with all parameters.
	"""
	d = parse_input_to_dict(inputFile)
	p = params(**d)
	return p

###################################################################################################
# ENUM NAMES THE QP STATES - THESE ARE EXPORTED INTO THE GLOBAL NAMESPACE!

class QP(Enum):
	"""	
	Allowed types of quasiparticle states.
	"""
	ZERO = (0, 0)
	UP = (1/2, 1)
	DOWN = (-1/2, 1)
	UPDN = (0, 2)

	def __init__(self, Sz, n):
		self.Sz = Sz
		self.n = n

	@property
	def bit_integer(self):
		"""
		The returned value in bitstring form corresponds to the occupation
		basis vector for the QP state.
		"""
		if self == ZERO:
			return 0
		elif self == DOWN:
			return 1
		elif self == UP:
			return 2
		elif self == UPDN:
			return 3

	@classmethod
	def export_to(cls, namespace):
	    namespace.update(cls.__members__) 

	def __str__(self):
		"""
		This makes it so printing the enum gives ZERO instead of QP.ZERO
		"""
		return self.name 

QP.export_to(globals()) #this exports the class enums into global namespace, so they can be called by eg. ZERO instead of QP.ZERO

class QP_STATE:
	"""
	A state of QPs at the impurity and both channels.
	"""

	def __init__(self, qp_imp, qp_L, qp_R):
		self.qp_imp = qp_imp
		self.qp_L = qp_L
		self.qp_R = qp_R
	
		self.nqp = qp_imp.n + qp_L.n + qp_R.n
		self.nqpSC = qp_L.n + qp_R.n
		self.Sz = qp_imp.Sz + qp_L.Sz + qp_R.Sz

	def bitstring(self):
		return self.qp_R.bit_integer + ( 4 * self.qp_L.bit_integer ) + ( 16 * self.qp_imp.bit_integer ) 


	def __repr__(self):
		return f"({str(self.qp_imp)}, {str(self.qp_L)}, {str(self.qp_R)})"

	def __eq__(self, other):
		return self.qp_imp == other.qp_imp and self.qp_L == other.qp_L and self.qp_R == other.qp_R

###################################################################################################
# STATES DEFINING CLASSES

@dataclass
class IMP:
	"""
	Class for the impurity level.
	"""
	state: QP

	U: float
	epsimp: float

	@property
	def n(self):
		return self.state.n

	def energy(self):
		#Upart = self.U if self.state == UPDN else 0
		#return (self.epsimp * self.state.n) + Upart

		if self.U != 0:
			nu = 0.5 - ( self.epsimp / self.U )
			return 0.5* self.U * ( self.state.n - nu )**2
		else:
			return self.epsimp * self.state.n	

	def __eq__(self, other):
		return self.state == other.state

@dataclass
class SC_BATH:
	"""
	Class for the SC level. Contains the number of Cooper pairs and the quasiparticle state.
	"""
	M: int
	qp: QP

	L: int

	alpha: float
	Ec: float
	n0: float

	turn_off_finite_size_effects: bool

	@property
	def n(self):
		return self.qp.n + ( 2 * self.M ) 

	@property
	def unblocked(self):
		return self.L - self.qp.n

	@property
	def occupiedLevels(self):
		return self.M + self.qp.n

	def energy(self):
		U = self.unblocked
		sc_energy = (-2 * self.alpha * ( U - self.M ) * self.M / self.L)
		qp_energy = self.alpha * (self.L - U) / self.L
		charging_energy = self.Ec * (self.n - self.n0)**2
		if self.turn_off_finite_size_effects:
			"""
			Only add alpha for each quasiparticle. No dependence on L!
			"""
			return (self.alpha * self.qp.n) + charging_energy
		else:
			return sc_energy + qp_energy + charging_energy

	def __eq__(self, other):
		condition = self.M == other.M and self.qp== other.qp
		return condition		

class BASIS_STATE:
	"""
	A basis state is a product state of the impurity level and the two SCs.
	"""
	def __init__(self, qp_imp, Ml, qp_L, Mr, qp_R, p):
		self.imp = IMP(qp_imp, p.U, p.epsimp)
		self.L = SC_BATH(Ml, qp_L, p.LL, p.alpha_L, p.Ec_L, p.n0_L, p.turn_off_SC_finite_size_effects)
		self.R = SC_BATH(Mr, qp_R, p.LL, p.alpha_R, p.Ec_R, p.n0_R, p.turn_off_SC_finite_size_effects)
		self.QP_state = QP_STATE(qp_imp, qp_L, qp_R)

		self.nimp = self.imp.n
		self.nL = self.L.n
		self.nR = self.R.n
		self.dM = self.L.M - self.R.M

	def energy(self):
		return self.imp.energy() + self.L.energy() + self.R.energy()

	@property
	def n(self):
		return self.imp.n + self.L.n + self.R.n

	@property
	def qp_bit_integer(self):
		"""
		Returns the bitstring corresponding to the qp state. 
		"""
		tot = 0
		tot += self.imp.state.bit_integer 	* 2**4
		tot += self.L.qp.bit_integer 	* 2**2
		tot += self.R.qp.bit_integer 	* 2**0
		return tot

	def __str__(self):
		return f"|{self.imp.state}, {self.L.M}, {self.L.qp}, {self.R.M}, {self.R.qp}>"

	def __eq__(self, other):
		condition = self.L == other.L and self.R == other.R and self.imp == other.imp
		return condition

class STATE:
	"""
	A state is a superposition of basis states with given amplitudes.
	"""

	def __init__(self, type_index, *amplitudes_and_basis_states):
		"""
		The input should be any number of tuples (amplitude, BASIS_STATE).
		type_index indentifies QP configuration this state has. It corresponds to the order in
		which the states are given in the mathematica notebook and is relevant for matrix parsing.
		"""
		self.type_index = type_index
		self.amplitudes = [amp for amp, st in amplitudes_and_basis_states]
		self.basis_states = [st for amp, st in amplitudes_and_basis_states]
		self.amplitudes_and_basis_states = amplitudes_and_basis_states
		self.check_if_normalized()

		self.n = self.basis_states[0].n
		self.mL = self.basis_states[0].L.M #All basis states have the same mL so this is OK.
		self.mR = self.basis_states[0].R.M #All basis states have the same mR so this is OK.
		self.dM = self.mL - self.mR
		self.nqp = self.n - 2 * (self.mL + self.mR)	

	def check_if_normalized(self):
		tot = sum(a**2 for a in self.amplitudes)
		if abs(tot - 1) > 1e-10:
			raise Exception(f"State not normalized! <psi|psi> = {tot}")	

	@property
	def nqp_no_imp(self):
		tot = 0
		for i in range(len(self.amplitudes)):
			tot += abs(self.amplitudes[i])**2  * ( self.basis_states[i].L.qp.n + self.basis_states[i].R.qp.n)
		return tot	

	@property
	def QP_state(self):
		"""	
		Saves only the state of the QPs, without the number of pairs.
		"""
		qps = []
		for amp, bstate in self.amplitudes_and_basis_states:
			qps.append( (amp, (bstate.QP_state)) )
		return qps	

	def energy(self):
		E = 0
		totAmps = np.vdot(self.amplitudes, self.amplitudes)
		for i in range(len(self.amplitudes)):
			E += abs(self.amplitudes[i])**2  * self.basis_states[i].energy()
		return E/totAmps

	@property	
	def nimp(self):
		n = 0
		for i, bs in enumerate(self.basis_states):
			n += self.amplitudes[i]**2 * bs.imp.n
		return n

	@property
	def nL(self):
		n = 0
		for i, bs in enumerate(self.basis_states):
			n += self.amplitudes[i]**2 * bs.L.n
		return n

	@property
	def nR(self):
		n = 0
		for i, bs in enumerate(self.basis_states):
			n += self.amplitudes[i]**2 * bs.R.n
		return n

	def __str__(self):
		s = ""
		for i in range(len(self.amplitudes)-1):
			s += f"{self.amplitudes[i]} * {self.basis_states[i]}  "
			if self.amplitudes[i+1] > 0:
				s += "+"
		s += f"{self.amplitudes[-1]} * {self.basis_states[-1]}"
		return s	

class PHI_STATE:
	"""
	A FT of STATE. Quantum numbers are phi and QP_state. QP_state_list is the same as the one from class STATE.
	"""

	def __init__(self, phi, QP_state_list):
		self.phi = phi
		self.QP_state_list = QP_state_list #this is a list of [ (amp, QP_STATE), ... ]

	@property
	def nqp_SC(self):
		tot = 0
		for amp, QP_state in self.QP_state_list:
			nL, nR = QP_state.qp_L.n, QP_state.qp_R.n
			tot += abs(amp)**2 * (nL + nR)
		return tot	

	@property
	def nqp(self):
		tot = 0
		for amp, QP_state in self.QP_state_list:
			nimp, nL, nR = QP_state.qp_imp.n, QP_state.qp_L.n, QP_state.qp_R.n
			tot += abs(amp)**2 * (nimp + nL + nR)
		return tot	

	def __repr__(self):
		s=f"phi/pi = {round(self.phi/np.pi, 3)}; "

		for i in range(len(self.QP_state_list)-1):
			amp, basisQPstate = self.QP_state_list[i]

			s += f"{round(amp, 4)} * ({str(basisQPstate.qp_imp)}, {str(basisQPstate.qp_L)}, {str(basisQPstate.qp_R)}) "

			if self.QP_state_list[i+1][0] >= 0:
				s += "+ "

		lastAmp, lastQPs = self.QP_state_list[-1]
		s += f"{round(lastAmp, 4)} * ({str(lastQPs.qp_imp)}, {str(lastQPs.qp_L)}, {str(lastQPs.qp_R)})"
	
		return s

###################################################################################################
# UTILITY

def delta(x, y):
	return 1.0 if x == y else 0.0
