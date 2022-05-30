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

	calc_occupancies: bool = True
	calc_dMs: bool = True
	calc_phase: bool = True
	calc_nqp: bool = True
	
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
		Given the name of a parameter (string!), sets its default value if it is currently nan.
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

	def __repr__(self):
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

	@classmethod
	def export_to(cls, namespace):
	    namespace.update(cls.__members__)

	def __str__(self):
		"""
		This makes it so printing the enum gives ZERO instead of QP.ZERO
		"""
		return self.name 

QP.export_to(globals()) #this exports all enums into global namespace, so they can be called by eg. ZERO instead of QP.ZERO


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

	def n(self):
		return self.state.n

	def energy(self):
		Upart = self.U if self.state == UPDN else 0
		return self.epsimp * self.state.n + Upart

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

	def n(self):
		return self.qp.n + ( 2 * self.M ) 

	def unblocked(self):
		return self.L - self.qp.n

	def occupiedLevels(self):
		return self.M + self.qp.n

	def energy(self):
		U = self.unblocked()
		sc_energy = (-2 * self.alpha * ( U - self.M ) * self.M / self.L)
		qp_energy = self.alpha * (self.L - U) / self.L
		charging_energy = self.Ec * (self.n() - self.n0)**2
		if self.turn_off_finite_size_effects:
			"""
			Only add alpha for each quasiparticle. No dependence on L!
			"""
			return (self.alpha * self.qp.n) + charging_energy
		else:
			return sc_energy + qp_energy + charging_energy

class BASIS_STATE:
	"""
	A basis state is a product state of the impurity level and the two SCs.
	"""
	def __init__(self, qp_imp, Ml, qp_L, Mr, qp_R, p):
		self.imp = IMP(qp_imp, p.U, p.epsimp)
		self.L = SC_BATH(Ml, qp_L, p.LL, p.alpha_L, p.Ec_L, p.n0_L, p.turn_off_SC_finite_size_effects)
		self.R = SC_BATH(Mr, qp_R, p.LL, p.alpha_R, p.Ec_R, p.n0_R, p.turn_off_SC_finite_size_effects)
		self.QP_state = (qp_imp, qp_L, qp_R)

	def energy(self):
		return self.imp.energy() + self.L.energy() + self.R.energy()

	def n(self):
		return self.imp.n() + self.L.n() + self.R.n()

	def __repr__(self):
		return f"|{self.imp.state}, {self.L.M}, {self.L.qp}, {self.R.M}, {self.R.qp}>"



class PHI_BASIS_STATE:
	"""
	A basis state with a well defined phase difference between the channels. 
	Quantum numbers:
	qp_imp, phi, qp_L, qp_R
	"""
	def __init__(self, qp_imp, qp_L, qp_R, p):
		self.imp = IMP(qp_imp, p.U, p.epsimp)
		self.qp_L = qp_L
		self.qp_R = qp_R


class STATE:
	"""
	A state is a superposition of basis states with given amplitudes.
	"""

	def __init__(self, *amplitudes_and_basis_states):
		"""
		The input should be any number of tuples (amplitude, BASIS_STATE).
		"""
		self.amplitudes = [amp for amp, st in amplitudes_and_basis_states]
		self.basis_states = [st for amp, st in amplitudes_and_basis_states]
		self.amplitudes_and_basis_states = amplitudes_and_basis_states
		self.check_if_normalized()

	@property
	def mL(self):
		"""
		All basis states have the same mL so this is OK.
		"""
		return self.basis_states[0].L.M

	@property
	def mR(self):
		"""
		All basis states have the same mR so this is OK.
		"""
		return self.basis_states[0].R.M

	@property
	def dM(self):
		"""
		This is also a good quantum number!
		"""
		return self.mL - self.mR

	@property
	def nqp(self):
		return self.n - 2 * (self.mL + self.mR)	

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
		
	def check_if_normalized(self):
		tot = sum(a**2 for a in self.amplitudes)
		if abs(tot - 1) > 1e-10:
			raise Exception(f"State not normalized! <psi|psi> = {tot}")

	@property
	def n(self):
		nn = self.basis_states[0].n()
		for bs in self.basis_states:
			if bs.n() != nn:
				raise Exception("State does not have well defined charge!")
		return nn		

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
			n += self.amplitudes[i]**2 * bs.imp.n()
		return n

	@property
	def nL(self):
		n = 0
		for i, bs in enumerate(self.basis_states):
			n += self.amplitudes[i]**2 * bs.L.n()
		return n

	@property
	def nR(self):
		n = 0
		for i, bs in enumerate(self.basis_states):
			n += self.amplitudes[i]**2 * bs.R.n()
		return n

	def __repr__(self):
		s = ""
		for i in range(len(self.amplitudes)-1):
			s += f"{self.amplitudes[i]} * {self.basis_states[i]}  "
			if self.amplitudes[i+1] > 0:
				s += "+"
		s += f"{self.amplitudes[-1]} * {self.basis_states[-1]}"
		return s	

###################################################################################################
# UTILITY

def delta(x, y):
	return 1 if x == y else 0
