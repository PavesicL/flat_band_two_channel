#!/usr/bin/env python3

import re
import sympy
from sympy import I
from sympy.parsing.sympy_parser import parse_expr
from sympy.functions.special.tensor_functions import KroneckerDelta
###################################################################################################
	
def parse_hopping_matrix(which):
	"""
	Parse the hopping matrix elements from files doublet_mat and singlet_mat. They are for general mL, mR and nL, nR. 
	Rewrite each element first into a sympy symbolic expression and then into a lambda function.
	The output is a matrix of lambda functions, each giving a general hopping matrix element.

	The matrix is obtained from Mathematica with: Export["singlet_mat.dat", mat]. Delimiter is tab, \t.
	"""
	mat = []
	strMat = [] # here matrix elements are saved as strings. Not used anywhere as of now, but useful for debugging.
	mL, mR, nL, nR, vL, vR, tsc, tspinL, tspinR, l = sympy.symbols("mL mR nL nR vL vR tsc tspinL, tspinR l")

	with open(which, "r") as f:
		for i, line in enumerate(f):
			mat.append([])
			strMat.append([])

			matElements = line.split("\t")
			
			for elem in matElements:
				elem = elem.strip()

				# change the brackets in Sqrt and KroneckerDelta, [ ] -> ( )
				# sympy will now understand the expression 
				
				elem = re.sub(r"Sqrt\[(.*?)\]", r"sqrt(\1)", elem)
				elem = re.sub(r"KroneckerDelta\[(.*?)\]", r"KroneckerDelta(\1)", elem)

				a = parse_expr(elem)
				a = sympy.lambdify([mL, mR, nL, nR, vL, vR, tsc, tspinL, tspinR, l], a)
			
				# each element of this matrix is a function of the above parameters,
				# giving the matrix element for two states with general occupation mL, mR and nL, nR
				mat[i].append(a)
				strMat[i].append(elem)
	return mat, strMat

def parse_phi_matrix(which):
	"""
	Parses the matrix of the phi operator. These elements are all of type +/- 1/4 delta(mL, nL) delta(mR, nR).
	Only Kronecker delta has to be renamed then.
	"""

	mat = []
	strMat = []
	mL, mR, nL, nR = sympy.symbols("mL mR nL nR")

	with open(which, "r") as f:
		for i, line in enumerate(f):
			mat.append([])

			matElements = line.split("\t")
			for elem in matElements:
				elem = elem.strip()

				# change the brackets in Sqrt and KroneckerDelta, [ ] -> ( )
				# sympy will now understand the expression 
				
				elem = re.sub(r"\^", "**", elem)				
				elem = re.sub(r"KroneckerDelta\[(.*?)\]", r"KroneckerDelta(\1)", elem)

				a = parse_expr(elem)
				a = sympy.lambdify([mL, mR, nL, nR], a)
				
				# each element of this matrix is a function of the above parameters,
				# giving the matrix element for two states with general occupation mL, mR and nL, nR
				mat[i].append(a)

	return mat
