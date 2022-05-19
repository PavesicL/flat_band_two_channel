#!/usr/bin/env python3

import re
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.functions.special.tensor_functions import KroneckerDelta
###################################################################################################
	
def parse_matrix(which):
	"""
	Parse the hopping matrix elements from files doublet_mat and singlet_mat. They are for general mL, mR and nL, nR. 
	Rewrite each element first into a sympy symbolic expression and then into a lambda function.
	The output is a matrix of lambda functions, each giving a general hopping matrix element.

	The matrix is obtained from Mathematica with: Export["singlet_mat.dat", mat]. Delimiter is tab, \t.
	"""
	mat = []
	strMat = []
	mL, mR, nL, nR, vL, vR, l = sympy.symbols("mL mR nL nR vL vR l")

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
				a = sympy.lambdify([mL, mR, nL, nR, vL, vR, l], a)
				
				# each element of this matrix is a function of the above parameters,
				# giving the matrix element for two states with general occupation mL, mR and nL, nR
				mat[i].append(a)
				strMat[i].append(elem)

	return mat, strMat