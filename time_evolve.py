#!/usr/bin/env python3

from helper import *
from observables_calculation import *
from matrix import generate_total_matrix, reorder_matrix_dM, reorder_matrix_phi, fourier_transform_matrix
import sys
from math import pi
import time
from numpy.linalg import eigh
import h5py
from joblib import Parallel, delayed

###################################################################################################
#input: 
# - results from the exact diagonalisation - read them from the hdf5 file 
# - time evolution parameters
# - which eigenstate to evolve!

# load the eigenvalues and eigenvectors

# load the matrix of the time dependent operator and transform it into the same basis as the eigenvectors

# set up the time dependent amplitude function of the operator, a(t)

# set up and solve the set of time dependent equations for the evolution