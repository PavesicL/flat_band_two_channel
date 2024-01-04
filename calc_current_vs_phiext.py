#!/usr/bin/env python3

import sys
import helper
import matrix
import numpy as np
from numpy.linalg import eigh
from joblib import Parallel, delayed
import h5py

###################################################################################################

def main():

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} inputFile dphi")
        exit()
    inputFile = sys.argv[1]
    dphiext = float(sys.argv[2]) # this is the difference used to compute the numerical differential
    h5file = h5py.File("current_overlaps.h5", "w")

    # now generate three instances of params, with phiext, phiext+dphiext, phiext-dphiext
    p = helper.parse_params(inputFile)
    pplus = helper.parse_params(inputFile)
    pminus = helper.parse_params(inputFile)
    pplus.phiext += dphiext
    pminus.phiext -= dphiext

    print(p)

    print(f"Computing in subspaces: {p.subspace_list}")

    #compute for each subspace
    num_processes = len(p.subspace_list) if p.parallel else 1
    results = Parallel(n_jobs = num_processes)(delayed(get_derivatives)(n, p, pplus, pminus, dphiext) for n in p.subspace_list)

    print("Done. Now writing...")

    print("Overlaps:")
    for ndx, n in enumerate(p.subspace_list):
        print(f"\nSubspace n={n}")
        for i in range(p.number_of_overlaps):
            for j in range(p.number_of_overlaps):
                print( f"({i},{j}): {results[ndx][i,j]}" )

    # write the results to the hdf5 file
    for i, n in enumerate(p.subspace_list):
        h5file.create_dataset( f"/{n}", data=results[i])


def get_derivatives(n, p, pplus, pminus, dphiext):
    """
	Computes the derivative according to the formula:
        [ E_a(phiext) - E_b(phiext) ] * d<a|/dphiext |b>
	The derivative is approximated by the difference ( a(phiext+dphiext) - a(phiext-dphiext) )/ 2 dphiext

	n - number of particles, defining the sector
	p - parameter dictionary
	"""

    res = np.zeros( shape = (p.number_of_overlaps, p.number_of_overlaps), dtype="complex128" )
    for i in range(p.number_of_overlaps):
        for j in range(p.number_of_overlaps):

            val0, vec0 = get_valvec_for_one_case(n, p)
            _, vecp = get_valvec_for_one_case(n, pplus)
            _, vecm = get_valvec_for_one_case(n, pminus)

            energy_diff = val0[i] - val0[j]
            vector_diff = ( vecp[i] - vecm[i] ) / (2 * dphiext)
            overlap = np.dot( np.conjugate(vector_diff), vec0[j] )

            res[i,j] += energy_diff * overlap

    return res


def get_valvec_for_one_case(n, p):
    """
    Solves the problem in subspace n for phiext = p.phiext + dphiext.
    """

    # generate the subspace name
    if  n%2==0:
        subspaceName = "singlet"
    else:
        if p.doublet_both_Sz:
            subspaceName = "doublet_both_Sz"
        else:
            subspaceName = "doublet"

    # generate the hamiltonian matrix
    mat, bas = matrix.generate_total_matrix(subspaceName, n, p)

    # diagonalize
    val, vec = eigh(mat)
    vec = vec.T #eigh() returns eigenvectors as columns of a matrix, but we want vec[i] to be i-th eigenvector.

    return val, vec

###################################################################################################

if __name__ == "__main__":
	main()