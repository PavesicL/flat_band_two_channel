# Flat band two channel
Code for the flat band SC - QD - SC problem.

# Requirements
Python3 with standard libraries contained in Anaconda, and my_second_quantization.
Make sure to clone recursively: git clone --recursive [repo]

# Usage
Run calc.py inputFile 

# Input file
The input file should contain a set of parameters. These define the model parameters and the observables to compute.
See an example: inputFile.

# Parameters 
A list of parameters and their default values.

## Physical parameters

* N - system size, includes both leads and the QD. Required!

### QD level
* U = 0 - on-site interaction on the QD level
* epsimp = -U/2 - energy of the QD level
* Ez_imp = 0 - Zeeman splitting of the QD level

### Leads
* alpha = 0 - dimensionless pairing in the leads. Positive for superconductivity.
* Ec = 0 - charging energy in the leads.
* n0 = (N-1)/2 - favoured filling of the leads.
* Ez_sc = 0 - Zeeman splitting in the leads.

These parameters can be separately specified for each lead by adding _L or _R. They default to values of the corresponding general lead parameters.
* alpha_L = alpha - dimensionless pairing in the left lead. Positive for superconductivity.
* alpha_R = alpha - dimensionless pairing in the right lead. Positive for superconductivity.
* Ec_L = Ec - charging energy in the left lead.
* Ec_R = Ec - charging energy in the right lead.
* n0_L = n0 - favoured filling of the left lead.
* n0_R = n0 - favoured filling of the right lead.
* Ez_L = Ez_sc - Zeeman splitting in the left lead.
* Ez_R = Ez_sc - Zeeman splitting in the right lead.

### Hopping

* v = 0 - standard spin conserving QD-lead hopping.
* v_L = v - QD-L hopping
* v_R = v - QD-R hoppin

* tpair = 0 - pair hopping between the two SCs.
* phiext = 0 - externally enforced phase in the auxiliary Josephson junction

#### spin-orbit coupling
* tsc = 0 - hopping directly between the f_L and f_R levels
* tspin = 0 - spin-flipping hopping between the QD and the leads
* tspin_L = tspin - spin-flipping QD-L hopping
* tspin_R = tspin - spin-flipping QD-R hopping

### Calculation parameters
* nref = N, half-filling. If U, Ec_L or Ec_R are set, sum nu + n0_L + n0_R and convert to integer. The reference value of particle number sector the calculations are centered around.
* nrange = 0 - the calculations are performed in all sectors with charge within n = nref +/- nrange.
* verbose = False - turns on printing of some warning messages (not particularly strictly implemented, OCT 2023)
* doublet_both_Sz = False - whether to expand the doublet basis with Sz=-1/2 states. Needed when spin-orbit interaction is on.
* save_all_states = False - whether to save all eigenvectors of the Hamiltonian.
* num_states_to_save = 10 - number of eigenvectors with the lowest energies to save in each sector
* parallel = False - turn on parallel calculations for each sector

### Approximations
* turn_off_all_finite_size_effects = False - enables all options below
* turn_off_SC_finite_size_effects = turn_off_all_finite_size_effects - no finite size effects in the leads. The Hamiltonian is given by alpha * number of quasiparticles in f-orbitals
* turn_off_hopping_finite_size_effects = turn_off_all_finite_size_effects - all hoppings are taken in the limit of half filling and large L - matrices with no_finite_size are used
* use_all_states = turn_off_all_finite_size_effects - does not throw away unphysical states, eg. a state with mL = L and a quasiparticle in the same channel is kept
* restrict_basis_to_make_periodic = turn_off_all_finite_size_effects - throws away the states that form the blocks which are not full (14x14) but smaller. Use with add_periodic_hopping_blocks for (almost) perfect periodization.
* add_periodic_hopping_blocks = turn_off_all_finite_size_effects - adds hopping blocks between the first and last dM states. Akin to periodic boundary conditions.

### Property calculation parameters

* save_matrix = False - saves the Hamiltonian matrix for each sector in matrix_n{number of particles}
* reorder_matrix_dM = True - reorders the basis of the Hamiltonian in the order of ascending dM before saving
* phase_fourier_transform = False - performs the fourier transform into the phi basis on the Hamiltonian matrix

* print_states_dM = False - print amplitudes in the dM basis for all saved states
* print_states_phi = False - print amplitudes in the phi basis for all saved states. Requires phase_fourier_transform.
* number_of_states_to_print = 0 - number of states to print
* print_states_precision = 0.01 - prints only basis contributions with amplitude larger than this number

* print_energies = 10 - the number of energies to print. All are saved.
* print_precision = 5 - the number of digits to print for energies

* calc_occupancies = True - calculate and save occupation of each part of the system. Saved in n/Sz/i/occupancies/{nimp, nL, nR}.
* calc_tot_Sz = False - calculate the total Sz. Saved in n/Sz/i/tot_Sz.
* calc_dMs = False - calculate expected value and variance of dM. Saved in n/Sz/i/{dM, dM2}.
* calc_QP_phase = False - one way to gauge the expected value of phase by calculating the expected values of f^dag_L,down f^dag_L,up f_R,up f_R,down. The ampltidue and phase of the complex number are saved in n/Sz/i/{QP_phi, QP_phi_size}. Idea from [Gobert, 2004](10.1140/epjb/e2004-00145-6).
* calc_abs_phase = True - gauges the expected value of phase by reflecting all phi intro the first quadrants (from 0 to phi) and averaging over phi in the phi basis. The average and variance are saved to n/Sz/i/{abs_phi, abs_phi2}.
* calc_sin_cos_phi = False - calculates the expected value of c = amplitude * cos(phi) and s = amplitude * sin(phi), and np.arctan(s,c). Saved in n/Sz/i/{cos_phi, sin_phi, atan2_phi}.
* save_dM_amplitudes = False - saves amplitudes in the dM basis. The dMs and corresponding amplitudes and average nqps are saved in n/Sz/i/dM_amplitudes/{dMs, nqps, amplitudes}.
* save_phi_amplitudes = False - saves amplitudes in the phi basis. The dMs and corresponding amplitudes and average nqps are saved in n/Sz/i/dM_amplitudes/{dMs, nqps, amplitudes}.
* calc_wigner_distribution = False - saves the wigner distributions. dMs, phis and corresponding values saved in n/Sz/i/wigner_distribution/{dMs, phis, vals}.
* calc_nqp = True - calculate the expected value of the number of quasiparticles, ie. occupation of the f_L, f_R and QD orbitals. Saved in n/Sz/i/nqps.
* calc_imp_spin_correlations = False - compute spin-spin correlations between the active orbitals. Saved in n/Sz/i/SS/{impimp, LL, RR, impL, impR, LR}.
* calc_parity = False - calculates the expected value of the spatial inversion parity operator by evaluating the overlap of the state with itself after the transforming all strings of operators by: cdag_L cdag_R -> cdag_R cdag_L. Saved in n/Sz/i/parity.
* calc_dEs = False - prints (does not save!) the energy difference of each state to the previous one. 

# Program organisation

The model contains two superconducting islands (SIs) and a quantum dot (QD) embedded between them. In the flat-band approximation, the SIs are described by one active orbital f, and a number of Cooper pairs they contain, m. For conserved total charge n, it is enough to track the difference in the number of pairs, $\Delta m$. 
The basis is a product of all allowed $\Delta m$ and all $[d,f_L,f_R]$ states for a given spin ($S=0$, $S=1/2$). States can be written as $\vert \Delta m, [d,f_L,f_R]\rangle$.


The algebra of applying second quantization operators to the kets is defined in matrices/0_generate_matrices.nb, a Wolfram Mathematica notebook. It generates hopping matrix elements for a generic $[(m_L,m_R),(n_L,n_R)]$ matrix. These are saved in the folder matrices/. 
Code in parse_matrices.py then parses the symbolic expressions for the matrix elements. The complete basis and the Hamiltonian are generated from these expressions in matrix.py.
The parsing of parameters and classes defining the kets, QD and SIs are defined in helper.py. 
Observables calculations are defined in observables_calculation.py.
The Hamiltonian is diagonalised in calc.py.





