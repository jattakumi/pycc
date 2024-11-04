import psi4
import pytest
import sys
import os  # Import os for file handling
import numpy as np  # Ensure numpy is imported for array handling

sys.path.append("/Users/jattakumi/pycc/pycc/")
from data.molecules import *

num = ["1", "2", "3", "4", "5", "6", "7"]

# File to save results for all molecules
combined_diag_file = "conv_polarizability_combined_diagonal.txt"
combined_timings_file = "conv_polarizability_combined_timings.txt"

# Write headers for each combined file once
if not os.path.exists(combined_diag_file):
    with open(combined_diag_file, 'w') as f_diag:
        f_diag.write("Molecule\tDiagonal Component 1 (a.u.)\tDiagonal Component 2 (a.u.)\tDiagonal Component 3 (a.u.)\tAverage Dynamic Polarizability (a.u.)\n")

if not os.path.exists(combined_timings_file):
    with open(combined_timings_file, 'w') as f_timings:
        f_timings.write("Molecule\tTiming Source\tTiming (seconds)\n")  # Write header for timings

for n in num:
    geom = "(H2O)_" + str(n)
    psi4.core.clean_options()
    psi4.set_memory('8 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({
        'basis': 'aug-cc-pvdz',
        'scf_type': 'pk',
        'freeze_core': 'true',
        'e_convergence': 1e-08,
        'd_convergence': 1e-08,
        'r_convergence': 1e-08,
        'diis': False
    })

    mol = psi4.geometry(moldict[geom])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    from pycc.ccwfn import *
    from lccwfn import *
    from pycc.cchbar import *
    from pycc.cclambda import *
    from pycc.ccdensity import *
    from pycc.ccresponse import *

    e_conv = 1e-08
    r_conv = 1e-08

    # Conventional calculations
    cc = ccwfn(rhf_wfn)
    ecc = cc.solve_cc(e_conv, r_conv)
    hbar = cchbar(cc)
    cclambda = cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    density = ccdensity(cc, cclambda)

    resp = ccresponse(density)

    omega1 = 0.0428
    X_A = {}
    X_B = {}

    # Calculate local responses
    for axis in range(0, 3):
      string = "MU_" + resp.cart[axis]
      A = resp.pertbar[string]
      X_A[string] = resp.solve_right(A, omega1, e_conv=e_conv, r_conv=r_conv)
      X_B[string] = resp.solve_right(A, -omega1, e_conv=e_conv, r_conv=r_conv)

    # Grabbing X, Y and declaring the matrix space for linear response
    polar_AB = np.zeros((3, 3))

    for a in range(0, 3):
      string_a = "MU_" + resp.cart[a]
      X1_A, X2_A, _ = X_A[string_a]
      for b in range(0, 3):
        string_b = "MU_" + resp.cart[b]
        X_1B, X_2B, _ = X_B[string_b]
        polar_AB[a, b] = resp.sym_linresp(string_a, string_b, X1_A, X2_A, X_1B, X_2B)

    # Calculate average polarizability
    polar_AB_avg = np.average([polar_AB[0, 0], polar_AB[1, 1], polar_AB[2, 2]])

    # Save diagonal components and average to a single combined file
    with open(combined_diag_file, 'a') as f_diag:
      diagonal_components = [polar_AB[0, 0], polar_AB[1, 1], polar_AB[2, 2]]
      f_diag.write(f"{geom}\t" + '\t'.join(map(str, diagonal_components)) + f"\t{polar_AB_avg}\n")

    # Gather timings
    timings = []
    timing_sources = [
      "lX1_t", "lX2_t", "lpseudoresponse_t",
      "lLCX_t", "lLHX1Y1_t", "lLHX2Y2_t",
      "lLHX1Y2_t", "lsym_lr_t"
    ]

    for source in timing_sources:
      timings.append(getattr(resp, source))

    # Save timings to the combined timings file
    with open(combined_timings_file, 'a') as f_timings:
      for time_index in range(len(timings)):
        f_timings.write(f"{geom}\t{timing_sources[time_index]}\t{timings[time_index]}\n")

# Clean up
del cc, hbar, cclambda, density, resp
