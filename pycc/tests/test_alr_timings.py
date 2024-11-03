import psi4
import pytest
import sys
import os  # Import os for file handling
import numpy as np  # Ensure numpy is imported for array handling

sys.path.append("/Users/jattakumi/pycc/pycc/")
from data.molecules import *

num = ["1", "2", "3", "4", "5", "6", "7"]

# File to save results for all molecules
combined_diag_file = "asympolarizability_combined_diagonal.txt"
combined_timings_file = "asympolarizability_combined_timings.txt"

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
        'r_convergence': 1e-08
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

    # Local calculations
    lcc = ccwfn(rhf_wfn, local='PNO++', local_mos='BOYS', local_cutoff=1e-07, filter=False)
    lecc = lcc.lccwfn.solve_lcc(e_conv, r_conv)
    lhbar = cchbar(lcc)
    lcclambda = cclambda(lcc, lhbar)
    llecc = lcclambda.solve_llambda(e_conv, r_conv)
    ldensity = ccdensity(lcc, lcclambda)

    lresp = ccresponse(ldensity)

    omega1 = 0.0428
    X_1 = {}
    X_2 = {}
    Y_1 = {}
    Y_2 = {}

    # Calculate local responses
    for axis in range(0, 3):
        string = "MU_" + lresp.cart[axis]
        A = lresp.lpertbar[string]
        X_2[string] = lresp.local_solve_right(A, omega1, lhbar, e_conv=1e-08, r_conv=1e-08)
        Y_2[string] = lresp.local_solve_left(A, omega1, e_conv=1e-08, r_conv=1e-08)
        X_1[string] = lresp.local_solve_right(A, -omega1, lhbar, e_conv=1e-08, r_conv=1e-08)
        Y_1[string] = lresp.local_solve_left(A, -omega1, e_conv=1e-08, r_conv=1e-08)

    # Grabbing X, Y and declaring the matrix space for linear response
    lpolar_AB = np.zeros((3, 3))

    for a in range(0, 3):
        string_a = "MU_" + lresp.cart[a]
        for b in range(0, 3):
            string_b = "MU_" + lresp.cart[b]
            Y1_B, Y2_B, _ = Y_2[string_b]
            X1_B, X2_B, _ = X_2[string_b]
            lpolar_AB[a, b] = lresp.local_linresp(a, string_a, X1_B, Y1_B, X2_B, Y2_B)


    # Calculate average polarizability
    lpolar_AB_avg = np.average([lpolar_AB[0, 0], lpolar_AB[1, 1], lpolar_AB[2, 2]])

    # Save diagonal components and average to a single combined file
    with open(combined_diag_file, 'a') as f_diag:
        diagonal_components = [lpolar_AB[0, 0], lpolar_AB[1, 1], lpolar_AB[2, 2]]
        f_diag.write(f"{geom}\t" + '\t'.join(map(str, diagonal_components)) + f"\t{lpolar_AB_avg}\n")

    # Gather timings
    timings = []
    timing_sources = [
        "lX1_t", "lX2_t", "lY1_t", "lY2_t",
        "lpseudoresponse_t",
        "llcx_t", "quad_terms"]

    for source in timing_sources:
        timings.append(getattr(lresp, source))

    # Save timings to the combined timings file
    with open(combined_timings_file, 'a') as f_timings:
        for time_index in range(len(timings)):
            f_timings.write(f"{geom}\t{timing_sources[time_index]}\t{timings[time_index]}\n")

# Clean up
del lcc, lhbar, lcclambda, ldensity, lresp
