"""
Test CCSD linear response functions.
"""

# Import package, test suite, and other packages as needed
import sys
import psi4
import numpy as np
import pytest
import pycc
from ..data.molecules import *

def test_sym_linresp():
    h2_3 = """
    H 0.000000 0.000000 0.000000
    H 0.750000 0.000000 0.000000
    H 0.000000 1.500000 0.000000
    H 0.375000 1.500000 -0.649520
    H 0.000000 3.000000 0.000000
    H -0.375000 3.000000 -0.649520
    symmetry c1
    """

    psi4.core.clean_options()
    psi4.set_memory('2 GiB')
    psi4.core.set_output_file('output.dat', False)
    psi4.set_options({'basis': 'cc-pVDZ',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'true',
                      'e_convergence': 1e-10,
                      'd_convergence': 1e-10,
                      'r_convergence': 1e-10})

    mol = psi4.geometry(moldict["(H2)_2"])
    rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

    e_conv = 1e-10
    r_conv = 1e-10

    cc = pycc.ccwfn(rhf_wfn, local = 'PNO++', local_mos = 'BOYS', local_cutoff = 1e-7, filter=True)
    ecc = cc.solve_cc(e_conv, r_conv)
    hbar = pycc.cchbar(cc)
    cclambda = pycc.cclambda(cc, hbar)
    lecc = cclambda.solve_lambda(e_conv, r_conv)
    density = pycc.ccdensity(cc, cclambda)

    resp = pycc.ccresponse(density)

    omega1 = 0.0656

    # Creating dictionaries
    # X_A = {}
    # X_B = {}
    X_1 = {}
    X_2 = {}
    Y_1 = {}
    Y_2 = {}

    for axis in range(0, 3):
        string = "MU_" + resp.cart[axis]
        A = resp.pertbar[string]
        # X_A[string] = resp.solve_right(A, omega1, e_conv = 1e-08, r_conv = 1e-08)
        # X_B[string] = resp.solve_right(A, -omega1, e_conv = 1e-08, r_conv = 1e-08)
        X_2[string] = resp.solve_right(A, omega1, e_conv = 1e-08, r_conv = 1e-08)
        Y_2[string] = resp.solve_left(A, omega1, e_conv = 1e-08, r_conv = 1e-08)
        X_1[string] = resp.solve_right(A, -omega1, e_conv = 1e-08, r_conv = 1e-08)
        Y_1[string] = resp.solve_left(A, -omega1, e_conv = 1e-08, r_conv = 1e-08)

    # Grabbing X, Y and declaring the matrix space for LR
    polar_AB = np.zeros((3,3))
    polar_AB_sym = np.zeros((3, 3))

    for a in range(0, 3):
        string_a = "MU_" + resp.cart[a]
        X1_A, X2_A, _ = X_A[string_a]
        for b in range(0, 3):
            string_b = "MU_" + resp.cart[b]
            X_1B, X_2B, _ = X_B[string_b]
            polar_AB_sym[a,b] = resp.sym_linresp(string_a, string_b, X1_A, X2_A, X_1B, X_2B)
            Y1_B, Y2_B, _ = Y_2[string_b]
            X1_B, X2_B, _ = X_2[string_b]
            polar_AB[a, b] = resp.linresp_asym(string_a, X1_B, X2_B, Y1_B, Y2_B)

    print(f"Dynamic Polarizability Tensor @ w = {omega1} a.u.:")
    print(polar_AB)
    print("Average Dynamic Polarizability:")
    polar_AB_avg = np.average([polar_AB[0,0], polar_AB[1,1], polar_AB[2,2]])
    print(polar_AB_avg)

    # print('Symmetric stuff')
    # print(f"Dynamic Polarizability Tensor @ w = {omega1} a.u.:")
    # print(polar_AB_sym)
    # print("Average Dynamic Polarizability:")
    # polar_AB_avg_sym = np.average([polar_AB_sym[0, 0], polar_AB_sym[1, 1], polar_AB_sym[2, 2]])
    # print(polar_AB_avg_sym)
    #
    # # Validating from psi4
    # polar_xx = 9.932240347101651
    # polar_yy = 13.446487681337629
    # polar_zz = 11.344346098120035
    # polar_avg = 11.574358042186
    #
    # assert(abs(polar_AB[0,0] - polar_xx) < 1e-7)
    # assert(abs(polar_AB[1,1] - polar_yy) < 1e-7)
    # assert(abs(polar_AB[2,2] - polar_zz) < 1e-7)
    # assert(abs(polar_AB_avg - polar_avg) < 1e-7)

    lcc = pycc.ccwfn(rhf_wfn, local = 'PNO++', local_mos = 'BOYS', local_cutoff = 1e-7, filter = False)
    lecc = lcc.lccwfn.solve_lcc(e_conv, r_conv)
    lhbar = pycc.cchbar(lcc)
    lcclambda = pycc.cclambda(lcc, lhbar)
    llecc = lcclambda.solve_llambda(e_conv, r_conv)
    ldensity = pycc.ccdensity(lcc, lcclambda)

    lresp = pycc.ccresponse(ldensity)

    omega1 = 0.0656


    # Creating dictionaries
    # X_A = {}
    # X_B = {}
    X_1 = {}
    X_2 = {}
    Y_1 = {}
    Y_2 = {}

    for axis in range(0, 3):
        string = "MU_" + lresp.cart[axis]
        A = lresp.lpertbar[string]
        # X_A[string] = lresp.local_solve_right(A, omega1, lhbar,  e_conv = 1e-08, r_conv = 1e-08)
        # X_B[string] = lresp.local_solve_right(A, -omega1, lhbar,  e_conv = 1e-08, r_conv = 1e-08)
        X_2[string] = lresp.local_solve_right(A, omega1, lhbar, e_conv=1e-08, r_conv=1e-08)
        Y_2[string] = lresp.local_solve_left(A, omega1, e_conv=1e-08, r_conv=1e-08)
        X_1[string] = lresp.local_solve_right(A, -omega1, lhbar, e_conv=1e-08, r_conv=1e-08)
        Y_1[string] = lresp.local_solve_left(A, -omega1, e_conv=1e-08, r_conv=1e-08)
    # Grabbing X, Y and declaring the matrix space for LR
    lpolar_AB = np.zeros((3, 3))

    for a in range(0, 3):
        string_a = "MU_" + lresp.cart[a]
        # X1_A, X2_A, _ = X_A[string_a]
        for b in range(0, 3):
            string_b = "MU_" + lresp.cart[b]
            # X_1B, X_2B, _ = X_B[string_b]
            # lpolar_AB[a, b] = lresp.lsym_linresp(string_a, string_b, X1_A, X2_A, X_1B, X_2B)
            Y1_B, Y2_B, _ = Y_2[string_b]
            X1_B, X2_B, _ = X_2[string_b]
            lpolar_AB[a, b] = lresp.local_linresp(a, string_a, X1_B, Y1_B, X2_B, Y2_B)

    print(f"Dynamic Polarizability Tensor @ w = {omega1} a.u.:")
    print(lpolar_AB)
    print("Average Dynamic Polarizability:")
    lpolar_AB_avg = np.average([lpolar_AB[0, 0], lpolar_AB[1, 1], lpolar_AB[2, 2]])
    print(lpolar_AB_avg)

    # # Validating from psi4
    # polar_xx = 9.932240347101651
    # polar_yy = 13.446487681337629
    # polar_zz = 11.344346098120035
    # polar_avg = 11.574358042186
    #
    # assert (abs(lpolar_AB[0, 0] - polar_xx) < 1e-7)
    # assert (abs(lpolar_AB[1, 1] - polar_yy) < 1e-7)
    # assert (abs(lpolar_AB[2, 2] - polar_zz) < 1e-7)
    # assert (abs(lpolar_AB_avg - polar_avg) < 1e-7)


