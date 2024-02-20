"""
Test CCSD linear response functions.
"""
import numpy as np
# Import package, test suite, and other packages as needed
import psi4
from opt_einsum import contract

import pycc
#from ccwfn import ccwfn
#from cchbar import cchbar
#from cclambda import cclambda
#from ccdensity import ccdensity
#from ccresponse import ccresponse
import sys
sys.path.append("/Users/jattakumi/pycc/pycc/")
from data.molecules import *
geom = """
O
H 1 1.8084679
H 1 1.8084679 2 104.5
units bohr
symmetry c1 
no_reorient
"""

hf = """
F  0.000000000000   0.000000000000  0.000000000000
H  0.000000000000   0.000000000000  -1.732800000000
units bohr
no_reorient
symmetry c1
"""

psi4.set_memory('2 GiB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'STO-3G',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12
})
mol = psi4.geometry(moldict["H2O"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

e_conv = 1e-12
r_conv = 1e-12

cc = pycc.ccwfn(rhf_wfn, local = 'PNO', local_cutoff = 0, filter = True)
ecc = cc.solve_cc(e_conv, r_conv)
hbar = pycc.cchbar(cc)
cclambda = pycc.cclambda(cc, hbar)
lecc = cclambda.solve_lambda(e_conv, r_conv)
density = pycc.ccdensity(cc, cclambda)

resp = pycc.ccresponse(density)

omega1 = 0.0656
#omega2 = 0.0656


#A = resp.pertbar.Aoo

#resp.linresp(omega1)
# Creating dictionaries
# X_1 = X(-omega); X_2 = X(omega)
# Y_1 = Y(-omega); Y_2 = Y(omega)
# X_neg = X1(-omega) , X2(-omega)
# X_pos, Y_neg, Y_pos
X_1 = {}
X_2 = {}
Y_1 = {}
Y_2 = {}

string = "MU_X"

A = resp.pertbar[string]
    #print("Aoo",A)

X_2[string] = resp.solve_right(A, omega1, e_conv=1e-08, r_conv=1e-08, maxiter=10)
# print("first X", X_2[string][0])
#Y_2[string] = resp.solve_left(A, omega1)
#X_1[string] = resp.solve_right(A, -omega1, e_conv=1e-08, r_conv=1e-08, maxiter=10)
#Y_1[string] = resp.solve_left(A, -omega1)

#resp.polar(omega1)
# Grabbing X, Y and declaring the matrix space for LR
# polar_AB_pos = np.zeros((3,3))
# polar_AB_neg = np.zeros((3,3))
# polar_AB_aveg = np.zeros((3,3))
# print("second X", X_2["MU_X"][0])

#pertbar = resp.pertbar
#for a in range(0, 1):
#    string_a = "MU_" + resp.cart[a]
#    for b in range(0, 1):
#        string_b = "MU_" + resp.cart[b]
        #Y1_B, Y2_B, _ = Y_2[string_b]
        #print(X_2[string_b][0])
#        X1_B, X2_B, _ = X_2[string_b]
        # polar_AB_pos[a,b] = resp.linresp_asym(string_a, string_b, X1_B, X2_B, Y1_B, Y2_B) #, X1_B, X2_B, Y1_B, Y2_B)     #, X_1[string_a], X_1[string_b], Y_1[string_a], Y_1[string_b])
        # if a==0 and b==0:
#        rX1 = resp.r_X1(pertbar[string_a], omega1)
        # lX1 = resp.lr_X1(pertbar[string_a], omega1)
# print("="*5, "Canonical", "="*5)
# print(polar_AB_pos[0,0])
# print(polar_AB_pos[1,1])
# print(polar_AB_pos[2,2])

#convetional to local
lcc = pycc.ccwfn(rhf_wfn)
lecc =lcc.solve_cc(e_conv, r_conv)
lhbar = pycc.cchbar(lcc)
lcclambda = pycc.cclambda(lcc, lhbar)
llecc = lcclambda.solve_lambda(e_conv, r_conv)
ldensity = pycc.ccdensity(lcc, lcclambda)

lresp = pycc.ccresponse(ldensity)

omega1 = 0.0656
#omega2 = 0.0656


#A = resp.pertbar.Aoo

#resp.linresp(omega1)
# Creating dictionaries
# X_1 = X(-omega); X_2 = X(omega)
# Y_1 = Y(-omega); Y_2 = Y(omega)
# X_neg = X1(-omega) , X2(-omega)
# X_pos, Y_neg, Y_pos
X_1 = {}
X_2 = {}
Y_1 = {}
Y_2 = {}

string = "MU_X"

A = lresp.pertbar[string]
#print("Aoo",A)

X_2[string] = lresp.local_solve_right(A, omega1, cc.Local.Q, cc.Local.L, cc.Local.dim, cc.Local.eps, cclambda.hbar.Hoo, e_conv=1e-08, r_conv=1e-08, maxiter=10)
#Y_2[string] = lresp.solve_left(A, omega1)
#X_1[string] = lresp.local_solve_right(A, -omega1, e_conv=1e-08, r_conv=1e-08, maxiter=10)
#Y_1[string] = lresp.solve_left(A, -omega1)

#resp.polar(omega1)
# Grabbing X, Y and declaring the matrix space for LR
# lpolar_AB_pos = np.zeros((3,3))
# lpolar_AB_neg = np.zeros((3,3))
# lpolar_AB_aveg = np.zeros((3,3))

#pertbar = resp.pertbar
#for a in range(0, 1):
#    string_a = "MU_" + lresp.cart[a]
#    for b in range(0, 1):
#        string_b = "MU_" + lresp.cart[b]
#        #Y1_B, Y2_B, _ = Y_2[string_b]
#        X1_B, X2_B, _ = X_2[string_b]

        # lpolar_AB_pos[a,b] =lresp.local_linresp(string_a, X1_B, Y1_B, X2_B, Y2_B,cc.Local.Q, cc.Local.L, cc.Local.dim)
        # rX1 = resp.r_X1(pertbar, omega1)
#        lX1 = resp.lr_X1(pertbar[string_a], omega1)
# print("="*5, "Local", "="*5)
# print(lpolar_AB_pos[0,0])
# print(lpolar_AB_pos[1,1])
# print(lpolar_AB_pos[2,2])

