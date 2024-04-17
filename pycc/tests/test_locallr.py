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
#sys.path.append("/Users/jattakumi/pycc/pycc/")
sys.path.append("/Users/josemarcmadriaga/pycc_josh/pycc/pycc")
from data.molecules import *

psi4.set_memory('2 GiB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': '3-21G',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12
})
mol = psi4.geometry(moldict["H2O"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

e_conv = 1e-12
r_conv = 1e-12

cc = pycc.ccwfn(rhf_wfn, local = 'PNO', local_cutoff = 1e-05, filter = True)
ecc = cc.solve_cc(e_conv, r_conv)
hbar = pycc.cchbar(cc)
cclambda = pycc.cclambda(cc, hbar)
lecc = cclambda.solve_lambda(e_conv, r_conv)
density = pycc.ccdensity(cc, cclambda)

resp = pycc.ccresponse(density)

omega1 = 0.0656

X_1 = {}
X_2 = {}
Y_1 = {}
Y_2 = {}

string = "MU_X"

A = resp.pertbar[string]

X_2[string] = resp.solve_right(A, omega1, e_conv=1e-12, r_conv=1e-12, maxiter=10)

#local
lcc = pycc.ccwfn(rhf_wfn,  local = 'PNO', local_cutoff = 1e-05, filter=False)
lecc =lcc.lccwfn.solve_lcc(e_conv, r_conv)
lhbar = pycc.cchbar(lcc)
lcclambda = pycc.cclambda(lcc, lhbar)
llecc = lcclambda.solve_llambda(e_conv, r_conv)
ldensity = pycc.ccdensity(lcc, lcclambda)

lresp = pycc.ccresponse(ldensity)

omega1 = 0.0656

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

A = lresp.lpertbar[string]

X_2[string] = lresp.local_solve_right(A, omega1, e_conv=1e-12, r_conv=1e-12, maxiter=10)

