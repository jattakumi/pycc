"""
Test CCSD linear response functions.
"""
import numpy as np
# Import package, test suite, and other packages as needed
import psi4
from opt_einsum import contract

#import pycc
from ccwfn import ccwfn
from lccwfn import lccwfn
from cchbar import cchbar
from cclambda import cclambda
from ccdensity import ccdensity
from ccresponse import ccresponse
import sys
#sys.path.append("/Users/jattakumi/pycc/pycc/")
sys.path.append("/Users/josemarcmadriaga/pycc_josh/pycc/pycc")
from data.molecules import *

psi4.set_memory('2 GiB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': '3-21G',
                  'scf_type': 'direct',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'r_convergence': 1e-13
})
mol = psi4.geometry(moldict["(H2)_2"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

e_conv = 1e-13
r_conv = 1e-13

# just going grab conventional hbar
#conv_cc = ccwfn(rhf_wfn)
#conv_ecc = conv_cc.solve_cc(e_conv, r_conv)
#conv_hbar = cchbar(conv_cc)

#sim 
cc_sim = ccwfn(rhf_wfn, local = 'PNO', local_mos = 'BOYS', local_cutoff = 1e-06, filter = True)
ecc = cc_sim.solve_cc(e_conv, r_conv)
hbar_sim = cchbar(cc_sim)

#Q = cc_sim.Local.Q
#L = cc_sim.Local.L
#ERI = cc_sim.H.ERI
#t1 = cc_sim.t1
#v = cc_sim.v
#for i in range(cc_sim.no):
#    ii = i*cc_sim.no + i 
#    QL_ii = Q[ii] @ L[ii]
#    for j in range(cc_sim.no):
#        ij = i*cc_sim.no + j
#        #jj = j*cc_sim.no + j 
#        QL_ij = Q[ij] @ L[ij]
#        #QL_jj = Q[jj] @ L[jj]
#        for m in range(cc_sim.no):
#            mm = m*cc_sim.no + m
#            im = i*cc_sim.no + m
#            ijm = ij*cc_sim.no + m 
#            QL_mm = Q[mm] @ L[mm]
#            QL_im = Q[im] @ L[im]
#            print("t", ijm, t1[i] @ QL_im) 
#            print("ER", ijm, QL_mm.T @ ERI[i,j,v,v] @ QL_ij) 
#            print("Hvovv", ijm, contract('fea, fF, eE, aA-> FEA', hbar_sim.Hvovv[:,j,:,:], QL_im, QL_mm, QL_ij))
#        #print("Hvvvo", ij, contract('abe, aA, bB, eE-> ABE', hbar_sim.Hvvvo[:,:,:,j], QL_ij, QL_ij, QL_ii))  

cclambda_sim = cclambda(cc_sim, hbar_sim)
lecc = cclambda_sim.solve_lambda(e_conv, r_conv)
density = ccdensity(cc_sim, cclambda_sim)
resp = ccresponse(density)

omega1 = 0.0656

X_1 = {}
X_2 = {}
Y_1 = {}
Y_2 = {}

string = "MU_Z"

A = resp.pertbar[string]

X_2[string] = resp.solve_right(A, omega1, e_conv=1e-09, r_conv=1e-09, maxiter=20)
Y_2[string] = resp.solve_left(A, omega1, e_conv=1e-09, r_conv=1e-09, maxiter=20)

#local
lcc = ccwfn(rhf_wfn,  local = 'PNO', local_mos = 'BOYS', local_cutoff = 1e-06, filter=False)
lecc =lcc.lccwfn.solve_lcc(e_conv, r_conv)
lhbar = cchbar(lcc)
lcclambda = cclambda(lcc, lhbar)
llecc = lcclambda.solve_llambda(e_conv, r_conv)
ldensity = ccdensity(lcc, lcclambda)

lresp = ccresponse(ldensity)

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

string = "MU_Z"

A = lresp.lpertbar[string]

X_2[string] = lresp.local_solve_right(A, omega1, hbar_sim, e_conv=1e-09, r_conv=1e-09, maxiter = 20) 
Y_2[string] = lresp.local_solve_left(A, omega1, e_conv= 1e-09, r_conv=1e-09, maxiter =20)
