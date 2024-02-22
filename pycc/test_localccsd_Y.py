"""
Test basic local lambda CCSD energies(local = PNO, PNO++, PAO)
"""
# Import package, test suite, and other packages as needed
import psi4
import pycc 
import sys
sys.path.append("/Users/josemarcmadriaga/pycc_josh/pycc/pycc") 
from data.molecules import *
 
"""H2O PNO-Lambda CCSD Test"""    
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': '3-21G',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-13,
                  'd_convergence': 1e-13,
                  'r_convergence': 1e-13,
                  'diis': 1})
mol = psi4.geometry(moldict["H2O"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

maxiter = 200
e_conv = 1e-12
r_conv = 1e-12
   
#simulation code of pno-ccsd lambda
ccsd_sim = pycc.ccwfn(rhf_wfn, model='CCSD',local='PNO', local_cutoff=1e-5,it2_opt=False,filter=True)
eccsd_sim = ccsd_sim.solve_cc(e_conv, r_conv, maxiter)
hbar_sim = pycc.cchbar(ccsd_sim)
cclambda_sim = pycc.cclambda(ccsd_sim, hbar_sim)
l_ccsd_sim = cclambda_sim.solve_lambda(e_conv, r_conv, maxiter)
density_sim = pycc.ccdensity(ccsd_sim, cclambda_sim)
resp_sim = pycc.ccresponse(density_sim)
omega1 =  0.0656

# Creating dictionaries
# X_1 = X(-omega); X_2 = X(omega)
# Y_1 = Y(-omega); Y_2 = Y(omega)
# X_neg = X1(-omega) , X2(-omega)
# X_pos, Y_neg, Y_pos
X_1 = {}
X_2 = {}
Y_1 = {}
Y_2 = {}


for axis in range(0, 3):
    string = "MU_" + resp_sim.cart[axis]

    A = resp_sim.pertbar[string]
    X_2[string] = resp.solve_right(A, omega1)
    Y_2[string] = resp.solve_left(A, omega1)
    #X_1[string] = resp.solve_right(A, -omega1)
    #Y_1[string] = resp.solve_left(A, -omega1)

#pno-ccsd lambda
lccsd = pycc.ccwfn(rhf_wfn,model='CCSD', local='PNO', local_cutoff=1e-5,it2_opt=False)
elccsd = lccsd.lccwfn.solve_lcc(e_conv, r_conv, maxiter)
lhbar = pycc.cchbar(lccsd)  
lcclambda = pycc.cclambda(lccsd, lhbar)
l_lccsd = lcclambda.solve_llambda(e_conv, r_conv, maxiter) 
ldensity = pycc.ccdensity(lccsd, lcclambda)
lresp = pycc.ccresponse(ldensity) 

# Creating dictionaries
# X_1 = X(-omega); X_2 = X(omega)
# Y_1 = Y(-omega); Y_2 = Y(omega)
# X_neg = X1(-omega) , X2(-omega)
# X_pos, Y_neg, Y_pos
lX_1 = {}
lX_2 = {}
lY_1 = {}
lY_2 = {}

#for axis in range(0, 3): 
#    string = "MU_" + resp.cart[axis]

#    A = resp.lpertbar[string]

#    lX_2[string] = resp.solve_right(A, omega1)
#    lY_2[string] = resp.lsolve_left(A, omega1)
#    lX_1[string] = resp.solve_right(A, -omega1)
#    lY_1[string] = resp.lsolve_left(A, -omega1)

#assert(abs(Y_1[0] - lY_1[0]) < 1e-8)
#assert(abs(Y_1[1] - lY_1[1]) < 1e-8)
#assert(abs(Y_2[0] - lY_2[0]) < 1e-8)
#assert(abs(Y_2[1] - lY_2[1]) < 1e-8)
