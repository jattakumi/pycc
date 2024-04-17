# Test case for computing dipole moment with simulation code" 
import psi4 
import pycc
import numpy as np
#import h5py
# from ccwfn import ccwfn
# from cchbar import cchbar
# from cclambda import cclambda
# from ccdensity import ccdensity
import sys
sys.path.append ("/Users/jattakumi/pycc/pycc")
from data.molecules import *
from opt_einsum import contract
#from hamiltonian_AO import Hamiltonian_AO
#from hfwfn import hfwfn
import os

# Psi4 Setup
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': '3-21G',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12,
                  'diis': 1
})

hf = """
F 0.0000000 0.0000000  -0.087290008493927
H 0.0000000 0.0000000 1.645494724632280
units bohr
no_reorient
symmetry c1
"""

mol = psi4.geometry(hf)

PNO_cutoff = 0
basis = 'PNO'
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
ccsd = pycc.ccwfn(rhf_wfn, model= 'CCSD', local=basis, local_cutoff = PNO_cutoff, filter=True)

e_conv = 1e-8
r_conv = 1e-8
maxiter = 200

eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter)
hbar = pycc.cchbar(ccsd)
cclambda = pycc.cclambda(ccsd, hbar)
lccsd = cclambda.solve_lambda(e_conv, r_conv)
ccdensity = pycc.ccdensity(ccsd, cclambda, True)
ecc_density = ccdensity.compute_energy()
t1 = ccdensity.ccwfn.t1
t2 = ccdensity.ccwfn.t2
l1 = ccdensity.cclambda.l1
l2 = ccdensity.cclambda.l2
opdm = ccdensity.compute_onepdm(t1,t2,l1,l2) # withref = True)

#hard coding local dipole moment contract('pq,pq->', opdm, mu)
mu = ccsd.H.mu
o = ccsd.o 
v = ccsd.v
no = ccsd.no
nv = ccsd.nv
lQ = ccsd.Local.Q
lL = ccsd.Local.L
dim = ccsd.Local.dim

mu_z_oo = 0
mu_z_vo = 0
mu_z_ov = 0
mu_z_vv = 0

# simulation code 
for i in range(no):
    for j in range(no):
        #print("i,j", i,j)
        mu_z_oo += opdm[i,j] * mu[2][i,j]

print("added ij", mu_z_oo)
for i in range(no):
    for a in range(nv):
        mu_z_ov += opdm[i,no + a] * mu[2][i, no+ a]
        mu_z_vo += opdm[no +a , i] * mu[2][ no + a, i]

print("added ia", mu_z_ov)
print("added ai", mu_z_vo)

for a in range(nv):
    for b in range(nv):
        mu_z_vv += opdm[no + a,no+ b] * mu[2][no + a,no + b]

dpz_scf = psi4.core.variable('SCF Dipole')[2]
print("added ab", mu_z_vv)
print("total", mu_z_oo + mu_z_ov + mu_z_vo + mu_z_vv ) #+ dpz_scf)  

## local code 
# intialize variables from ccwfn, cchabr, cclambda, hamiltonian (MO mu), etc. 
# Psi4 Setup
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': '3-21G',
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12,
                  'diis': 1
})

mol = psi4.geometry(hf)

PNO_cutoff = 0
basis = 'PNO'
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
ccsd_local = pycc.ccwfn(rhf_wfn, model= 'CCSD', local=basis, local_cutoff = PNO_cutoff, filter=True)

e_conv = 1e-12
r_conv = 1e-12
maxiter = 200

eccsd = ccsd_local.solve_cc(e_conv, r_conv, maxiter)
hbar_local = pycc.cchbar(ccsd_local)
cclambda_local = pycc.cclambda(ccsd_local, hbar_local)
lccsd_local = cclambda_local.solve_lambda(e_conv, r_conv)
ccdensity_local = pycc.ccdensity(ccsd_local, cclambda_local)
ecc_density = ccdensity_local.compute_energy()
t1 = ccdensity_local.ccwfn.t1
t2 = ccdensity_local.ccwfn.t2
l1 = ccdensity_local.cclambda.l1
l2 = ccdensity_local.cclambda.l2
lDvv = []
lDov = []
#opdm = ccdensity_local.compute_onepdm(t1,t2,l1,l2) # withref = True)
lDoo, oo_energy = ccdensity_local.build_lDoo(t1, t2, l1, l2)
lDvv, vv_energy = ccdensity_local.build_lDvv(lDvv, t1, t2, l1, l2)
lDov, ov_energy = ccdensity_local.build_lDov(lDov, t1, t2, l1, l2)
lDvo, vo_energy = ccdensity_local.build_lDvo(t2, l2)

#hard coding local dipole moment contract('pq,pq->', opdm, mu)
mu = ccsd_local.H.mu
o = ccsd_local.o
v = ccsd_local.v
no = ccsd_local.no
nv = ccsd_local.nv
lQ = ccsd_local.Local.Q
lL = ccsd_local.Local.L
dim = ccsd_local.Local.dim

# convert MO mu to PNO mu
lmu_ov = []
lmu_vv = []
lmu_oo = []
lmu_vo = []
for i in range(no):
    for j in range(no):
        ij = i * no + j
        lmu_ov = np.zeros((no, dim[ij]))
        lmu_ov = mu[2][i] @ (lQ[ij] @ lL[ij])

        lmu_vv = np.zeros((dim[ij], dim[ij]))
        lmu_vv = mu[2][i] @ (lQ[ij] @ lL[ij])

        lmu_oo = np.zeros((no, no))
        lmu_oo = mu[2][i] @ (lQ[ij] @ lL[ij])

        lmu_vo = np.zeros((dim[ij], no))
        lmu_vo = mu[2][i] @ (lQ[ij] @ lL[ij])

# similar format to simulation code block 
# simulation code 
for i in range(no):
    for j in range(no):
        #print("i,j", i,j)
        mu_z_oo += lDoo[i, j] * lmu_oo[i, j]

print("added ij", mu_z_oo)
for i in range(no):
    for a in range(nv):
        mu_z_ov += lDov[i,no + a] * lmu_ov[i, no + a]
        mu_z_vo += lDvo[no +a , i] * lmu_vo[no + a, i]

print("added ia", mu_z_ov)
print("added ai", mu_z_vo)

for a in range(nv):
    for b in range(nv):
        mu_z_vv += lDvv[no + a,no+ b] * lmu_vv[no + a,no + b]

print("added ab", mu_z_vv)
print("total", mu_z_oo + mu_z_ov + mu_z_vo + mu_z_vv ) #+ dpz_scf)
