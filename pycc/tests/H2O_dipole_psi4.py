import psi4
import sys 
sys.path.append('/Users/jattakumi/pycc/pycc') 
from data.molecules import * 
 
hf ="""
H   -0.000000000000     0.000000000000    -1.866311312218
F    0.000000000000     0.000000000000     0.099003860602
#F 0.0000000 0.0000000  -0.087290008493927
#H 0.0000000 0.0000000 1.645494724632280
units bohr
no_reorient
symmetry c1
"""

methyl_fluoride =""" 
#C           -0.138665616539    -1.504938860272     0.000000000000
#H           -1.168566355560    -2.099552330851     1.681820860415
#H           -1.168566355560    -2.099552330851    -1.681820860415
#F            0.119678932455     1.298876245835     0.000000000000
#H            1.732145544803    -2.366824980822     0.000000000000
#C           -0.073378684176    -0.796379348308     0.000000000000    
#H           -0.618378684518    -1.111035246096     0.889981271761    
#H           -0.618378684518    -1.111035246096    -0.889981271761    
#F            0.063331363652     0.687335708776     0.000000000000    
#H            0.916611947873    -1.252469841496     0.000000000000 
C 0.00000000 0.00000000 0.00000000
H 1.09000000 0.00000000 0.00000000
H -0.36333333 0.83908239 0.59332085
F -0.49666667 0.12889147 -1.39885997
H -0.36333333 -0.93337212 0.43000624
units bohr
no_reorient
symmetry c1
"""

geom_inp ="""
N -0.000000    0.000000    0.127486
H  0.000000    0.932442   -0.297468
H -0.807519   -0.466221   -0.297468
H  0.807519   -0.466221   -0.297468
units angstrom
no_reorient
symmetry c1
"""

psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', True)
psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-12,
                  'r_convergence': 1e-12,
                  'diis': 1,
                  #'perturb_h': True,
                  #'perturb_with':'Dipole',
                  #'perturb_dipole': [0.0,0.0,0.0001]
})

geom = psi4.geometry(methyl_fluoride)
rhf_e, rhf_wfn = psi4.energy('scf', return_wfn=True)
psi4.properties('CCSD',properties=['dipole'])

dpz_tot = psi4.core.variable('CCSD DIPOLE')

#dpz_tot contains contributions from SCF and CCSD as well as nuclear, need to remove SCF contribution and nuclear contribution
dpz_scf = psi4.core.variable('SCF DIPOLE')
mu_n = geom.nuclear_dipole()
print('dpz_tot', dpz_tot - mu_n[2])
elec_muz_scf = dpz_scf[2] - mu_n[2]
elec_muz_tot = dpz_tot[2] - mu_n[2]
elec_muz_cc = elec_muz_tot - elec_muz_scf
print('electric contribution only for CC', elec_muz_tot - elec_muz_scf) 
print(elec_muz_cc + elec_muz_scf + mu_n[2])
