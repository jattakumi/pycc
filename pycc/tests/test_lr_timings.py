# Import package, test suite, and other packages as needed
import psi4
# import pycc
import pytest
import sys
sys.path.append("/Users/jattakumi/pycc/pycc/")
from data.molecules import *

num = ["1"]#, "2", "3", "4", "5", "6", "7"]

for n in num:

  geom = "(H2O)_" + str(n)
  psi4.core.clean_options()
  psi4.set_memory('8 GiB')
  psi4.core.set_output_file('output.dat', False)
  psi4.set_options({'basis': 'cc-pvdz',
           'scf_type': 'pk',
           'freeze_core':'true',
           'e_convergence': 1e-08,
           'd_convergence': 1e-08,
           'r_convergence': 1e-08
  })
  mol = psi4.geometry(moldict["H2O"])
  rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

  from pycc.ccwfn import *
  from lccwfn import *
  from pycc.cchbar import *
  from pycc.cclambda import *
  from pycc.ccdensity import *
  from pycc.ccresponse import *

  e_conv = 1e-08
  r_conv = 1e-08

  #local
  lcc = ccwfn(rhf_wfn, local = 'PNO++', local_mos = 'BOYS', local_cutoff = 1e-07, filter=False)
  lecc = lcc.lccwfn.solve_lcc(e_conv, r_conv)
  lhbar = cchbar(lcc)
  lcclambda = cclambda(lcc, lhbar)
  llecc = lcclambda.solve_llambda(e_conv, r_conv)
  ldensity = ccdensity(lcc, lcclambda)

  lresp = ccresponse(ldensity)

  omega1 = 0.0428
  X_A = {}
  X_B = {}
  # X_1 = {}
  # X_2 = {}
  # Y_1 = {}
  # Y_2 = {}

  for axis in range(0, 3):
    string = "MU_" + lresp.cart[axis]
    A = lresp.lpertbar[string]
    X_A[string] = lresp.local_solve_right(A, omega1, lhbar,  e_conv = 1e-08, r_conv = 1e-08)
    X_B[string] = lresp.local_solve_right(A, -omega1, lhbar,  e_conv = 1e-08, r_conv = 1e-08)
    # X_2[string] = lresp.local_solve_right(A, omega1, lhbar, e_conv=1e-08, r_conv=1e-08)
    # Y_2[string] = lresp.local_solve_left(A, omega1, e_conv=1e-08, r_conv=1e-08)
    # X_1[string] = lresp.local_solve_right(A, -omega1, lhbar, e_conv=1e-08, r_conv=1e-08)
    # Y_1[string] = lresp.local_solve_left(A, -omega1, e_conv=1e-08, r_conv=1e-08)
  # Grabbing X, Y and declaring the matrix space for LR
  lpolar_AB = np.zeros((3, 3))

  for a in range(0, 3):
    string_a = "MU_" + lresp.cart[a]
    X1_A, X2_A, _ = X_A[string_a]
    for b in range(0, 3):
      string_b = "MU_" + lresp.cart[b]
      X_1B, X_2B, _ = X_B[string_b]
      lpolar_AB[a, b] = lresp.lsym_linresp(string_a, string_b, X1_A, X2_A, X_1B, X_2B)
      # Y1_B, Y2_B, _ = Y_2[string_b]
      # X1_B, X2_B, _ = X_2[string_b]
      # lpolar_AB[a, b] = lresp.local_linresp(a, string_a, X1_B, Y1_B, X2_B, Y2_B)

  print(f"Dynamic Polarizability Tensor @ w = {omega1} a.u.:")
  print(lpolar_AB)
  print("Average Dynamic Polarizability:")
  lpolar_AB_avg = np.average([lpolar_AB[0, 0], lpolar_AB[1, 1], lpolar_AB[2, 2]])
  print(lpolar_AB_avg)

  timings = []
  # timings.append(lresp.pertbar_t)
  timings.append(lresp.lX1_t)
  timings.append(lresp.lX2_t)
  # timings.append(lresp.lY1_t)
  # timings.append(lresp.lY2_t)
  timings.append(lresp.lpseudoresponse_t)
  timings.append(lresp.lLCX_t)
  timings.append(lresp.lLHX1Y1_t)
  timings.append(lresp.lLHX2Y2_t)
  timings.append(lresp.lLHX1Y2_t)
  timings.append(lresp.lsym_lr_t)

  B_avg = str(geom)+"_polarizability_PNO++_timings.txt"

  with open(B_avg, 'a') as f1:
    for time in range(0,len(timings)):
       f1.write(str(timings[time]) + ' ')
       if time == len(timings):
         f1.write(str(timings[time]) + '\n')

  B_avg = str(geom)+"_polarizability_PNO++_Bavg.txt"
  with open(B_avg, 'a') as f1:
    f1.write(str(lcc.Local.T2_ratio)+' ')

    for i in range(0,3):
       if i != 2:
         f1.write(str(lpolar_AB[0, 0]) + ' ' +str(lpolar_AB[1, 1]) +' '+str(lpolar_AB[2, 2])+' ')
    f1.write(str(lpolar_AB[0][2,2,2]) + ' '+str(lpolar_AB[1])+ '\n')

  del lcc, lhbar, lcclambda, ldensity, lresp