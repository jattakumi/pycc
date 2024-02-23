"""
ccresponse.py: CC Response Functions
"""
import itertools

from opt_einsum import contract

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
import time
from .utils import helper_diis
from .cclambda import cclambda
#from .local import local 

class ccresponse(object):
    """
    An RHF-CC Response Property Object.

    Methods
    -------
    linresp():
        Compute a CC linear response function.
    solve_right():
        Solve the right-hand perturbed wave function equations.
    pertcheck():
        Check first-order perturbed wave functions for all available perturbation operators.
    """

    def __init__(self, ccdensity, omega1 = 0, omega2 = 0):
        """
        Parameters
        ----------
        ccdensity : PyCC ccdensity object
            Contains all components of the CC one- and two-electron densities, as well as references to the underlying ccwfn, cchbar, and cclambda objects
        omega1 : scalar
            The first external field frequency (for linear and quadratic response functions)
        omega2 : scalar
            The second external field frequency (for quadratic response functions)

        Returns
        -------
        None
        """

        self.ccwfn = ccdensity.ccwfn
        self.cclambda = ccdensity.cclambda
        self.H = self.ccwfn.H
        self.hbar = self.cclambda.hbar
        self.contract = self.ccwfn.contract
        # self.l1 = self.cclambda.l1
        # self.l2 = self.cclambda.l2
        self.no = self.ccwfn.no

        if self.ccwfn.local is not None and self.ccwfn.filter is not True: 
            self.lccwfn = ccdensity.lccwfn
            self.cclambda = ccdensity.cclambda
            self.H = self.ccwfn.H
            self.cchbar = self.cclambda.hbar
            self.contract = self.ccwfn.contract
            self.no = self.ccwfn.no
            self.Local = self.ccwfn.Local       

            # Cartesian indices
            self.cart = ["X", "Y", "Z"]

            # Build dictionary of similarity-transformed property integrals
            self.lpertbar = {}

            # Electric-dipole operator (length)
            for axis in range(3):
                key = "MU_" + self.cart[axis]
                self.lpertbar[key] = lpertbar(self.H.mu[axis], self.ccwfn, self.lccwfn)

        else: 
            if self.ccwfn.local is not None:
                self.Local = self.ccwfn.Local

            # Cartesian indices
            self.cart = ["X", "Y", "Z"]
    
            # Build dictionary of similarity-transformed property integrals
            self.pertbar = {}
    
            # Electric-dipole operator (length)
            for axis in range(3):
                key = "MU_" + self.cart[axis]
                self.pertbar[key] = pertbar(self.H.mu[axis], self.ccwfn)
    
            # # Magnetic-dipole operator
            # for axis in range(3):
            #     key = "M_" + self.cart[axis]
            #     self.pertbar[key] = pertbar(self.H.m[axis], self.ccwfn)
    
            # # Complex-conjugate of magnetic-dipole operator
            # for axis in range(3):
            #     key = "M*_" + self.cart[axis]
            #     self.pertbar[key] = pertbar(np.conj(self.H.m[axis]), self.ccwfn)
    
            # # Electric-dipole operator (velocity)
            # for axis in range(3):
            #     key = "P_" + self.cart[axis]
            #     self.pertbar[key] = pertbar(self.H.p[axis], self.ccwfn)
    
            # # Complex-conjugate of electric-dipole operator (velocity)
            # for axis in range(3):
            #     key = "P*_" + self.cart[axis]
            #     self.pertbar[key] = pertbar(np.conj(self.H.p[axis]), self.ccwfn)
    
            # # Traceless quadrupole
            # ij = 0
            # for axis1 in range(3):
            #     for axis2 in range(axis1,3):
            #         key = "Q_" + self.cart[axis1] + self.cart[axis2]
            #         self.pertbar[key] = pertbar(self.H.Q[ij], self.ccwfn)
            #         if (axis1 != axis2):
            #             key2 = "Q_" + self.cart[axis2] + self.cart[axis1]
            #             self.pertbar[key2] = self.pertbar[key]
            #         ij += 1
    
            # HBAR-based denominators
            #modying to only run in simulation code
            if self.ccwfn.filter is True:
                eps_occ = np.diag(self.hbar.Hoo)
                #print("eps_occ", eps_occ)
                eps_vir = np.diag(self.hbar.Hvv)
                #tmp = 0 
                #for i in range(self.ccwfn.no):
                    #ii = i*self.ccwfn.no + i 
                    #QL = self.ccwfn.Local.Q[ii] @ self.ccwfn.Local.L[ii] 
                    #tmp = QL.T @ self.hbar.Hvv @ QL
                    #tmp = QL @ tmp @ QL.T                   
                    #print("eps_vir[ii]", ii, np.diag(QL.T @ self.hbar.Hvv @ QL)) 
                #print("reshaped eps_occ", eps_occ.reshape(-1,1))
                #eps_vir = np.diag(tmp)
                self.Dia = eps_occ.reshape(-1,1) #- eps_vir
                self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) #- eps_vir.reshape(-1,1) - eps_vir

    def pertcheck(self, omega, e_conv=1e-13, r_conv=1e-13, maxiter=200, max_diis=8, start_diis=1):
        """
        Build first-order perturbed wave functions for all available perturbations and return a dict of their converged pseudoresponse values.  Primarily for testing purposes.

        Parameters
        ----------
        omega: float
            The external field frequency.
        e_conv : float
            convergence condition for the pseudoresponse value (default if 1e-13)
        r_conv : float
            convergence condition for perturbed wave function rmsd (default if 1e-13)
        maxiter : int
            maximum allowed number of iterations of the wave function equations (default is 100)
        max_diis : int
            maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
        start_diis : int
            earliest iteration to start DIIS extrapolations (default is 1)

        Returns
        -------
        check: dictionary
            Converged pseudoresponse values for all available perturbations.
        """
        # dictionaries for perturbed wave functions and test pseudoresponses
        X1 = {}
        X2 = {}
        check = {}

        # Electric-dipole (length)
        for axis in range(3):
            pertkey = "MU_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Magnetic-dipole
        for axis in range(3):
            pertkey = "M_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Complex-conjugate of magnetic-dipole
        for axis in range(3):
            pertkey = "M*_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Electric-dipole (velocity)
        for axis in range(3):
            pertkey = "P_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Complex-conjugate of electric-dipole (velocity)
        for axis in range(3):
            pertkey = "P*_" + self.cart[axis]
            X_key = pertkey + "_" + f"{omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
            check[X_key] = polar
            if (omega != 0.0):
                X_key = pertkey + "_" + f"{-omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar

        # Traceless quadrupole
        for axis1 in range(3):
            for axis2 in range(3):
                pertkey = "Q_" + self.cart[axis1] + self.cart[axis2]
                X_key = pertkey + "_" + f"{omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check[X_key] = polar
                if (omega != 0.0):
                    X_key = pertkey + "_" + f"{-omega:0.6f}"
                    print("Solving right-hand perturbed wave function for %s:" % (X_key))
                    X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                    check[X_key] = polar
        

        return check

    def linresp(self, A, B, omega, e_conv=1e-13, r_conv=1e-13, maxiter=200, max_diis=8, start_diis=1):
        """
        Calculate the CC linear-response function for one-electron perturbations A and B at field-frequency omega (w).

        The linear response function, <<A;B>>w, generally requires the following perturbed wave functions and frequencies:
            A(-w), A*(w), B(w), B*(-w)
        If the external field is static (w=0), then we need:
            A(0), A*(0), B(0), B*(0)
        If the perturbation A is real and B is pure imaginary:
            A(-w), A(w), B(w), B*(-w)
        or vice versa:
            A(-w), A*(w), B(w), B(-w)
        If the perturbations are both real and the field is static:
            A(0), B(0)
        If the perturbations are identical then:
            A(w), A*(-w) or A(0), A*(0)
        If the perturbations are identical, the field is dynamic and the operator is real:
            A(-w), A(w)
        If the perturbations are identical, the field is static and the operator is real:
            A(0)

        Parameters:
        -----------
        A: string
            String identifying the left-hand perturbation operator.
        B: string
            String identifying the right-hand perturbation operator.
        NB: Allowed values for A and B are:
            "MU": Electric dipole operator (length)
            "P": Electric dipole operator (velocity)
            "P*": Complex conjugate of electric dipole operator (velocity)
            "M": Magnetic dipole operator
            "M*": Complex conjugate of Magnetic dipole operator
            "Q": Traceless quadrupole operator
        omega: float
            The external field frequency.
        e_conv : float
            convergence condition for the pseudoresponse value (default if 1e-13)
        r_conv : float
            convergence condition for perturbed wave function rmsd (default if 1e-13)
        maxiter : int
            maximum allowed number of iterations of the wave function equations (default is 100)
        max_diis : int
            maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
        start_diis : int
            earliest iteration to start DIIS extrapolations (default is 1)

        Returns:
        --------
        linresp: NumPy array
            A 3x3 or 9 x 3 x 3 array of values of the chosen linear response function.
        """

        A = A.upper()
        B = B.upper()

        # dictionaries for perturbed wave functions
        X1 = {}
        X2 = {}
        for axis in range(3):
            # A(-w) or A(0)
            pertkey = A + "_" + self.cart[axis]
            X_key = pertkey + "_" + f"{-omega:0.6f}"
            print("Solving right-hand perturbed wave function for %s:" % (X_key))
            X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)

            # A(w) or A*(w) 
            if (omega != 0.0):
                if (np.iscomplexobj(self.pertbar[pertkey].Aoo)):
                    pertkey = A + "*_" + self.cart[axis]
                X_key = pertkey + "_" + f"{omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)


        if (B != A):
            for axis in range(3):
                pertkey = B + "_" + self.cart[axis]
                X_key = pertkey + "_" + f"{omega:0.6f}"
                print("Solving right-hand perturbed wave function for %s:" % (X_key))
                #X_2[pertkey] = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check.append(polar)
                X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                check.append(polar)
                if (omega != 0.0):
                    X_key = pertkey + "_" + f"{-omega:0.6f}"
                    print("Solving right-hand perturbed wave function for %s:" % (X_key))
                    X1[X_key], X2[X_key], polar = self.solve_right(self.pertbar[pertkey], -omega, e_conv, r_conv, maxiter, max_diis, start_diis)
                    check.append(polar)


    def linresp_asym(self, pertkey_a, pertkey_b, X1_B, X2_B, Y1_B, Y2_B):

        # Defining the l1 and l2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2

        # Grab X and Y amplitudes corresponding to perturbation B, omega1
        # X1_B = ccpert_X_B[0]
        # X2_B = ccpert_X_B[1]
        # Y1_B = ccpert_Y_B[0]
        # Y2_B = ccpert_Y_B[1]

        # Please refer to eqn 78 of [Crawford:xxxx].
        # Writing H(1)(omega) = B, T(1)(omega) = X, L(1)(omega) = y
        # <<A;B>> = <0|Y(B) * A_bar|0> + <0| (1 + L(0))[A_bar, X(B)}|0>
        #                 polar1                polar2
        polar1 = 0
        polar2 = 0
        # <0|Y1(B) * A_bar|0>
        pertbar_A = self.pertbar[pertkey_a]
        pertbar_B = self.pertbar[pertkey_b]

        # print("="*10, "Avo", "="*10, '\n', np.linalg.norm(pertbar_A.Avo))

        # print("="*10, "Y_oovv", "="*10, '\n', np.linalg.norm(Y2_B))

        Avvoo = pertbar_A.Avvoo.swapaxes(0,2).swapaxes(1,3)
        polar1 += contract("ai, ia -> ", pertbar_A.Avo, Y1_B)
        # <0|Y2(B) * A_bar|0>
        polar1 += 0.5 * contract("abij, ijab -> ", Avvoo, Y2_B)
        polar1 += 0.5 * contract("baji, ijab -> ", Avvoo, Y2_B)
        # <0|[A_bar, X(B)]|0>
        polar2 += 2.0 * contract("ia, ia -> ", pertbar_A.Aov, X1_B)
        # <0|L1(0) [A_bar, X2(B)]|0>
        tmp = contract("ia, ic -> ac", l1, X1_B)
        polar2 += contract("ac, ac -> ", tmp, pertbar_A.Avv)
        tmp = contract("ia, ka -> ik", l1, X1_B)
        polar2 -= contract("ik, ki -> ", tmp, pertbar_A.Aoo)
        # # # # <0|L1(0)[a_bar, X2(B)]|0>
        tmp = contract("ia, jb -> ijab", l1, pertbar_A.Aov)
        polar2 += 2.0 * contract("ijab, ijab -> ", tmp, X2_B)
        polar2 += -1.0 * contract("ijab, ijba -> ", tmp, X2_B)
        # <0|L2(0)[A_bar, X1(B)]|0>
        #lij b_{ij} c_{ij} Ab_{ij} c_{ij} a_{ii} j -> Zia_{ii} * Xia_{ii}
        tmp = contract("ijbc, bcaj -> ia", l2, pertbar_A.Avvvo)
        polar2 += contract("ia, ia -> ", tmp, X1_B)
        tmp = contract("ijab, kbij -> ak", l2, pertbar_A.Aovoo)
        polar2 -= 0.5 * contract("ak, ka -> ", tmp, X1_B)
        tmp = contract("ijab, kaji -> bk", l2, pertbar_A.Aovoo)
        polar2 -= 0.5 * contract("bk, kb -> ", tmp, X1_B)
        # <0|L2(0)[A_bar, X2(B)]|0>
        tmp = contract("ijab, kjab -> ik", l2, X2_B)
        polar2 -= 0.5 * contract("ik, ki -> ", tmp, pertbar_A.Aoo)
        tmp = contract("ijab, kiba-> jk", l2, X2_B)
        polar2 -= 0.5 * contract("jk, kj -> ", tmp, pertbar_A.Aoo)
        tmp = contract("ijab, ijac -> bc", l2, X2_B)
        polar2 += 0.5 * contract("bc, bc -> ", tmp, pertbar_A.Avv)
        tmp = contract("ijab, ijcb -> ac", l2, X2_B)
        polar2 += 0.5 * contract("ac, ac -> ", tmp, pertbar_A.Avv)

        return -1.0 * (polar1 + polar2)

    def local_linresp(self, pertkey_a, X1_B, Y1_B, X2_B, Y2_B, Q, L, dim  ):
        contract = self.contract
        polar1 = 0.0
        polar2 = 0.0
        # Aoo = []
        Aov = []
        Avv = []
        Avo = []
        Avvoo = []
        Avvvo = []
        Avvvo_ii = []
        Aovoo = []
        X_ov = []
        Y_ov = []
        X_oovv = []
        Y_oovv = []
        no = self.no
        pertbar_A = self.pertbar[pertkey_a]
        Aoo = pertbar_A.Aoo
        # print('Aoo\n', Aoo)
        #Q = self.ccwfn.Local.Q
        #L = self.ccwfn.Local.L
        l1 = []
        l2 = []
        for i in range(no):
            ii = i * no + i
            QL_ii = Q[ii] @ L[ii]
            # print("="*10, "local_Avo", "="*10,'\n', (Avo))  #np.linalg.norm
            lX_ov = contract('ia, aA -> iA', X1_B, QL_ii)
            lY_ov = contract('ia, aA -> iA', Y1_B, QL_ii)
            # if ii ==0:
            #     print("="*10, "local_Avo", "="*10,'\n', np.linalg.norm(Avo))  #np.linalg.norm
            l_1 = contract('ia, aA -> iA', self.cclambda.l1, QL_ii)
            X_ov.append(lX_ov)
            Y_ov.append(lY_ov)
            l1.append(l_1)
            # print("="*10, "local_Y_ov", "="*10,'\n', np.linalg.norm(Y_ov))
            for j in range(no):
                ij = i * no + j
                QL_ij = Q[ij] @ L[ij]
                lAvo = (pertbar_A.Avo.T @ QL_ij).T
                lAov = (pertbar_A.Aov @ QL_ij)
                lAvv = (QL_ij.T @ pertbar_A.Avv @ QL_ij)
                Avo.append(lAvo)
                Aov.append(lAov)
                Avv.append(lAvv)
                lAvvoo = contract('ijab, aA, bB->ABij', pertbar_A.Avvoo, QL_ij, QL_ij) # QL_ij.T @ pertbar_A.Avvoo[i,j] @ QL_ij
                Avvoo.append(lAvvoo)
                lAvvvo = np.zeros((dim[ij], dim[ij], dim[ij], no))
                lAvvvo += contract('abck, aA, bB, cC ->ABCk', pertbar_A.Avvvo, QL_ij, QL_ij, QL_ij)
                Avvvo.append(lAvvvo)
                #Ab_{ij} c_{ij} a_{ii} j
                lAvvvo = np.zeros((dim[ij], dim[ij], dim[ii], no))
                lAvvvo += contract('abck, aA, bB, cC ->ABCk', pertbar_A.Avvvo, QL_ij, QL_ij, QL_ii)
                Avvvo_ii.append(lAvvvo)
                lAovoo = np.zeros((no, dim[ij], no, no))
                lAovoo += contract('iajk, aA-> iAjk', pertbar_A.Aovoo, QL_ij)
                Aovoo.append(lAovoo)
                lX_oovv = np.zeros((no, no, dim[ij], dim[ij]))
                lX_oovv += contract('ijab, aA, bB -> ijAB', X2_B, QL_ij, QL_ij)
                X_oovv.append(lX_oovv)
                lY_oovv = np.zeros((no, no, dim[ij], dim[ij]))
                lY_oovv += contract('ijab, aA, bB -> ijAB', Y2_B, QL_ij, QL_ij)
                Y_oovv.append(lY_oovv)
                l_2 = contract('ijab, aA, bB -> ijAB', self.cclambda.l2, QL_ij, QL_ij)
                l2.append(l_2)
                # if ij == 0:
                #     print("=" * 10, "local_lY_oovv", "=" * 10, '\n', np.linalg.norm(lY_oovv))
        # Now implementing the local linear response code to compute polarizability
        for i in range(no):
            ii = i * no + i
            QL_ii = Q[ii] @ L[ii]
            # <0|Y1(B) * A_bar|0>
            # the first [ij] indicates the pair space, ij and the second [i] indicates the o part of Avo and Yov
            polar1 += contract("a, a -> ", Avo[ii][:, i], Y_ov[i][i])
            # <0|[A_bar, X(B)]|0>
            polar2 += 2.0 * contract("a, a -> ", Aov[ii][i, :], X_ov[i][i])
            # < 0|L1(0)[A_bar, X2(B)]|0 >
            tmp = contract("a, c -> ac", l1[i][i], X_ov[i][i])
            polar2 += contract("ac, ac -> ", tmp, Avv[ii][:, :])
            #6th polarterm
            for k in range(no):
                kk = k*no + k
                Siikk = (Q[ii] @ L[ii]).T @ (Q[kk] @ L[kk])  # (Q[ii] @ L[ii]).T @ (Q[kk] @ L[kk])
                tmp = contract("a, a -> ", l1[i][i], X_ov[k][k] @ Siikk.T)
                polar2 -= tmp * Aoo[k, i]
            for j in range(no):
                ij = i * no + j
                jj = j * no + j
                # <0|Y2(B) * A_bar|0>
                polar1 += 0.5 * contract("ab, ab -> ", Avvoo[ij][:, :, i, j], Y_oovv[ij][i, j, :, :])
                polar1 += 0.5 * contract("ba, ab -> ", Avvoo[ij][:, :, j, i], Y_oovv[ij][i, j, :, :])
                # <0|L1(0)[a_bar, X2(B)]|0>
                #lia_{ii} Ajb_{jj} -> (Sij,ii * Za_{ii} b_{jj} * Sijjj.T) * Xija_{ij} b_{ij}
                Sijii = (Q[ij] @ L[ij]).T @ (Q[ii] @ L[ii])
                Sijjj = (Q[ij] @ L[ij]).T @ (Q[jj] @ L[jj])
                tmp = contract("a, b -> ab", l1[i][i], Aov[jj][j, :])
                polar2 += 2.0 * contract("ab, ab -> ", Sijii @ tmp @ Sijjj.T, X_oovv[ij][i, j, :, :])
                polar2 += -1.0 * contract("ab, ba -> ", Sijii @ tmp @ Sijjj.T, X_oovv[ij][i, j, :, :])

                # <0|L2(0)[A_bar, X1(B)]|0>
                # lij b_{ij} c_{ij} Ab_{ij} c_{ij} a_{ii} j -> Zia_{ii} * Xia_{ii}
                tmp = contract("bc, bca -> a", l2[ij][i,j], Avvvo_ii[ij][: , :, :, j])
                polar2 += contract("a, a -> ", tmp, X_ov[i][i])
                for k in range(no):
                    kk = k * no + k
                    kj = k * no + j
                    ki = k * no + i
                    Skkij = (Q[kk] @ L[kk]).T @ (Q[ij] @ L[ij])
                    Skjij = (Q[kj] @ L[kj]).T @ (Q[ij] @ L[ij])
                    Skiij = (Q[ki] @ L[ki]).T @ (Q[ij] @ L[ij])
                    # 10th and 11th polar terms
                    #lija_{ij b_{ij} * Akb_{ij}ij -> Za_{ij}k * (Xka_{kk} * Skkij)
                    tmp = contract("ab, b -> a", l2[ij][i,j], Aovoo[ij][k, :, i, j])
                    polar2 -= 0.5 * contract("a, a -> ", tmp, X_ov[k][k] @ Skkij)

                    # lija_{ij b_{ij} * Aka_{ij}ij -> Zb_{ij}k * (Xkb_{kk} * Skkij)
                    tmp = contract("ab, a -> b", l2[ij][i, j], Aovoo[ij][k, :, j, i])
                    polar2 -= 0.5 * contract("b, b -> ", tmp, X_ov[k][k] @ Skkij)

                    # <0|L2(0)[A_bar, X1(B)]|0>
                    tmp = contract("ab, kjab -> kj", l2[ij][i, j], X_oovv[ij][:, :])
                    # 12 polar term
                    polar2 -= 0.5 * tmp[k, j] * Aoo[k, i]
                    tmp = contract("ab, ba-> ", l2[ij][i, j], Skiij.T @ X_oovv[ki][k, i, :, :] @ Skiij)
                    polar2 -= 0.5 * tmp * Aoo[k, j] #contract("jk, kj -> ", tmp, pertbar_A.Aoo)
                tmp = contract("ab, ac -> bc", l2[ij][i, j], X_oovv[ij][i, j, :, :])
                polar2 += 0.5 * contract("bc, bc -> ", tmp, Avv[ij][:, :])
                tmp = contract("ab, cb -> ac", l2[ij][i, j], X_oovv[ij][i, j, :, :])
                polar2 += 0.5 * contract("ac, ac -> ", tmp, Avv[ij][:, :])
        return -1.0 * (polar1 + polar2)

    def solve_right(self, pertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
        solver_start = time.time()

        Dia = self.Dia
        Dijab = self.Dijab
        
        # initial guess
        X1 = pertbar.Avo.T /(Dia) # + omega)
        X2 = pertbar.Avvoo/(Dijab) # + omega)

        X1, X2 = self.ccwfn.Local.filter_res(X1, X2)

        #for i in range(self.ccwfn.no):
            #ii = i*self.ccwfn.no + i
            #QL = self.Local.Q[ii] @ self.Local.L[ii]
            #print("Avo", i, pertbar.Avo[:,i].T @ QL)
            #eps_vir =  np.diag(QL.T @ self.hbar.Hvv @ QL)
            #for a in range(self.ccwfn.Local.dim[ii]):
            #     X1[a] /= eps_occ[i] - eps_vir[a] 
            #print("X", i, X1[i] @ QL)

        pseudo = self.pseudoresponse(pertbar, X1, X2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        #diis = helper_diis(X1, X2, max_diis)
        contract = self.ccwfn.contract

        self.X1 = X1
        self.X2 = X2

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            X1 = self.X1
            X2 = self.X2

            r1 = self.r_X1(pertbar, omega)
            r2 = self.r_X2(pertbar, omega)

            if self.ccwfn.local is not None:
                inc1, inc2 = self.ccwfn.Local.filter_pertamps(r1, r2, self.hbar)
                self.X1 += inc1
                self.X2 += inc2
                     
                rms = contract('ia,ia->', np.conj(inc1 / (Dia )), inc1 / (Dia)) # + omega))
                rms += contract('ijab,ijab->', np.conj(inc2 / (Dijab)), inc2 / (Dijab)) # + omega))
                rms = np.sqrt(rms)
            else:
                self.X1 += r1 / (Dia + omega)
                self.X2 += r2 / (Dijab + omega)

                rms = contract('ia,ia->', np.conj(r1 / (Dia + omega)), r1 / (Dia + omega))
                rms += contract('ijab,ijab->', np.conj(r2 / (Dijab + omega)), r2 / (Dijab + omega))
                rms = np.sqrt(rms)

            pseudo = self.pseudoresponse(pertbar, self.X1, self.X2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")

            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                return self.X1, self.X2, pseudo

            if niter == maxiter:
                print("\nPerturbed wave function not fully converged in %.3f seconds.\n" % (time.time() - solver_start))
                return self.X1, self.X2, pseudo



            #diis.add_error_vector(self.X1, self.X2)
            #if niter >= start_diis:
            #    self.X1, self.X2 = diis.extrapolate(self.X1, self.X2)

    def local_solve_right(self, lpertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200):#max_diis=7, start_diis=1):
        solver_start = time.time()

        no = self.no

        
        eps_occ = np.diag(self.cchbar.Hoo)
        #print("eps_occ", eps_occ)
        eps_lvir = []
        #eps_vir = np.diag(self.cchbar.Hvv)
        #for i in range(no):
            #ii = i *no + i 
           #for j in range(no):
                #ij = i*no + j 
                #eps_lvir.append(np.diag(self.cchbar.Hvv[ij]))
                #print("eps_lvir_ij", ij, self.cchbar.Hvv[ij])
        contract =self.contract

        Q = self.Local.Q
        L = self.Local.L
        Avo = lpertbar.Avo.copy()
        Avvoo = lpertbar.Avvoo.copy()
 
        self.X1 = []
        self.X2 = []
        #norm = 0
        for i in range(no):
            ii = i * no + i
            QL_ii = Q[ii] @ L[ii]

            #Xv{ii}
            lX1 = Avo[ii].copy()
            eps_lvir = self.cchbar.Hvv[ii] 
            #print("length of eps_lvir", len(eps_lvir))
            for a in range(self.Local.dim[ii]):
                lX1[a] /= (eps_occ[i]) #  - eps_lvir[a,a])
            #print("X", ii, lX1)
            #norm += np.linalg.norm(lX1)
            self.X1.append(lX1)
            #for j in range(no):
                #ij = i * no + j
                #QL_ij = Q[ij] @ L[ij]
                # if ij == ii:
                    # print("Avo", i, lAvo[:,i])
                #lX2 = np.zeros((no, no, dim[ij], dim[ij]))
                #lX2 = contract('ab, aA, bB -> AB', Avvoo[i,j] , QL_ij, QL_ij)/(eps[ij].reshape(1,-1) + eps[ij].reshape(-1,1)
                #    - self.H.F[i,i] - self.H.F[j,j] + omega)
                #self.X2.append(lX2)

                # lX2 = np.zeros((no, no, self.Local.dim[ij], self.Local.dim[ij]))
                # lX2 += contract('ijab, aA, bB -> ijAB', self.X2, QL_ij, QL_ij)
                # self.X2.append(lX2)
        #print("X_inital", norm)

        pseudo = self.local_pseudoresponse(lpertbar, self.X1, self.X2, omega)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        #diis = helper_diis(X1, X2, max_diis)
        contract = self.ccwfn.contract

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            X1 = self.X1
            # X2 = self.X2

            r1 = self.lr_X1(lpertbar, omega)
            # r2 = self.r_X2(pertbar, omega)

            #start loop
            rms = 0
            for i in range(no):
                ii = i * no + i
                 
                #commented out error prone component
                self.X1[i] += r1[i] / (eps_occ[i]) #  - eps_lvir[ii].reshape(-1,)) # + omega)
            # self.X2 += r2 / (Dijab + omega)

                rms += contract('a,a->', np.conj(r1[i] / (eps_occ[i])), (r1[i] / (eps_occ[i])))
            # rms += contract('ijab,ijab->', np.conj(r2 / (Dijab + omega)), r2 / (Dijab + omega))
            rms = np.sqrt(rms)
            #end loop

            pseudo = self.local_pseudoresponse(lpertbar, self.X1, self.X2, omega)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")

            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                return self.X1, self.X2, pseudo

            if niter == maxiter:
                print("\nPerturbed wave function not fully converged in %.3f seconds.\n" % (time.time() - solver_start))
                return self.X1, self.X2, pseudo

            #diis.add_error_vector(self.X1, self.X2)
            #if niter >= start_diis:
                #self.X1, self.X2 = diis.extrapolate(self.X1, self.X2)

    def solve_left(self, pertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
        solver_start = time.time()

        Dia = self.Dia
        Dijab = self.Dijab

        # initial guess
        X1_guess = pertbar.Avo.T/(Dia + omega)
        X2_guess = pertbar.Avvoo/(Dijab + omega)
        #print("guess X1", X1_guess)
        #print("guess X2", X2_guess)
        #print("X1 used for inital Y1", self.X1)
        #print("X2 used for initial Y2", self.X2)

        # initial guess
        Y1 = 2.0 * X1_guess.copy()
        Y2 = 4.0 * X2_guess.copy()
        Y2 -= 2.0 * X2_guess.copy().swapaxes(2,3)              
        #print("initial Y1", Y1)
        #print("inital Y2", Y2) 
        # need to understand this
        pseudo = self.pseudoresponse(pertbar, Y1, Y2, )
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")
        
        diis = helper_diis(Y1, Y2, max_diis)
        contract = self.ccwfn.contract

        self.Y1 = Y1
        self.Y2 = Y2 
        
        # uses updated X1 and X2
        self.im_Y1 = self.in_Y1(pertbar, self.X1, self.X2)
        self.im_Y2 = self.in_Y2(pertbar, self.X1, self.X2)
        print("Im_y1 density", np.sqrt(np.einsum('ia, ia ->', self.im_Y1, self.im_Y1)))
        #print("im_Y1", self.im_Y1)
        #print("im_Y2", self.im_Y2)  
        for niter in range(1, maxiter+1):
            pseudo_last = pseudo
            
            Y1 = self.Y1
            Y2 = self.Y2
            
            r1 = self.r_Y1(pertbar, omega)
            r2 = self.r_Y2(pertbar, omega)

            if self.ccwfn.local is not None:
                inc1, inc2 = self.ccwfn.Local.filter_amps(r1, r2)
                self.Y1 += inc1
                self.Y2 += inc2

                rms = contract('ia,ia->', np.conj(inc1 / (Dia + omega)), inc1 / (Dia + omega))
                rms += contract('ijab,ijab->', np.conj(inc2 / (Dijab + omega)), inc2 / (Dijab + omega))
                rms = np.sqrt(rms)
            else:
                self.Y1 += r1 / (Dia + omega)
                self.Y2 += r2 / (Dijab + omega)

                rms = contract('ia,ia->', np.conj(r1 / (Dia + omega)), r1 / (Dia + omega))
                rms += contract('ijab,ijab->', np.conj(r2 / (Dijab + omega)), r2 / (Dijab + omega))
                rms = np.sqrt(rms)

            # need to undertsand this
            pseudo = self.pseudoresponse(pertbar, self.Y1, self.Y2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")
                
            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                return self.Y1, self.Y2 , pseudo

            if niter == maxiter:
                print("\nPerturbed wave function not fully converged in %.3f seconds.\n" % (time.time() - solver_start))
                return self.Y1, self.Y2, pseudo

            diis.add_error_vector(self.Y1, self.Y2)
            if niter >= start_diis:
                self.Y1, self.Y2 = diis.extrapolate(self.Y1, self.Y2)
#    def solve_lleft(self, pertbar, omega, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):

    def r_X1(self, pertbar, omega):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        X1 = self.X1
        X2 = self.X2
        hbar = self.hbar

        r_X1 = (pertbar.Avo.T - omega * X1).copy()
        #r_X1 += contract('ie,ae->ia', X1, hbar.Hvv)
        # print("Canonical r_X1\n", np.linalg.norm(r_X1))
        # r_X1 -= contract('ma,mi->ia', X1, hbar.Hoo)
        # r_X1 += 2.0*contract('me,maei->ia', X1, hbar.Hovvo)
        # r_X1 -= contract('me,maie->ia', X1, hbar.Hovov)
        # r_X1 += contract('me,miea->ia', hbar.Hov, (2.0*X2 - X2.swapaxes(0,1)))
        # r_X1 += contract('imef,amef->ia', X2, (2.0*hbar.Hvovv - hbar.Hvovv.swapaxes(2,3)))
        # r_X1 -= contract('mnae,mnie->ia', X2, (2.0*hbar.Hooov - hbar.Hooov.swapaxes(0,1)))

        return r_X1

    def lr_X1(self, lpertbar, omega):
        contract = self.contract
        no = self.ccwfn.no
        hbar = self.hbar
        Avo = lpertbar.Avo

        lr_X1_all = []
        for i in range(no):
            ii = i*no + i

            # lr_X1 += contract('e, ae -> a', self.X1[i], hbar.Hvv)
            lr_X1 = (Avo[ii] - omega * self.X1[i]).copy()
            # for m in range(no):
            #     # lr_X1 -= contract('ma, mi -> ia', X1, hbar.Hoo)
            #     lr_X1 -= contract('a, i -> ia', self.X1[m], hbar.Hoo[m])
            lr_X1_all.append(lr_X1)

        # lr_X1 += 2.0 * contract('me, maei -> ia', X1, hbar.Hovvo)
        # lr_X1 -= contract('me, maie -> ia', X1, hbar.Hovov)
        # lr_X1 += contract('me, miea -> ia', hbar.Hov, (2.0 * X2 - X2.swapaxes(0, 1)))
        # lr_X1 += contract('imef, amef -> ia', X2, (2.0 * hbar.Hvovv - hbar.Hvovv.swapaxes(2, 3)))
        # lr_X1 -= contract('mnae, mnie -> ia', X2, (2.0 * hbar.Hooov -hbar.Hooov.swapaxes(0, 1)))

        return lr_X1_all

    def r_X2(self, pertbar, omega):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        X1 = self.X1
        X2 = self.X2
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L

        Zvv = contract('amef,mf->ae', (2.0*hbar.Hvovv - hbar.Hvovv.swapaxes(2,3)), X1)
        Zvv -= contract('mnef,mnaf->ae', L[o,o,v,v], X2)

        Zoo = -1.0*contract('mnie,ne->mi', (2.0*hbar.Hooov - hbar.Hooov.swapaxes(0,1)), X1)
        Zoo -= contract('mnef,inef->mi', L[o,o,v,v], X2)

        r_X2 = pertbar.Avvoo - 0.5 * omega*X2
        r_X2 += contract('ie,abej->ijab', X1, hbar.Hvvvo)
        r_X2 -= contract('ma,mbij->ijab', X1, hbar.Hovoo)
        r_X2 += contract('mi,mjab->ijab', Zoo, t2)
        r_X2 += contract('ae,ijeb->ijab', Zvv, t2)
        r_X2 += contract('ijeb,ae->ijab', X2, hbar.Hvv)
        r_X2 -= contract('mjab,mi->ijab', X2, hbar.Hoo)
        r_X2 += 0.5*contract('mnab,mnij->ijab', X2, hbar.Hoooo)
        r_X2 += 0.5*contract('ijef,abef->ijab', X2, hbar.Hvvvv)
        r_X2 -= contract('imeb,maje->ijab', X2, hbar.Hovov)
        r_X2 -= contract('imea,mbej->ijab', X2, hbar.Hovvo)
        r_X2 += 2.0*contract('miea,mbej->ijab', X2, hbar.Hovvo)
        r_X2 -= contract('miea,mbje->ijab', X2, hbar.Hovov)

        r_X2 = r_X2 + r_X2.swapaxes(0,1).swapaxes(2,3)

        return r_X2

    def in_Y1(self, pertbar, X1, X2):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        #X1 = self.X1
        #X2 = self.X2
        Y1 = self.Y1
        
        Y2 = self.Y2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L

        # Inhomogenous terms appearing in Y1 equations
        #seems like these imhomogenous terms are computing at the beginning and not involve in the iteration itself
        #may require moving to a sperate function
 
        # <O|A_bar|phi^a_i> good
        r_Y1 = 2.0 * pertbar.Aov.copy()
        # <O|L1(0)|A_bar|phi^a_i> good
        r_Y1 -= contract('im,ma->ia', pertbar.Aoo, l1)
        r_Y1 += contract('ie,ea->ia', l1, pertbar.Avv)
        # <O|L2(0)|A_bar|phi^a_i>
        r_Y1 += contract('imfe,feam->ia', l2, pertbar.Avvvo)
   
        #can combine the next two to swapaxes type contraction
        r_Y1 -= 0.5 * contract('ienm,mnea->ia', pertbar.Aovoo, l2)
        r_Y1 -= 0.5 * contract('iemn,mnae->ia', pertbar.Aovoo, l2)

        # <O|[Hbar(0), X1]|phi^a_i> good
        r_Y1 +=  2.0 * contract('imae,me->ia', L[o,o,v,v], X1)

        # <O|L1(0)|[Hbar(0), X1]|phi^a_i>
        tmp  = -1.0 * contract('ma,ie->miae', hbar.Hov, l1)
        tmp -= contract('ma,ie->miae', l1, hbar.Hov)
        tmp -= 2.0 * contract('mina,ne->miae', hbar.Hooov, l1)

        #double check this one
        tmp += contract('imna,ne->miae', hbar.Hooov, l1)

       #can combine the next two to swapaxes type contraction
        tmp -= 2.0 * contract('imne,na->miae', hbar.Hooov, l1)
        tmp += contract('mine,na->miae', hbar.Hooov, l1)

        #can combine the next two to swapaxes type contraction
        tmp += 2.0 * contract('fmae,if->miae', hbar.Hvovv, l1)
        tmp -= contract('fmea,if->miae', hbar.Hvovv, l1)

        #can combine the next two to swapaxes type contraction
        tmp += 2.0 * contract('fiea,mf->miae', hbar.Hvovv, l1)
        tmp -= contract('fiae,mf->miae', hbar.Hvovv, l1)
        r_Y1 += contract('miae,me->ia', tmp, X1)

        # <O|L1(0)|[Hbar(0), X2]|phi^a_i> good

        #can combine the next two to swapaxes type contraction
        tmp  = 2.0 * contract('mnef,nf->me', X2, l1)
        tmp  -= contract('mnfe,nf->me', X2, l1)
        r_Y1 += contract('imae,me->ia', L[o,o,v,v], tmp)
        #print("Goo denisty", np.sqrt(np.einsum('ij, ij ->', cclambda.build_Goo(X2, L[o,o,v,v]), cclambda.build_Goo(X2, L[o,o,v,v]))))
        #print("l1 density", np.sqrt(np.einsum('ia, ia ->', l1, l1)))
        r_Y1 -= contract('ni,na->ia', cclambda.build_Goo(X2, L[o,o,v,v]), l1)
        r_Y1 += contract('ie,ea->ia', l1, cclambda.build_Gvv(L[o,o,v,v], X2))

        # <O|L2(0)|[Hbar(0), X1]|phi^a_i> good

        # can reorganize thesenext four to two swapaxes type contraction
        tmp   = -1.0 * contract('nief,mfna->iema', l2, hbar.Hovov)
        tmp  -= contract('ifne,nmaf->iema', hbar.Hovov, l2)
        tmp  -= contract('inef,mfan->iema', l2, hbar.Hovvo)
        tmp  -= contract('ifen,nmfa->iema', hbar.Hovvo, l2)

        #can combine the next two to swapaxes type contraction
        tmp  += 0.5 * contract('imfg,fgae->iema', l2, hbar.Hvvvv)
        tmp  += 0.5 * contract('imgf,fgea->iema', l2, hbar.Hvvvv)

        #can combine the next two to swapaxes type contraction
        tmp  += 0.5 * contract('imno,onea->iema', hbar.Hoooo, l2)
        tmp  += 0.5 * contract('mino,noea->iema', hbar.Hoooo, l2)
        r_Y1 += contract('iema,me->ia', tmp, X1)

       #contains regular Gvv as well as Goo, think about just calling it from cclambda instead of generating it
        tmp  =  contract('nb,fb->nf', X1, cclambda.build_Gvv(l2, t2))
        r_Y1 += contract('inaf,nf->ia', L[o,o,v,v], tmp)
        tmp  =  contract('me,fa->mefa', X1, cclambda.build_Gvv(l2, t2))
        r_Y1 += contract('mief,mefa->ia', L[o,o,v,v], tmp)
        tmp  =  contract('me,ni->meni', X1, cclambda.build_Goo(t2, l2))
        r_Y1 -= contract('meni,mnea->ia', tmp, L[o,o,v,v])
        tmp  =  contract('jf,nj->fn', X1, cclambda.build_Goo(t2, l2))
        r_Y1 -= contract('inaf,fn->ia', L[o,o,v,v], tmp)

        # <O|L2(0)|[Hbar(0), X2]|phi^a_i>
        r_Y1 -= contract('mi,ma->ia', cclambda.build_Goo(X2, l2), hbar.Hov)
        r_Y1 += contract('ie,ea->ia', hbar.Hov, cclambda.build_Gvv(l2, X2))
        tmp   = contract('imfg,mnef->igne', l2, X2)
        r_Y1 -= contract('igne,gnea->ia', tmp, hbar.Hvovv)
        tmp   = contract('mifg,mnef->igne', l2, X2)
        r_Y1 -= contract('igne,gnae->ia', tmp, hbar.Hvovv)
        tmp   = contract('mnga,mnef->gaef', l2, X2)
        r_Y1 -= contract('gief,gaef->ia', hbar.Hvovv, tmp)

        #can combine the next two to swapaxes type contraction
        tmp   = 2.0 * contract('gmae,mnef->ganf', hbar.Hvovv, X2)
        tmp  -= contract('gmea,mnef->ganf', hbar.Hvovv, X2)
        r_Y1 += contract('nifg,ganf->ia', l2, tmp)

        #can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('giea,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))
        r_Y1 += contract('giae,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))
        tmp   = contract('oief,mnef->oimn', l2, X2)
        r_Y1 += contract('oimn,mnoa->ia', tmp, hbar.Hooov)
        tmp   = contract('mofa,mnef->oane', l2, X2)
        r_Y1 += contract('inoe,oane->ia', hbar.Hooov, tmp)
        tmp   = contract('onea,mnef->oamf', l2, X2)
        r_Y1 += contract('miof,oamf->ia', hbar.Hooov, tmp)

        #can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('mioa,mo->ia', hbar.Hooov, cclambda.build_Goo(X2, l2))
        r_Y1 += contract('imoa,mo->ia', hbar.Hooov, cclambda.build_Goo(X2, l2))

        #can combine the next two to swapaxes type contraction
        tmp   = -2.0 * contract('imoe,mnef->ionf', hbar.Hooov, X2)
        tmp  += contract('mioe,mnef->ionf', hbar.Hooov, X2)
        r_Y1 += contract('ionf,nofa->ia', tmp, l2) 

        return r_Y1
 
    def r_lY1(self, pertbar, X1, X2):
        contract = self.contract 
        o = self.ccwfn.o
        v = self.ccwfn.v


    def r_Y1(self, pertbar, omega):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        Y1 = self.Y1 
        Y2 = self.Y2
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L

        #imhomogenous terms
        r_Y1 = self.im_Y1.copy()
        #homogenous terms appearing in Y1 equations
        r_Y1 += omega * Y1
        r_Y1 += contract('ie,ea->ia', Y1, hbar.Hvv)
        r_Y1 -= contract('im,ma->ia', hbar.Hoo, Y1)
        r_Y1 += 2.0 * contract('ieam,me->ia', hbar.Hovvo, Y1)
        r_Y1 -= contract('iema,me->ia', hbar.Hovov, Y1)
        r_Y1 += contract('imef,efam->ia', Y2, hbar.Hvvvo)
        r_Y1 -= contract('iemn,mnae->ia', hbar.Hovoo, Y2)

        #can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('eifa,ef->ia', hbar.Hvovv, cclambda.build_Gvv(t2, Y2))
        r_Y1 += contract('eiaf,ef->ia', hbar.Hvovv, cclambda.build_Gvv(t2, Y2))

        #can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('mina,mn->ia', hbar.Hooov, cclambda.build_Goo(t2, Y2))
        r_Y1 += contract('imna,mn->ia', hbar.Hooov, cclambda.build_Goo(t2, Y2))

        return r_Y1
   
    def in_Y2(self, pertbar, X1, X2):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        #X1 = self.X1
        #X2 = self.X2
        Y1 = self.Y1
        Y2 = self.Y2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L
        ERI = self.H.ERI

        # Inhomogenous terms appearing in Y2 equations
        # <O|L1(0)|A_bar|phi^ab_ij> good

        #next two turn to swapaxes contraction
        r_Y2  = 2.0 * contract('ia,jb->ijab', l1, pertbar.Aov.copy())
        r_Y2 -= contract('ja,ib->ijab', l1, pertbar.Aov)

        # <O|L2(0)|A_bar|phi^ab_ij> good
        r_Y2 += contract('ijeb,ea->ijab', l2, pertbar.Avv)
        r_Y2 -= contract('im,mjab->ijab', pertbar.Aoo, l2)

        # <O|L1(0)|[Hbar(0), X1]|phi^ab_ij> good
        tmp   = contract('me,ja->meja', X1, l1)
        r_Y2 -= contract('mieb,meja->ijab', L[o,o,v,v], tmp)
        tmp   = contract('me,mb->eb', X1, l1)
        r_Y2 -= contract('ijae,eb->ijab', L[o,o,v,v], tmp)
        tmp   = contract('me,ie->mi', X1, l1)
        r_Y2 -= contract('mi,jmba->ijab', tmp, L[o,o,v,v])
        tmp   = 2.0 *contract('me,jb->mejb', X1, l1)
        r_Y2 += contract('imae,mejb->ijab', L[o,o,v,v], tmp)

        # <O|L2(0)|[Hbar(0), X1]|phi^ab_ij> 
        tmp   = contract('me,ma->ea', X1, hbar.Hov)
        r_Y2 -= contract('ijeb,ea->ijab', l2, tmp)
        tmp   = contract('me,ie->mi', X1, hbar.Hov)
        r_Y2 -= contract('mi,jmba->ijab', tmp, l2)
        tmp   = contract('me,ijef->mijf', X1, l2)
        r_Y2 -= contract('mijf,fmba->ijab', tmp, hbar.Hvovv)
        tmp   = contract('me,imbf->eibf', X1, l2)
        r_Y2 -= contract('eibf,fjea->ijab', tmp, hbar.Hvovv)
        tmp   = contract('me,jmfa->ejfa', X1, l2)
        r_Y2 -= contract('fibe,ejfa->ijab', hbar.Hvovv, tmp)

        #swapaxes contraction
        tmp   = 2.0 * contract('me,fmae->fa', X1, hbar.Hvovv)
        tmp  -= contract('me,fmea->fa', X1, hbar.Hvovv)
        r_Y2 += contract('ijfb,fa->ijab', l2, tmp)

        #swapaxes contraction
        tmp   = 2.0 * contract('me,fiea->mfia', X1, hbar.Hvovv)
        tmp  -= contract('me,fiae->mfia', X1, hbar.Hvovv)
        r_Y2 += contract('mfia,jmbf->ijab', tmp, l2)
        tmp   = contract('me,jmna->ejna', X1, hbar.Hooov)
        r_Y2 += contract('ineb,ejna->ijab', l2, tmp)

        tmp   = contract('me,mjna->ejna', X1, hbar.Hooov)
        r_Y2 += contract('nieb,ejna->ijab', l2, tmp)
        tmp   = contract('me,nmba->enba', X1, l2)
        r_Y2 += contract('jine,enba->ijab', hbar.Hooov, tmp)

        #swapaxes
        tmp   = 2.0 * contract('me,mina->eina', X1, hbar.Hooov)
        tmp  -= contract('me,imna->eina', X1, hbar.Hooov)
        r_Y2 -= contract('eina,njeb->ijab', tmp, l2)

        #swapaxes
        tmp   = 2.0 * contract('me,imne->in', X1, hbar.Hooov)
        tmp  -= contract('me,mine->in', X1, hbar.Hooov)
        r_Y2 -= contract('in,jnba->ijab', tmp, l2)

        # <O|L2(0)|[Hbar(0), X2]|phi^ab_ij>
        tmp   = 0.5 * contract('ijef,mnef->ijmn', l2, X2)
        r_Y2 += contract('ijmn,mnab->ijab', tmp, ERI[o,o,v,v])
        tmp   = 0.5 * contract('ijfe,mnef->ijmn', ERI[o,o,v,v], X2)
        r_Y2 += contract('ijmn,mnba->ijab', tmp, l2)
        tmp   = contract('mifb,mnef->ibne', l2, X2)
        r_Y2 += contract('ibne,jnae->ijab', tmp, ERI[o,o,v,v])
        tmp   = contract('imfb,mnef->ibne', l2, X2)
        r_Y2 += contract('ibne,njae->ijab', tmp, ERI[o,o,v,v])
        tmp   = contract('mjfb,mnef->jbne', l2, X2)
        r_Y2 -= contract('jbne,inae->ijab', tmp, L[o,o,v,v])

        #temp intermediate?
        r_Y2 -= contract('in,jnba->ijab', cclambda.build_Goo(L[o,o,v,v], X2), l2)
        r_Y2 += contract('ijfb,af->ijab', l2, cclambda.build_Gvv(X2, L[o,o,v,v]))
        r_Y2 += contract('ijae,be->ijab', L[o,o,v,v], cclambda.build_Gvv(X2, l2))
        r_Y2 -= contract('imab,jm->ijab', L[o,o,v,v], cclambda.build_Goo(l2, X2))
        tmp   = contract('nifb,mnef->ibme', l2, X2)
        r_Y2 -= contract('ibme,mjea->ijab', tmp, L[o,o,v,v])
        tmp   = 2.0 * contract('njfb,mnef->jbme', l2, X2)
        r_Y2 += contract('imae,jbme->ijab', L[o,o,v,v], tmp)

#        r_Y2 = r_Y2 + r_Y2.swapaxes(0,1).swapaxes(2,3)        

        return r_Y2

    def r_Y2(self, pertbar, omega):
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        Y1 = self.Y1
        Y2 = self.Y2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L
        ERI = self.H.ERI

        #inhomogenous terms
        r_Y2 = self.im_Y2.copy()

        # Homogenous terms now!
        # a factor of 0.5 because of the relation/comment just above
        # and due to the fact that Y2_ijab = Y2_jiba
        r_Y2 += 0.5 * omega * self.Y2.copy()
        r_Y2 += 2.0 * contract('ia,jb->ijab', Y1, hbar.Hov)
        r_Y2 -= contract('ja,ib->ijab', Y1, hbar.Hov)
        r_Y2 += contract('ijeb,ea->ijab', Y2, hbar.Hvv)
        r_Y2 -= contract('im,mjab->ijab', hbar.Hoo, Y2)
        r_Y2 += 0.5 * contract('ijmn,mnab->ijab', hbar.Hoooo, Y2)
        r_Y2 += 0.5 * contract('ijef,efab->ijab', Y2, hbar.Hvvvv)
        r_Y2 += 2.0 * contract('ie,ejab->ijab', Y1, hbar.Hvovv)
        r_Y2 -= contract('ie,ejba->ijab', Y1, hbar.Hvovv)
        r_Y2 -= 2.0 * contract('mb,jima->ijab', Y1, hbar.Hooov)
        r_Y2 += contract('mb,ijma->ijab', Y1, hbar.Hooov)
        r_Y2 += 2.0 * contract('ieam,mjeb->ijab', hbar.Hovvo, Y2)
        r_Y2 -= contract('iema,mjeb->ijab', hbar.Hovov, Y2)
        r_Y2 -= contract('mibe,jema->ijab', Y2, hbar.Hovov)
        r_Y2 -= contract('mieb,jeam->ijab', Y2, hbar.Hovvo)
        r_Y2 += contract('ijeb,ae->ijab', L[o,o,v,v], cclambda.build_Gvv(t2, Y2))
        r_Y2 -= contract('mi,mjab->ijab', cclambda.build_Goo(t2, Y2), L[o,o,v,v])

        r_Y2 = r_Y2 + r_Y2.swapaxes(0,1).swapaxes(2,3)

        return r_Y2

    def pseudoresponse(self, pertbar, X1, X2):
        contract = self.ccwfn.contract
        #polar3 = 0 
        for i in range(self.ccwfn.no):
            ii = i*self.ccwfn.no + i  
            QL = self.ccwfn.Local.Q[ii] @ self.ccwfn.Local.L[ii]
            #print("Avo in psuedo", pertbar.Avo[:,i] @ QL)
            #print("X in psuedo", X1[i] @ QL)  
            #polar3 += 2.0 * contract('a,a->', pertbar.Avo[:,i] @ QL, X1[i] @ QL) 
        polar1 = 2.0 * contract('ai,ia->', np.conj(pertbar.Avo), X1)
        # polar2 = 2.0 * contract('ijab,ijab->', np.conj(pertbar.Avvoo), (2.0*X2 - X2.swapaxes(2,3)))
        #print("polar3", polar3)
        return -2.0*(polar1) #+ polar2)

    def local_pseudoresponse(self, lpertbar, X1, X2, omega):
        contract = self.ccwfn.contract
        no = self.no
        Avo = lpertbar.Avo
        polar1 = 0
        norm = 0
        norm_1 = 0
        for i in range(no):
            ii = i * no + i
            #print("Avo in psuedo", ii, Avo[ii]) 
            #print("X in psuedo", ii, X1[i])
            polar1 += 2.0 * contract('a,a->', Avo[ii].copy(), X1[i].copy())
        # polar2 = 2.0 * contract('ijab,ijab->', np.conj(pertbar.Avvoo), (2.0*X2 - X2.swapaxes(2,3)))

        return -2.0*(polar1) #+ polar2)
        
class pertbar(object):
    def __init__(self, pert, ccwfn):
        o = ccwfn.o
        v = ccwfn.v
        t1 = ccwfn.t1
        t2 = ccwfn.t2
        contract = ccwfn.contract

        self.Aov = pert[o,v].copy()
        self.Aoo = pert[o,o].copy()
        self.Aoo += contract('ie,me->mi', t1, pert[o,v])

        self.Avv = pert[v,v].copy()
        self.Avv -= contract('ma,me->ae', t1, pert[o,v])

        self.Avo = pert[v,o].copy()
        self.Avo += contract('ie,ae->ai', t1, pert[v,v])
        self.Avo -= contract('ma,mi->ai', t1, pert[o,o])
        self.Avo += contract('miea,me->ai', (2.0*t2 - t2.swapaxes(2,3)), pert[o,v])
        self.Avo -= contract('ie,ma,me->ai', t1, t1, pert[o,v])

        self.Aovoo = contract('ijeb,me->mbij', t2, pert[o,v])

        self.Avvvo = -1.0*contract('miab,me->abei', t2, pert[o,v])

        # Note that Avvoo is permutationally symmetric, unlike the implementation in ugacc
        self.Avvoo = contract('ijeb,ae->ijab', t2, self.Avv)
        self.Avvoo -= contract('mjab,mi->ijab', t2, self.Aoo)
        self.Avvoo = 0.5*(self.Avvoo + self.Avvoo.swapaxes(0,1).swapaxes(2,3))

        #norm = 0 
        #for ij in range(ccwfn.no*ccwfn.no):
        #    i = ij // ccwfn.no
        #    j = ij % ccwfn.no
        #    ji = j*ccwfn.no + i 
        #    tmp = contract('ab, aA, bB-> AB', self.Avvoo[i,j,:,:], (ccwfn.Local.Q[ij] @ ccwfn.Local.L[ij]), (ccwfn.Local.Q[ij] @ ccwfn.Local.L[ij]))
        #    #tmp1 =  contract('ab, aA, bB-> AB', self.Avvoo[j,i,:,:], (ccwfn.Local.Q[ji] @ ccwfn.Local.L[ji]), (ccwfn.Local.Q[ji] @ ccwfn.Local.L[ji]))  
        #    norm += np.linalg.norm(tmp) #  + tmp1))
        #print("norm of Avvoo", norm)

class lpertbar(object):
    def __init__(self, pert, ccwfn, lccwfn):
        o = ccwfn.o
        v = ccwfn.v
        no = ccwfn.no
        t1 = lccwfn.t1
        t2 = lccwfn.t2
        contract = ccwfn.contract
        QL = ccwfn.Local.QL 
        self.Aov = []
        self.Avv = []
        self.Avo = []
        self.Aovoo = []
        lAvvoo = []
        self.Avvoo = []
        self.Avvvo = []

        self.Aoo = pert[o,o].copy()
        for i in range(no):
            ii = i*no + i
            for m in range(no):
                self.Aoo[m,i] += contract('e,e->',t1[i], (pert[m,v].copy() @ QL[ii]))

        norm = 0 
        for ij in range(no*no):
            i = ij // no
            j = ij % no
            ii = i*no + i
            ji = j*no + i

            #Aov
            self.Aov.append(pert[o,v].copy() @ QL[ij])
            
            #Avv
            tmp = QL[ij].T @ pert[v,v].copy() @ QL[ij]

            Sijmm = ccwfn.Local.Sijmm
            for m in range(no):
                mm = m*no + m
                ijm = ij*no + m
                tmp -= contract('a,e->ae', t1[m] @ Sijmm[ijm].T , pert[m,v].copy() @ QL[ij])
            self.Avv.append(tmp)
            
            #Avo 
            tmp = QL[ij].T @ pert[v,i].copy()
            tmp += t1[i] @ (QL[ij].T @ pert[v,v].copy() @ QL[ii]).T
            
            Sijmi = ccwfn.Local.Sijmi
            for m in range(no):
                mi = m*no + i
                ijm = ij*no + m 
                tmp -= (t1[m] @ Sijmm[ijm].T) * pert[m,i].copy()  
                tmp1 = (2.0*t2[mi] - t2[mi].swapaxes(0,1)) @ Sijmi[ijm].T
                tmp += contract('ea,e->a', tmp1, pert[m,v].copy() @ QL[mi]) 
                tmp -= contract('e,a,e->a', t1[i], t1[m] @ Sijmm[ijm].T, pert[m,v].copy() @ QL[ii]) 
            self.Avo.append(tmp)

            #Aovoo -> Aov_{ij}ij 
            tmp = contract('eb,me->mb',t2[ij], pert[o,v].copy() @ QL[ij])   
            self.Aovoo.append(tmp)
            
            #Avvvo -> Avvvi
            tmp = 0  
            for m in range(no):
                mi = m*no + i
                ijm = ij*no + m 
                tmp -= contract('ab,e->abe', Sijmi[ijm] @ t2[mi] @ Sijmi[ijm].T, pert[m,v] @ QL[ij])      
            self.Avvvo.append(tmp) 
      
            #Avvv_{ii}i 

            #Avvoo -> Aoovv -> Aijv_{ij} V_{ij}
        for i in range(no):
            for j in range(no):
                ij = i*no + j
                ji = j*no + i
                tmp = contract('eb,ae->ab', t2[ij], self.Avv[ij]) 
                Sijmj = ccwfn.Local.Sijmj
                for m in range(no):
                    mj = m*no + j
                    mi = m*no + i 
                    ijm = ij*no + m
                    jim = ji*no + m
                    Sjimi = QL[ji].T @ QL[mi]
                    tmp -= (Sijmj[ijm]  @ t2[mj] @ Sijmj[ijm].T) * self.Aoo[m,i].copy()  
                lAvvoo.append(tmp)    

        norm = 0 
        for i in range(no):
            for j in range(no):
                ij = i*no + j 
                ji = j*no + i 
                self.Avvoo.append(0.5 * (lAvvoo[ij].copy() + lAvvoo[ji].copy().transpose()))
                norm += np.linalg.norm( 0.5 * (lAvvoo[ij].copy() + lAvvoo[ji].copy().transpose()))                  
