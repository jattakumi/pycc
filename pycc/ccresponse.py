"""
ccresponse.py: CC Response Functions
"""
import itertools

from opt_einsum import contract
from scipy.linalg import polar

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import numpy as np
import time
from time import process_time
from .utils import helper_diis

class ccresponse(object):
    """
    An RHF-CC Response Property Object.

    Methods
    -------
    linresp():
        Compute a CC linear response function.
    quadresp():
        Compute a CC quadratic response function.
    hyperpolar():
        Compute a first electric dipole hyperpolarizability average. 
    solve_right():
        Solve the right-hand perturbed wave function equations.
    solve_left(): 
        Solve the left-hand perturbed wave function equations.
    pertcheck():
        Check first-order perturbed wave functions for all available perturbation operators.
    pert_quadresp():
        Obtain the solutions of the right- and left-hand perturbed wave function equations for the CC quadritc response function. 
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
        # self.dim = self.lccwfn.Local.dim
        self.l1 = self.cclambda.l1
        # self.l2 = self.cclambda.l2
        self.no = self.ccwfn.no
        self.psuedoresponse = []

        if self.ccwfn.local is not None and self.ccwfn.filter is not True: 
            self.lccwfn = ccdensity.lccwfn
            self.cclambda = ccdensity.cclambda
            self.H = self.ccwfn.H
            self.cchbar = self.cclambda.hbar
            self.contract = self.ccwfn.contract
            self.no = self.ccwfn.no
            self.Local = self.ccwfn.Local
            self.dim = self.lccwfn.Local.dim

            # initialize variables for timing each linear response terms
            # self.sym_lr = 0
            # self.LCX_t = 0
            # self.LHX1Y1_t = 0
            self.quad_terms = 0
            self.llcx_t = 0
            self.lsym_lr_t = 0
            self.lLCX_t = 0
            self.lLHX1Y1_t = 0
            self.lLHX2Y2_t = 0
            self.lLHX1Y2_t = 0

            self.lX1_t = 0
            self.lX2_t = 0
            self.lY1_t = 0
            self.lY2_t = 0
            self.lpseudoresponse_t = 0
            self.pseudoresponse_t = 0

            #Q = self.Local.Q
            #L = []
            #self.QL = []
            self.eps_occ = np.diag(self.cchbar.Hoo)
            self.eps_lvir = []
            for i in range(self.ccwfn.no):
                for j in range(self.ccwfn.no):
                    ij = i*self.ccwfn.no + j
                    #print(self.cchbar.Hvv[ij].shape, Q[ij].shape)
                    #eval, evec = np.linalg.eigh(self.cchbar.Hvv[ij])
                    #self.QL.append(Q[ij] @ evec)
                    #self.eps_lvir.append(eval)
                    self.eps_lvir.append(np.diag(self.cchbar.Hvv[ij])) 

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
                # self.dim = self.lccwfn.Local.dim

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
            if self.ccwfn.filter is True or self.ccwfn.filter is False:
                self.eps_occ = np.diag(self.hbar.Hoo)
                eps_vir = np.diag(self.hbar.Hvv)
                self.Dia = self.eps_occ.reshape(-1,1) - eps_vir
                self.Dijab = self.eps_occ.reshape(-1,1,1,1) + self.eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        #HBAR-based denominators for simulation code 
        if self.ccwfn.filter is True and self.ccwfn.local is not None:
            self.eps_vir = []
            for ij in range(self.ccwfn.no*self.ccwfn.no):
                tmp = self.ccwfn.Local.Q[ij].T @ self.hbar.Hvv @ self.ccwfn.Local.Q[ij]
                self.eps_vir.append(np.diag(self.ccwfn.Local.L[ij].T @ tmp @ self.ccwfn.Local.L[ij])) 
                #self.Dia = eps_occ.reshape(-1,1) #- eps_vir
                #self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) #- eps_vir.reshape(-1,1) - eps_vir

        #HBAR-based denominators for simulation code 
       # if self.ccwfn.filter is not True and self.ccwfn.local is not None:
       #     self.eps_occ = np.diag(self.hbar.Hoo)
       #     self.eps_vir = []
       #     for ij in range(self.ccwfn.no*self.ccwfn.no):
       #         tmp = self.ccwfn.Local.Q[ij].T @ self.hbar.Hvv @ self.ccwfn.Local.Q[ij]
       #         self.eps_vir.append(np.diag(self.ccwfn.Local.L[ij].T @ tmp @ self.ccwfn.Local.L[ij]))

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

    def sym_linresp(self, pertkey_a, pertkey_b, X1_A, X2_A, X1_B, X2_B):
        sym_lr_start = process_time()
        """
            Calculate the CC symmetric linear response function for polarizability at field-frequency omega(w1).

            The linear response function, <<A;B(w1)>> generally requires the following perturbed wave functions and frequencies:

            Parameters
            ----------
            pertkey_a: string
                String identifying the one-electron perturbation, A along a cartesian axis

            Return
            ------
            polar: float
                 A value of the chosen linear response function corresponding to compute polariazabiltity in a specified cartesian diresction.
        """

        # Please refer to eqn 94 of [Koch and Jørgensen, “Coupled Cluster Response Functions.”].
        # Writing H(1)(omega) = B, T^(1)(omega) = X, Lambda = L^(0)
        # <<A;B>> = <0|(1 + L^(0)) { [\bar{A}^(0), X^(1)(B)] + [\bar{B}^(1), X^(1)(B)] + [[\bar{B}^(0), X^(1)(B)], X^(1)(B)] } |0>

        contract = self.ccwfn.contract
        l2 = self.cclambda.l2
        t2 = self.ccwfn.t2
        o = self.ccwfn.o
        v = self.ccwfn.v
        L = self.ccwfn.H.L

        polar1 = 0.0

        polar1 += self.LCX(pertkey_a, X1_B, X2_B)
        polar1 += self.LCX(pertkey_b, X1_A, X2_A)

        # <0|(HX1Y1)|0>
        LHX1Y1 = 2.0 * contract('ijab, ia, jb', L[o, o, v, v], X1_B, X1_A)
        Goo = contract('mjab,ijab->mi', t2, l2)
        Gvv = -1.0 * contract('ijeb,ijab->ae', t2, l2)
        r2_Gvv = contract('ae,ijeb->ijab', Gvv, L[o, o, v, v])
        r2_Goo = -1.0 * contract('mi,mjab->ijab', Goo, L[o, o, v, v])
        r2_Gvv = r2_Gvv + r2_Gvv.swapaxes(0, 1).swapaxes(2, 3)
        r2_Goo = r2_Goo + r2_Goo.swapaxes(0, 1).swapaxes(2, 3)
        LHX1Y1 += contract('ijab,ia,jb', r2_Gvv, X1_A, X1_B)  # Gvv
        LHX1Y1 += contract('ijab,ia,jb', r2_Goo, X1_A, X1_B)  # Goo
        polar1 += self.LHX1Y1(X1_B, X1_A)
        polar1 += self.LHX1Y1(X1_A, X1_B)
        # #
        # # # <0|L2[[H, X2], Y2]|0>
        temp = contract("ikac, ijab -> kjbc", L[o, o, v, v], X2_A)
        temp = contract("kjbc, klcd -> jlbd", temp, X2_B)
        LHX2Y2 = 2 * contract("jlbd, jlbd -> ", temp, l2)
        polar1 += self.LHX2Y2(X2_A, X2_B)
        polar1 += self.LHX2Y2(X2_B, X2_A)
        # #
        polar1 += self.LHX1Y2(X1_B, X2_A)
        polar1 += self.LHX1Y2(X1_A, X2_B)
        # #

        polar1 += LHX1Y1 + LHX2Y2
        sym_lr_end = process_time()
        self.sym_lr = sym_lr_end - sym_lr_start

        return -1.0 * polar1

    def LCX(self, pertkey_a, X1_B, X2_B):
        LCX_start = process_time()
        """
        LCX: Is the first second term contributions to the linear response symmetric function.
        <0|(1 + L^(0)){ [\bar{A}^(0), X^(1)(B)] + [\bar{B}^(1), X^(1)(B)]}|0>
        """
        contract = self.ccwfn.contract
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2

        LCX = 0.0
        pertbar_A = self.pertbar[pertkey_a]

        Aov = pertbar_A.Aov
        Aoo = pertbar_A.Aoo
        Avv = pertbar_A.Avv
        Avvvo = pertbar_A.Avvvo
        Aovoo = pertbar_A.Aovoo

        # <0|[A_bar, X1]|0>
        LCX += 2.0 * contract("ia, ia -> ", Aov, X1_B)

        # <0|L1 [A_bar, X1]|0>
        temp_ov = contract('ab, ib -> ia', Avv, X1_B)
        temp_ov -= contract('ji, ja -> ia', Aoo, X1_B)
        LCX += contract('ia, ia', temp_ov, l1)
        #
        temp_ov = -1.0 * contract('ja, ijab -> ib', Aov, X2_B)
        temp_ov += 2.0 * contract('ja, jiab -> ib', Aov, X2_B)
        LCX += contract("ib, ib -> ", temp_ov, l1)
        #
        # <0|L2 [A_bar, X1]|0>
        tmp = contract("ijbc, bcaj -> ia", l2, Avvvo)
        LCX += contract("ia, ia -> ", tmp, X1_B)
        tmp = contract("ijab, kbij -> ak", l2, Aovoo)
        LCX -= 0.5 * contract("ak, ka -> ", tmp, X1_B)
        tmp = contract("ijab, kaji -> bk", l2, Aovoo)
        LCX -= 0.5 * contract("bk, kb -> ", tmp, X1_B)

        tmp = contract("ijab, kjab -> ik", l2, X2_B)
        LCX -= 0.5 * contract("ik, ki -> ", tmp, Aoo)
        tmp = contract("ijab, kiba-> jk", l2, X2_B)
        LCX -= 0.5 * contract("jk, kj -> ", tmp, Aoo)
        tmp = contract("ijab, ijac -> bc", l2, X2_B)
        LCX += 0.5 * contract("bc, bc -> ", tmp, Avv)
        tmp = contract("ijab, ijcb -> ac", l2, X2_B)
        LCX += 0.5 * contract("ac, ac -> ", tmp, Avv)

        LCX_end = process_time()
        self.LCX_t = LCX_end - LCX_start
        return LCX

    def LHX1Y1(self, X1_B, X1_A):
        LHX1Y1_start = process_time()
        """
        LHX1Y1: Is a function for the second term (quadratic term) contribution to the linear response
        function, where both perturbed amplitudes are first order.
        # <0|(1 + L^(0)){ [[\bar{B}^(0), X^(1)(B)], X^(1)(B)] }|0>
        """

        contract = self.ccwfn.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.ccwfn.H.L

        LHX1Y1 = 0.0

        # <0|L1[[H, X1], Y1]|0>
        tmp = contract('ja, ia -> ij', hbar.Hov, X1_A)
        tmp = contract('ij, jb -> ib', tmp, X1_B)
        LHX1Y1 -= contract('ib, ib', tmp, l1)

        tmp = contract('jika, ia -> jk', 2 * hbar.Hooov, X1_B)
        tmp = contract('jk, jb -> kb', tmp, X1_A)
        LHX1Y1 -= contract('kb, kb', tmp, l1)
        tmp = contract('jika, ia -> jk', hbar.Hooov.swapaxes(0, 1), X1_B)
        tmp = contract('jk, jb -> kb', tmp, X1_A)
        LHX1Y1 += contract('kb, kb', tmp, l1)

        tmp = contract('cjab, jb -> ac', 2 * hbar.Hvovv, X1_A)
        tmp = contract('ac, ia -> ic', tmp, X1_B)
        LHX1Y1 += contract('ic, ic -> ', tmp, l1)
        tmp = contract('cjab, jb -> ac', hbar.Hvovv.swapaxes(2, 3), X1_A)
        tmp = contract('ac, ia -> ic', tmp, X1_B)
        LHX1Y1 -= contract('ic, ic -> ', tmp, l1)
        #
        # # <0|L2[[H, X1], Y1]|0>
        temp = contract("jcka, ia -> ijkc", hbar.Hovov, X1_A)
        temp = contract("ijkc, jb -> kibc", temp, X1_B)
        LHX1Y1 -= contract("kibc, kibc -> ", temp, l2)
        temp = contract("jcak, ia -> ijkc", hbar.Hovvo, X1_A)
        temp = contract("ijkc, jb -> kicb", temp, X1_B)
        LHX1Y1 -= contract("kicb, kicb -> ", temp, l2)
        temp = contract('cdab, ia -> ibcd', hbar.Hvvvv, X1_B)
        temp = contract('ibcd, jb -> ijcd', temp, X1_A)
        LHX1Y1 += 0.5 * contract('ijcd, ijcd', temp, l2)
        temp = contract('ijkl, ia -> klaj', hbar.Hoooo, X1_B)
        temp = contract('klaj, jb -> klab', temp, X1_A)
        LHX1Y1 += 0.5 * contract('klab, klab', temp, l2)

        LHX1Y1_end = process_time()
        self.LHX1Y1_t = LHX1Y1_end - LHX1Y1_start

        return LHX1Y1

    def LHX2Y2(self, X2_A, X2_B):
        LHX2Y2_start = process_time()
        """
        LHX2Y2: Is a function for the second term (quadratic term) contribution to the linear response
        function, where both perturbed amplitudes are second order.
        # <0|(1 + L^(0)){ [[\bar{B}^(0), X^(1)(B)], X^(1)(B)] }|0>
        """

        contract = self.ccwfn.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.ccwfn.H.L
        ERI = self.ccwfn.H.ERI

        LHX2Y2 = 0.0
        temp = contract("ijac, ijab -> bc", L[o, o, v, v], X2_A)
        temp = contract("bc, klcd -> klbd", temp, X2_B)
        LHX2Y2 -= contract("klbd, klbd -> ", temp, l2)
        temp = contract("ijac, ikac -> jk", L[o, o, v, v], X2_A)
        temp = contract("jk, jlbd -> klbd", temp, X2_B)
        LHX2Y2 -= contract("klbd, klbd -> ", temp, l2)
        temp = contract("ijac, jkbc -> ikab", L[o, o, v, v], X2_A)
        temp = contract("ikab, ilad -> klbd", temp, X2_B)
        LHX2Y2 -= contract("klbd, klbd -> ", temp, l2)
        temp = contract("klab, ijab -> klij", ERI[o, o, v, v].copy(), X2_A)
        temp = contract("klij, klcd -> ijcd", temp, X2_B)
        LHX2Y2 += 0.5 * contract("ijcd, ijcd -> ", temp, l2)
        temp = contract("klab, ikac -> ilbc", ERI[o, o, v, v].copy(), X2_A)
        temp = contract("ilbc, jlbd -> ijcd", temp, X2_B)
        LHX2Y2 += 0.5 * contract("ijcd, ijcd -> ", temp, l2)
        temp = contract("klab, ilad -> ikbd", ERI[o, o, v, v].copy(), X2_A)
        temp = contract("ikbd, jkbc -> ijcd", temp, X2_B)
        LHX2Y2 += 0.5 * contract("ijcd, ijcd -> ", temp, l2)

        LHX2Y2_end = process_time()
        self.LHX2Y2_t = LHX2Y2_end - LHX2Y2_start
        return LHX2Y2

    def LHX1Y2(self, X1_B, X2_A):
        LHX1Y2_start = process_time()
        """
        LHX1Y2: Is a function for the second term (quadratic term) contribution to the linear response
        function, where perturbed amplitudes is a first order and a second order.
        # <0|(1 + L^(0)){ [[\bar{B}^(0), X^(1)(B)], X^(1)(B)] }|0>
        """

        contract = self.ccwfn.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        hbar = self.hbar
        L = self.ccwfn.H.L

        LHX1Y2 = 0.0
        # <O|L1(0)[[Hbar(0),X1(B)],X2(A)]]|0>
        temp = contract("ijac, ia -> jc", L[o, o, v, v], X1_B)
        temp = contract("jc, jkbc -> kb", temp, X2_A)
        LHX1Y2 -= contract("kb, kb", temp, l1)
        temp = contract("ijac, ia -> jc", L[o, o, v, v], X1_B)
        temp = contract("jc, jkcb -> kb", temp, X2_A)
        LHX1Y2 += 2.0 * contract("kb, kb", temp, l1)
        temp = contract("ijac, ikac -> jk", L[o, o, v, v], X2_A)
        temp = contract("jk, jb -> kb", temp, X1_B)
        LHX1Y2 -= contract("kb, kb -> ", temp, l1)
        temp = contract("ijac, ijab -> bc", L[o, o, v, v], X2_A)
        temp = contract("bc, kc -> kb", temp, X1_B)
        LHX1Y2 -= contract("kb, kb -> ", temp, l1)

        # # <O|L2(A)[[Hbar(0),X2(B)],X1(C)]]|0>
        temp = contract('ja, ia -> ij', hbar.Hov, X1_B)
        temp = contract('ij, jkbc -> ikbc', temp, X2_A)
        LHX1Y2 -= contract('ikbc, ikbc', temp, l2)
        temp = contract('ja, jb -> ab', hbar.Hov, X1_B)
        temp = contract('ab, ikac -> ikbc', temp, X2_A)
        LHX1Y2 -= contract('ikbc, ikbc', temp, l2)

        temp = contract('jima, ia -> jm', (2 * hbar.Hooov - hbar.Hooov.swapaxes(0, 1)), X1_B)
        temp = contract('jm, jkbc -> kmcb', temp, X2_A)
        LHX1Y2 -= contract('kmcb, kmcb -> ', temp, l2)
        temp = contract('jima, jb -> imab', (2 * hbar.Hooov - hbar.Hooov.swapaxes(0, 1)), X1_B)
        temp = contract('imab, ikac -> kmcb', temp, X2_A)
        LHX1Y2 -= contract('kmcb, kmcb', temp, l2)
        #
        temp = contract('djab, ia -> djib', (2 * hbar.Hvovv - hbar.Hvovv.swapaxes(2, 3)), X1_B)
        temp = contract('djib, jkbc -> kicd', temp, X2_A)
        LHX1Y2 += contract('kicd, kicd', temp, l2)
        temp = contract('djab, jb -> ad', (2 * hbar.Hvovv - hbar.Hvovv.swapaxes(2, 3)), X1_B)
        temp = contract('ad, ikac -> kicd', temp, X2_A)
        LHX1Y2 += contract('kicd, kicd', temp, l2)
        #
        temp = contract('dkab, ia -> ikdb', hbar.Hvovv, X1_B)
        temp = contract('ikdb, jkbc -> ijdc', temp, X2_A)
        LHX1Y2 -= contract('ijdc, ijdc', temp, l2)
        temp = contract('dkab, jb -> kjad', hbar.Hvovv, X1_B)
        temp = contract('kjad, ikac -> ijdc', temp, X2_A)
        LHX1Y2 -= contract('ijdc, ijdc', temp, l2)
        temp = contract('dkab, kc -> abcd', hbar.Hvovv, X1_B)
        temp = contract('abcd, ijab -> ijdc', temp, X2_A)
        LHX1Y2 -= contract('ijdc, ijdc', temp, l2)
        #
        temp = contract('jkla, ia -> ijkl', hbar.Hooov, X1_B)
        temp = contract('ijkl, jkbc -> libc', temp, X2_A)
        LHX1Y2 += contract('libc, libc', temp, l2)
        temp = contract('jkla, jb -> klab', hbar.Hooov, X1_B)
        temp = contract('klab, ikac -> libc', temp, X2_A)
        LHX1Y2 += contract('libc, libc -> ', temp, l2)
        temp = contract('jkla, kc -> jlac', hbar.Hooov, X1_B)
        temp = contract('jlac, ijab -> libc', temp, X2_A)
        LHX1Y2 += contract('libc, libc -> ', temp, l2)

        LHX1Y2_end = process_time()
        self.LHX1Y2_t = LHX1Y2_end - LHX1Y2_start
        return LHX1Y2

# Below are all the local parts of lin_resp sym
    def lsym_linresp(self, pertkey_a, pertkey_b, X1_A, X2_A, X1_B, X2_B):
        # lsym_lr_start = process_time()
        """
            Calculate the CC symmetric linear response function for polarizability at field-frequency omega(w1).

            The linear response function, <<A;B(w1)>> generally requires the following perturbed wave functions and frequencies:

            Parameters
            ----------
            pertkey_a: string
                String identifying the one-electron perturbation, A along a cartesian axis

            Return
            ------
            polar: float
                 A value of the chosen linear response function corresponding to compute polariazabiltity in a specified cartesian diresction.
        """

        # Please refer to eqn 94 of [Koch and Jørgensen, “Coupled Cluster Response Functions.”].
        # Writing H(1)(omega) = B, T^(1)(omega) = X, Lambda = L^(0)
        # <<A;B>> = <0|(1 + L^(0)) { [\bar{A}^(0), X^(1)(B)] + [\bar{B}^(1), X^(1)(B)] + [[\bar{B}^(0), X^(1)(B)], X^(1)(B)] } |0>

        contract = self.ccwfn.contract
        l2 = self.cclambda.l2
        t2 = self.lccwfn.t2
        no = self.ccwfn.no
        o = self.ccwfn.o
        v = self.ccwfn.v
        L = self.ccwfn.H.L
        QL = self.Local.QL
        Sijmn = self.Local.Sijmn

        polar1 = 0.0

        polar1 += self.lLCX(pertkey_a, X1_B, X2_B)
        polar1 += self.lLCX(pertkey_b, X1_A, X2_A)

        # # # <0|(HX1Y1)|0>
        # # LHX1Y1 = 2.0 * contract('ijab, ia, jb', L[o, o, v, v], X1_B, X1_A)
        lhx1y1_start = process_time()
        LHX1Y1 = 0
        for i in range(no):
            for j in range(no):
                ii = i * no + i
                jj = j * no + j
                LHX1Y1 += 2.0 * contract('ab, a, b', QL[ii].T @ L[i, j, v, v] @ QL[jj], X1_B[i], X1_A[j])
        #
        # # self.cclambda.build_lGoo(t2, l2) ...
        # # Goo = contract('mjab,ijab->mi', t2, l2)
        # # self.cclambda.build_lGvv(t2, l2) ... Gv_[mn]v_[mn]
        # # Gvv = -1.0 * contract('ijeb,ijab->ae', t2, l2)
        # # r2_Gvv = contract('ae,ijeb->ijab', Gvv, L[o, o, v, v])
        # # r2_Goo = -1.0 * contract('mi,mjab->ijab', Goo, L[o, o, v, v])
        # # r2_Gvv = r2_Gvv + r2_Gvv.swapaxes(0, 1).swapaxes(2, 3)
        # # r2_Goo = r2_Goo + r2_Goo.swapaxes(0, 1).swapaxes(2, 3)
        # # LHX1Y1 += contract('ijab,ia,jb', r2_Gvv, X1_A, X1_B)  # Gvv
        # # LHX1Y1 += contract('ijab,ia,jb', r2_Goo, X1_A, X1_B)  # Goo
        # #ija_{mn}b_{jj} +  jib{jj}a_{mn}
        Gvv = self.cclambda.build_lGvv(t2, l2)
        Goo = self.cclambda.build_lGoo(t2, l2)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                jj = j * no + j
                ij = i * no + j


                for m in range(no):
                    mj = m * no + j
                    mjjj = mj * (no * no) + jj
                    ijjj = ij * (no * no) + jj
                    mjii = mj * (no * no) + ii
                    ijjj = ij * (no * no) + jj
                    iiij = ii * (no * no) + ij

                    for n in range(no):
                        mn = m*no + n
                        iimn = ii*(no*no) + mn
                        jjmn = jj*(no*no) + mn

                        r_Gvv = contract('ae, eb ->ab', Sijmn[iimn] @ Gvv[mn], QL[mn].T @ L[i,j,v,v] @ QL[jj])
                        r_Gvvs = contract('be, ea ->ba', Sijmn[jjmn] @ Gvv[mn], QL[mn].T @ L[j,i,v,v] @ QL[ii])
                        r_Gvv = r_Gvv + r_Gvvs.swapaxes(0,1)
                        tmp = contract('ab, a->b', r_Gvv, X1_A[i])
                        LHX1Y1 += contract('b,b->', tmp, X1_B[j])

                    r_Goo = -1.0 * contract(' , ab->ab', Goo[m, i], QL[ii].T @ L[m, j, v, v] @ QL[jj])
                    r_Goos = -1.0 * contract(' , ba->ba', Goo[m, j], QL[jj].T @ L[m, i, v, v] @ QL[ii])
                    r_Goo = r_Goo + r_Goos.swapaxes(0, 1)
                    tmp = contract('ab, a->b', r_Goo, X1_A[i])
                    LHX1Y1 += contract('b,b->', tmp, X1_B[j])
        lhx1y1_end = process_time()
        self.lLHX1Y1_t += lhx1y1_end - lhx1y1_start
        polar1 += self.lLHX1Y1(X1_B, X1_A)
        polar1 += self.lLHX1Y1(X1_A, X1_B)
        # #
        # # # <0|L2[[H, X2], Y2]|0>
        # # temp = contract("ikac, ijab -> kjbc", L[o, o, v, v], X2_A)
        # # temp = contract("kjbc, klcd -> jlbd", temp, X2_B)
        # # LHX2Y2 = 2 * contract("jlbd, jlbd -> ", temp, l2)
        lhx2y2_start = process_time()
        LHX2Y2 = 0
        for i in range(no):
            for j in range(no):
                ij = i * no + j
                for k in range(no):
                    for l in range(no):
                        kl = k * no + l
                        jl = j * no + l
                        ijjl = ij * (no * no) + jl
                        jlkl = jl * (no * no) + kl
                        temp = contract('ac, ab -> bc', QL[ij].T @ L[i, k, v, v] @ QL[kl], X2_A[ij])
                        temp = contract('bc, cd -> bd', temp, X2_B[kl])
                        LHX2Y2 += 2.0 * contract('bd, bd', temp, Sijmn[ijjl] @ l2[jl] @ Sijmn[jlkl])
        lhx2y2_end = process_time()
        self.lLHX2Y2_t += lhx2y2_end - lhx2y2_start
        polar1 += self.lLHX2Y2(X2_A, X2_B)
        polar1 += self.lLHX2Y2(X2_B, X2_A)
        # #
        polar1 += self.lLHX1Y2(X1_B, X2_A)
        polar1 += self.lLHX1Y2(X1_A, X2_B)
        # #
        polar1 += LHX1Y1 + LHX2Y2
        lsym_lr_end = process_time()
        # self.lsym_lr_t += lsym_lr_end - lsym_lr_start
        return -1.0 * polar1

    def lLCX(self, pertkey_a, X1_B, X2_B):
        llcx_start = process_time()
        """
        LCX: Is the first second term contributions to the linear response symmetric function.
        <0|(1 + L^(0)){ [\bar{A}^(0), X^(1)(B)] + [\bar{B}^(1), X^(1)(B)]}|0>
        """
        contract = self.ccwfn.contract
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        Sijmn = self.Local.Sijmn
        LCX = 0.0
        pertbar_A = self.lpertbar[pertkey_a]
        no = self.ccwfn.no

        for i in range(no):
            ii = i * no + i
            # <0|[A_bar, X1]|0>
            LCX += 2.0 * contract('a, a', pertbar_A.Aov[ii][i], X1_B[i])
            # # <0|L1 [A_bar, X1]|0>
            temp = contract('ab, b -> a', pertbar_A.Avv[ii].copy(), X1_B[i])
            for j in range(no):
                jj = j * no + j
                iijj = ii*(no*no) + jj
                temp -= pertbar_A.Aoo[j,i] * (Sijmn[iijj] @ X1_B[j])
            LCX += contract('a, a', temp, l1[i])
        for i in range(no):
            ii = i*no + i
            temp_ov = 0
            for j in range(no):
                jj = j*no + j
                ij = i*no + j
                ji = j*no + i
                jjij = jj*(no*no) + ij
                jjji = jj*(no*no) + ji
                ijii = ij*(no*no) + ii
                jiii = ji*(no*no) + ii
                #Aov[pair slice] <- direct pair slice based on amplitude
                temp_ov -= 1.0 * contract('a, ab -> b', pertbar_A.Aov[ij][j], X2_B[ij] @ Sijmn[ijii])
                temp_ov += 2.0 * contract('a, ab -> b', pertbar_A.Aov[ji][j], X2_B[ji] @ Sijmn[jiii])
            LCX += contract('b, b', temp_ov, l1[i])

        # <0|L2 [A_bar, X1]|0>
        for i in range(no):
            for j in range(no):
                ij = i * no + j
                temp = contract('bc, bca -> a', l2[ij], pertbar_A.Avvvj_ii[ij])
                LCX += contract('a, a', temp, X1_B[i])

                temp = contract('ab, ac -> bc', l2[ij], X2_B[ij])
                LCX += 0.5 * contract('bc, bc ->', temp, pertbar_A.Avv[ij].copy())
                temp = contract('ab, cb -> ac', l2[ij], X2_B[ij])
                LCX += 0.5 * contract('ac, ac ->', temp, pertbar_A.Avv[ij].copy())
                for k in range(no):
                   kk = k * no + k
                   kj = k * no + j
                   ki = k * no + i
                   ijkj = ij*(no*no) + kj
                   kjij = kj * (no * no) + ij
                   ijki = ij*(no*no) + ki
                   ijkk = ij * (no * no) + kk
                   kiij = ki*(no*no) + ij

                   temp = contract('ab, b -> a', l2[ij], pertbar_A.Aovoo[ij][k])
                   LCX -= 0.5 * contract('a, a', temp, X1_B[k] @ Sijmn[ijkk].T)
                   temp = contract('ab, a -> b', l2[ij], pertbar_A.Aovoo_ji[ij][k])
                   LCX -= 0.5 * contract('b, b', temp, X1_B[k] @ Sijmn[ijkk].T)

                   temp = contract('ab, ab -> ', Sijmn[kjij] @ l2[ij] @ Sijmn[ijkj], X2_B[kj])
                   LCX -= 0.5 * temp * pertbar_A.Aoo[k, i]
                   temp = contract('ab, ba -> ', Sijmn[kiij] @ l2[ij] @ Sijmn[ijki], X2_B[ki])
                   LCX -= 0.5 * temp * pertbar_A.Aoo[k, j]
        llcx_end = process_time()
        self.lLCX_t += llcx_end - llcx_start
        return LCX

    def lLHX1Y1(self, X1_B, X1_A):
        llhx1y1_start = process_time()
        """
        LHX1Y1: Is a function for the second term (quadratic term) contribution to the linear response
        function, where both perturbed amplitudes are first order.
        # <0|(1 + L^(0)){ [[\bar{B}^(0), X^(1)(B)], X^(1)(B)] }|0>
        """

        contract = self.ccwfn.contract
        no = self.ccwfn.no
        v = self.ccwfn.v
        t1 = self.lccwfn.t1
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        t2 = self.lccwfn.t2
        hbar = self.hbar
        L = self.ccwfn.H.L
        ERI = self.ccwfn.H.ERI
        Sijmn = self.Local.Sijmn
        QL = self.Local.QL

        LHX1Y1 = 0.0

        # # <0|L1[[H, X1], Y1]|0>
        # tmp = contract('ja, ia -> ij', hbar.Hov, X1_A)
        # tmp = contract('ij, jb -> ib', tmp, X1_B)
        # LHX1Y1 -= contract('ib, ib', tmp, l1)
        temp = 0
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                jj = j * no + j
                iijj = ii * (no * no) + jj
                temp = contract('a, a -> ', hbar.Hov[ii][j], X1_A[i])
                temp = temp * X1_B[j]
                LHX1Y1 -= contract('b, b', temp, Sijmn[iijj].T @ l1[i])


        # tmp = contract('jika, ia -> jk', 2 * hbar.Hooov, X1_B)
        # tmp = contract('jk, jb -> kb', tmp, X1_A)
        # LHX1Y1 -= contract('kb, kb', tmp, l1)
        # used to be jika
        # tmp = contract('jika, ia -> jk', hbar.Hooov.swapaxes(0, 1), X1_B)
        # tmp = contract('jk, jb -> kb', tmp, X1_A)
        # LHX1Y1 += contract('kb, kb', tmp, l1)
        for i in range(no):
            ii = i*no + i
            for j in range(no):
                jj = j*no + j
                for k in range(no):
                    kk = k*no +k
                    jjkk = jj*(no*no) + kk
                    #jika_{ii}
                    Hjika = self.Local.ERIooov[ii][j, i, k, :].copy()

                    tmp = contract('eE, ef -> Ef', QL[ii], self.ccwfn.H.ERI[i, j, v, v])
                    tmp = contract('fF, Ef -> EF', QL[kk], tmp)
                    Hjika = Hjika + contract('f,ef->e', t1[k], tmp)

                    temp = contract('a, a -> ', 2 * Hjika, X1_B[i])
                    temp = contract(', b -> b', temp, X1_A[j])
                    LHX1Y1 -= contract('b, b', temp, Sijmn[jjkk] @ l1[k])

                    Hjika_swap = self.Local.ERIooov[ii][i, j, k, :].copy()
                    tmp = contract('eE, ef -> Ef', QL[ii], self.ccwfn.H.ERI[j, i, v, v])
                    tmp = contract('fF, Ef -> EF', QL[kk], tmp)
                    Hjika_swap = Hjika_swap + contract('f,ef->e', t1[k], tmp)

                    temp = contract('a, a -> ', Hjika_swap, X1_B[i])
                    temp = contract(', b -> b', temp, X1_A[j])
                    LHX1Y1 += contract('b, b', temp, Sijmn[jjkk] @ l1[k])

        # tmp = contract('cjab, jb -> ac', 2 * hbar.Hvovv, X1_A)
        # tmp = contract('ac, ia -> ic', tmp, X1_B)
        # LHX1Y1 += contract('ic, ic -> ', tmp, l1)
        # tmp = contract('cjab, jb -> ac', hbar.Hvovv.swapaxes(2, 3), X1_A)
        # tmp = contract('ac, ia -> ic', tmp, X1_B)
        # LHX1Y1 -= contract('ic, ic -> ', tmp, l1)
        # LHX1Y1 = 0
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                iij = ii*no + j

                tmp = contract('cab, b -> ac', 2.0 * hbar.Hamef[iij], X1_A[j])
                tmp = contract('ac, a ->c', tmp, X1_B[i])
                LHX1Y1 += contract('c,c->', tmp, l1[i])
                tmp = contract('cba, b -> ac', hbar.Hamfe[iij], X1_A[j])
                tmp = contract('ac, a ->c', tmp, X1_B[i])
                LHX1Y1 -= contract('c,c->', tmp, l1[i])
        #
        # # <0|L2[[H, X1], Y1]|0>
        # temp = contract("jcka, ia -> ijkc", hbar.Hovov, X1_A)
        # temp = contract("ijkc, jb -> kibc", temp, X1_B)
        # LHX1Y1 -= contract("kibc, kibc -> ", temp, l2)
        Sijmn = self.Local.Sijmn
        for j in range(no):
            jj = j * no + j
            for k in range(no):
                kk = k*no + k
                for i in range(no):
                    ii = i * no + i
                    ki = k*no + i

                    jjki = jj * (no * no) + ki

                    # Hovov_mm = Hovov_mm + contract('be,bB,eE->BE', ERI[i,v,m,v], QL[mm], QL[ii])
                    Hovov_mm = contract('be, bB->Be', ERI[j, v, k, v], QL[ki])
                    Hovov_mm = contract('Be, eE->BE', Hovov_mm, QL[ii])


                    tmp = contract('bef, bB->Bef', ERI[v, j, v, v], QL[ki])
                    tmp = contract('Bef, eE -> BEf', tmp, QL[ii])
                    tmp = contract('BEf, fF -> BEF', tmp, QL[kk])
                    Hovov_mm = Hovov_mm + contract('f,bef->be', t1[k], tmp)
                    for n in range(no):
                        kn = k * no + n
                        nn = n * no + n
                        kinn = ki*(no*no) + nn
                        jn = j*no + n
                        kikn = ki*(no*no) + kn

                        tmp1 = Sijmn[kinn] @ t1[n]
                        tst = contract('e, eE->E', ERI[j, n, k, v], QL[ii])
                        Hovov_mm = Hovov_mm - contract('b,e->be', tmp1, tst)

                        tmp2 = t2[kn] @ Sijmn[kikn].T
                        tmp3 = contract('ef, eE->Ef', ERI[n, j, v, v], QL[ii])
                        tmp3 = contract('Ef, fF -> EF', tmp3, QL[kn])
                        Hovov_mm = Hovov_mm - contract('fb,ef->be', tmp2, tmp3)

                        tmp3 = contract('ef, eE->Ef', ERI[n, j, v, v], QL[ii])
                        tmp3 = contract('Ef, fF -> EF', tmp3, QL[kk])
                        tmp = contract('f, ef -> e', t1[k], tmp3)
                        Hovov_mm = Hovov_mm - contract('b,e->be', Sijmn[kinn] @ t1[n], tmp)

                    temp = contract("ca, a -> c", Hovov_mm, X1_A[i])
                    temp = contract("c, b -> bc", temp, X1_B[j])
                    LHX1Y1 -= contract("bc, bc -> ", temp, Sijmn[jjki] @ l2[ki])


        # temp = contract("jcak, ia -> ijkc", hbar.Hovvo, X1_A)
        # temp = contract("ijkc, jb -> kicb", temp, X1_B)
        # LHX1Y1 -= contract("kicb, kicb -> ", temp, l2)
        # Hi v_mm v_ii o - Eqn 102
        Sijmn = self.Local.Sijmn
        for j in range(no):
            jj = j * no + j
            for k in range(no):
                kk = k * no + k
                for i in range(no):
                    ii = i * no + i
                    ki = k * no + i
                    jjki = jj * (no * no) + ki

                    # Hovov_mm = Hovov_mm + contract('be,bB,eE->BE', ERI[i,v,m,v], QL[mm], QL[ii])
                    Hovov_mm = contract('be, bB->Be', ERI[j, v, v, k], QL[ki])
                    Hovov_mm = contract('Be, eE->BE', Hovov_mm, QL[ii])

                    tmp = contract('bef, bB->Bef', ERI[j, v, v, v], QL[ki])
                    tmp = contract('Bef, eE -> BEf', tmp, QL[ii])
                    tmp = contract('BEf, fF -> BEF', tmp, QL[kk])
                    Hovov_mm = Hovov_mm + contract('f,bef->be', t1[k], tmp)
                    for n in range(no):
                        nn = n * no + n
                        kn = k * no + n
                        nk = n * no + k
                        kinn = ki * (no * no) + nn
                        jn = j * no + n
                        kikn = ki * (no * no) + kn
                        kink = ki * (no * no) + nk
                        tmp1 = Sijmn[kinn] @ t1[n]
                        tst = contract('e, eE->E', ERI[j, n, v, k], QL[ii])
                        Hovov_mm = Hovov_mm - contract('b,e->be', tmp1, tst)

                        tmp2 = t2[kn] @ Sijmn[kikn].T
                        tmp3 = contract('ef, eE->Ef', ERI[j, n, v, v], QL[ii])
                        tmp3 = contract('Ef, fF -> EF', tmp3, QL[kn])
                        Hovov_mm = Hovov_mm - contract('fb,ef->be', tmp2, tmp3)

                        tmp3 = contract('ef, eE->Ef', ERI[j, n, v, v], QL[ii])
                        tmp3 = contract('Ef, fF -> EF', tmp3, QL[kk])
                        tmp = contract('f, ef -> e', t1[k], tmp3)
                        Hovov_mm = Hovov_mm - contract('b,e->be', Sijmn[kinn] @ t1[n], tmp)
                        temp = contract('ef, eE -> Ef', L[j, n, v, v], QL[ii])
                        temp = contract('Ef, fF -> EF', temp, QL[nk])
                        Hovov_mm = Hovov_mm + contract('fb, ef -> be', t2[nk] @ Sijmn[kink].T, temp)

                    temp = contract("ca, a -> c", Hovov_mm, X1_A[i])
                    temp = contract("c, b -> cb", temp, X1_B[j])
                    LHX1Y1 -= contract("cb, cb -> ", temp, l2[ki] @ Sijmn[jjki].T)
        # temp = contract('cdab, ia -> ibcd', hbar.Hvvvv, X1_B)
        # temp = contract('ibcd, jb -> ijcd', temp, X1_A)
        # LHX1Y1 += 0.5 * contract('ijcd, ijcd', temp, l2)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                ij = i * no + j

                temp = contract('cdab, a -> bcd', hbar.Hvvvv_ij[ij], X1_B[i])
                temp = contract('bcd, b -> cd', temp, X1_A[j])
                LHX1Y1 += 0.5 * contract('cd, cd', temp, l2[ij])


        # temp = contract('ijkl, ia -> klaj', hbar.Hoooo, X1_B)
        # temp = contract('klaj, jb -> klab', temp, X1_A)
        # LHX1Y1 += 0.5 * contract('klab, klab', temp, l2)
        for i in range(no):
            for j in range(no):
                    for k in range(no):
                        for l in range(no):
                            kl = k * no + l
                            ii = i * no + i
                            jj = j * no + j
                            kljj = kl*(no*no)+jj
                            klii = kl*(no*no)+ii
                            temp = contract(', a -> a', self.hbar.Hoooo[i, j, k, l], X1_B[i])
                            temp = contract('a, b -> ab', temp, X1_A[j])
                            LHX1Y1 += 0.5 * contract('ab, ab', temp, Sijmn[klii].T @ l2[kl] @ Sijmn[kljj])
        llhx1y1_end = process_time()
        self.lLHX1Y1_t += llhx1y1_end - llhx1y1_start
        return LHX1Y1

    def lLHX2Y2(self, X2_A, X2_B):
        llhx2y2_start = process_time()
        """
        LHX2Y2: Is a function for the second term (quadratic term) contribution to the linear response
        function, where both perturbed amplitudes are second order.
        # <0|(1 + L^(0)){ [[\bar{B}^(0), X^(1)(B)], X^(1)(B)] }|0>
        """

        contract = self.ccwfn.contract
        no = self.ccwfn.no
        v = self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        t2 = self.lccwfn.t2
        hbar = self.hbar
        L = self.ccwfn.H.L
        ERI = self.ccwfn.H.ERI
        Sijmn = self.Local.Sijmn
        QL = self.Local.QL

        LHX2Y2 = 0.0
        # temp = contract("ijac, ijab -> bc", L[o, o, v, v], X2_A)
        # temp = contract("bc, klcd -> klbd", temp, X2_B)
        # LHX2Y2 -= contract("klbd, klbd -> ", temp, l2)
        for i in range(no):
            for j in range(no):
                ij = i * no + j
                for k in range(no):
                    ik = i * no + k
                    jk = j * no + k
                    for l in range(no):
                        kl = k * no + l
                        jl = j * no + l
                        il = i * no + l
                        ijkl = ij * (no * no) + kl
                        jlkl = jl * (no * no) + kl
                        jkkl = jk * (no * no) + kl
                        klij = kl * (no * no) + ij
                        klil = kl * (no * no) + il
                        ikij = ik * (no * no) + ij
                        ijjl = ij * (no * no) + jl
                        jkij = jk * (no * no) + ij
                        ijil = ij * (no * no) + il

                        temp = contract('ac, ab -> bc', QL[ij].T @ L[i, j, v, v] @ QL[kl], X2_A[ij])
                        temp = contract('bc, cd -> bd', temp, X2_B[kl])
                        LHX2Y2 -= contract('bd, bd', temp, Sijmn[ijkl] @ l2[kl])

        # temp = contract("ijac, ikac -> jk", L[o, o, v, v], X2_A)
        # temp = contract("jk, jlbd -> klbd", temp, X2_B)
        # LHX2Y2 -= contract("klbd, klbd -> ", temp, l2)
                        temp = contract('ac, ac', QL[ik].T @ L[i, j, v, v] @ QL[ik], X2_A[ik])
                        temp = contract(', bd -> bd', temp, X2_B[jl])
                        LHX2Y2 -= contract('bd, bd', temp, Sijmn[jlkl] @ l2[kl] @ Sijmn[jlkl].T)

        # temp = contract("ijac, jkbc -> ikab", L[o, o, v, v], X2_A)
        # temp = contract("ikab, ilad -> klbd", temp, X2_B)
        # LHX2Y2 -= contract("klbd, klbd -> ", temp, l2)
                        temp = contract('ac, bc -> ab', QL[il].T @ L[i, j, v, v] @ QL[jk], X2_A[jk])
                        temp = contract('ab, ad -> bd', temp, X2_B[il])
                        LHX2Y2 -= contract('bd, bd', temp, Sijmn[jkkl] @ l2[kl] @ Sijmn[klil])
        # temp = contract("klab, ijab -> klij", ERI[o, o, v, v].copy(), X2_A)
        # temp = contract("klij, klcd -> ijcd", temp, X2_B)
        # LHX2Y2 += 0.5 * contract("ijcd, ijcd -> ", temp, l2)
                        temp = contract('ab, ab', QL[ij].T @ ERI[k, l, v, v] @ QL[ij], X2_A[ij])
                        temp = contract(', cd -> cd', temp, X2_B[kl])
                        LHX2Y2 += 0.5 * contract('cd, cd', temp, Sijmn[klij] @ l2[ij] @ Sijmn[klij].T)
        # temp = contract("klab, ikac -> ilbc", ERI[o, o, v, v].copy(), X2_A)
        # temp = contract("ilbc, jlbd -> ijcd", temp, X2_B)
        # LHX2Y2 += 0.5 * contract("ijcd, ijcd -> ", temp, l2)
                        temp = contract('ab, ac -> bc', QL[ik].T @ ERI[k, l, v, v] @ QL[jl], X2_A[ik])
                        temp = contract('bc, bd -> cd', temp, X2_B[jl])
                        LHX2Y2 += 0.5 * contract('cd, cd', temp, Sijmn[ikij] @ l2[ij] @ Sijmn[ijjl])
        # temp = contract("klab, ilad -> ikbd", ERI[o, o, v, v].copy(), X2_A)
        # temp = contract("ikbd, jkbc -> ijcd", temp, X2_B)
        # LHX2Y2 += 0.5 * contract("ijcd, ijcd -> ", temp, l2)
                        temp = contract('ab, ad -> bd', QL[il].T @ ERI[k, l, v, v] @ QL[jk], X2_A[il])
                        temp = contract('bd, bc -> cd', temp, X2_B[jk])
                        LHX2Y2 += 0.5 * contract('cd, cd', temp, Sijmn[jkij] @ l2[ij] @ Sijmn[ijil])
        llhx2y2_end = process_time()
        self.lLHX2Y2_t += llhx2y2_end - llhx2y2_start
        return LHX2Y2

    def lLHX1Y2(self, X1_B, X2_A):
        llhx1y2_start = process_time()
        """
        LHX1Y2: Is a function for the second term (quadratic term) contribution to the linear response
        function, where perturbed amplitudes is a first order and a second order.
        # <0|(1 + L^(0)){ [[\bar{B}^(0), X^(1)(B)], X^(1)(B)] }|0>
        """

        contract = self.ccwfn.contract
        no = self.ccwfn.no
        o = self.ccwfn.o
        v = self.ccwfn.v
        t1 = self.lccwfn.t1
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        hbar = self.hbar
        L = self.ccwfn.H.L
        ERI = self.ccwfn.H.ERI
        QL = self.Local.QL
        Sijmn = self.Local.Sijmn

        LHX1Y2 = 0.0
        # # <O|L1(0)[[Hbar(0),X1(B)],X2(A)]]|0>
        # temp = contract("ijac, ia -> jc", L[o, o, v, v], X1_B)
        # temp = contract("jc, jkbc -> kb", temp, X2_A)
        # LHX1Y2 -= contract("kb, kb", temp, l1)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                ij = i * no + j
                jj = j * no + j
                for k in range(no):
                    kk = k * no + k
                    ik = i * no + k
                    jk = j * no + k
                    jkkk = jk * (no * no) + kk
                    ijkk = ij * (no * no) + kk
                    jjkk = jj * (no * no) + kk
                    temp = contract('ac, a -> c', QL[ii].T @ L[i, j, v, v] @ QL[jk], X1_B[i])
                    temp = contract('c, bc -> b', temp, X2_A[jk])
                    LHX1Y2 -= contract('b, b', temp, Sijmn[jkkk] @ l1[k])
        # temp = contract("ijac, ia -> jc", L[o, o, v, v], X1_B)
        # temp = contract("jc, jkcb -> kb", temp, X2_A)
        # LHX1Y2 += 2.0 * contract("kb, kb", temp, l1)
                    temp = contract('ac, a -> c', QL[ii].T @ L[i, j, v, v] @ QL[jk], X1_B[i])
                    temp = contract('c, cb -> b', temp, X2_A[jk])
                    LHX1Y2 += 2.0 * contract('b, b', temp, Sijmn[jkkk] @ l1[k])
        # temp = contract("ijac, ikac -> jk", L[o, o, v, v], X2_A)
        # temp = contract("jk, jb -> kb", temp, X1_B)
        # LHX1Y2 -= contract("kb, kb -> ", temp, l1)
                    temp = contract('ac, ac -> ', QL[ik].T @ L[i, j, v, v] @ QL[ik], X2_A[ik])
                    temp = contract(', b -> b', temp, X1_B[j])
                    LHX1Y2 -= contract('b, b', temp, Sijmn[jjkk] @ l1[k])
        # temp = contract("ijac, ijab -> bc", L[o, o, v, v], X2_A)
        # temp = contract("bc, kc -> kb", temp, X1_B)
        # LHX1Y2 -= contract("kb, kb -> ", temp, l1)
                    temp = contract('ac, ab -> bc', QL[ij].T @ L[i, j, v, v] @ QL[kk], X2_A[ij])
                    temp = contract('bc, c -> b', temp, X1_B[k])
                    LHX1Y2 -= contract('b, b', temp, Sijmn[ijkk] @ l1[k])

        # # <O|L2(A)[[Hbar(0),X2(B)],X1(C)]]|0>
        # # temp = contract('ja, ia -> ij', hbar.Hov, X1_B)
        # # temp = contract('ij, jkbc -> ikbc', temp, X2_A)
        # # LHX1Y2 -= contract('ikbc, ikbc', temp, l2)
        for i in range(no):
            for j in range(no):
                for k in range(no):
                    ii = i * no + i
                    jj = j * no + j
                    jk = j * no + k
                    ik = i * no + k
                    jkik = jk * (no * no) + ik
                    jjik = jj * (no * no) + ik
                    temp = contract('a, a -> ', hbar.Hov[ii][j], X1_B[i])
                    temp = contract(', bc -> bc', temp, X2_A[jk])
                    LHX1Y2 -= contract('bc, bc -> ', temp, Sijmn[jkik] @ l2[ik] @ Sijmn[jkik].T)
        # temp = contract('ja, jb -> ab', hbar.Hov, X1_B)
        # temp = contract('ab, ikac -> ikbc', temp, X2_A)
        # LHX1Y2 -= contract('ikbc, ikbc', temp, l2)
                    temp = contract('a, b -> ab', hbar.Hov[ik][j], X1_B[j])
                    temp = contract('ab, ac -> bc', temp, X2_A[ik])
                    LHX1Y2 -= contract('bc, bc', temp, Sijmn[jjik] @ l2[ik])
        #
        #
        # # temp = contract('jima, ia -> jm', (2 * hbar.Hooov - hbar.Hooov.swapaxes(0, 1)), X1_B)
        # # temp = contract('jm, jkbc -> kmcb', temp, X2_A)
        # # LHX1Y2 -= contract('kmcb, kmcb -> ', temp, l2)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                jj = j * no + j
                for k in range(no):
                    ik = i * no + k
                    jk = j * no + k
                    for m in range(no):
                        km = k * no + m
                        jkkm = jk * (no * no) + km
                        ikkm = ik * (no * no) + km
                        kmjj = km * (no * no) + jj
                        temp = contract('a, a', (2.0 * hbar.Hooov[ii][j, i, m] - hbar.Hooov[ii][i, j, m]), X1_B[i])
                        temp = contract(', bc -> cb', temp, X2_A[jk])
                        LHX1Y2 -= contract('cb, cb', temp, Sijmn[jkkm] @ l2[km] @ Sijmn[jkkm].T)
        # # temp = contract('jima, jb -> imab', (2 * hbar.Hooov - hbar.Hooov.swapaxes(0, 1)), X1_B)
        # # temp = contract('imab, ikac -> kmcb', temp, X2_A)
        # # LHX1Y2 -= contract('kmcb, kmcb', temp, l2)
                        temp = contract('a, b -> ab', (2.0 * hbar.Hooov[ik][j, i, m] - hbar.Hooov[ik][i, j, m]), X1_B[j])
                        temp = contract('ab, ac -> cb', temp, X2_A[ik])
                        LHX1Y2 -= contract('cb, cb', temp, Sijmn[ikkm] @ l2[km] @ Sijmn[kmjj])
        #
        # # temp = contract('djab, ia -> djib', (2 * hbar.Hvovv - hbar.Hvovv.swapaxes(2, 3)), X1_B)
        # # temp = contract('djib, jkbc -> kicd', temp, X2_A)
        # # LHX1Y2 += contract('kicd, kicd', temp, l2)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                for k in range(no):
                    ki = k * no + i
                    jk = j * no + k
                    kk = k * no + k
                    kikk = ki * (no * no) + kk
                    jkki = jk * (no * no) + ki

                      #Hv_{ki}jv_{ii]v_{jk}
                    Hvovv = contract('dab, dD, aA, bB -> DAB', ERI[v, j, v, v], QL[ki], QL[ii], QL[jk])
                      #Hv_{ki}jv_{jk]v_{ii}
                    Hvovv_swap = contract('dba, dD, bB, aA -> DBA', ERI[v, j, v, v], QL[ki], QL[jk], QL[ii])
                    for n in range(no):
                        nn = n * no + n
                        nnki = nn*(no*no) + ki
                        Hvovv = Hvovv - contract('d, ab -> dab', t1[n] @ Sijmn[nnki], QL[ii].T @ ERI[n,j,v,v] @ QL[jk])
                        Hvovv_swap = Hvovv_swap - contract('d, ba -> dba', t1[n] @ Sijmn[nnki] , QL[jk].T @ ERI[n, j, v, v] @ QL[ii])
                    temp = contract('dab, a -> db', 2.0 * Hvovv - Hvovv_swap.swapaxes(1,2), X1_B[i])
                    temp = contract('db, bc -> cd', temp, X2_A[jk])
                    LHX1Y2 += contract('cd, cd', temp, Sijmn[jkki] @ l2[ki])

        # # temp = contract('djab, jb -> ad', (2 * hbar.Hvovv - hbar.Hvovv.swapaxes(2, 3)), X1_B)
        # # temp = contract('ad, ikac -> kicd', temp, X2_A)
        # # LHX1Y2 += contract('kicd, kicd', temp, l2)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                jj = j * no + j
                for k in range(no):
                    ki = k * no + i
                    ik = i * no + k
                    jk = j * no + k
                    kk = k * no + k
                    ikki = ik * (no * no) + ki
                    jkki = jk * (no * no) + ki

                      #Hv_{ki}jv_{ii]v_{jk}
                    Hvovv = contract('dab, dD, aA, bB -> DAB', ERI[v, j, v, v], QL[ki], QL[ik], QL[jj])
                      #Hv_{ki}jv_{jk]v_{ii}
                    Hvovv_swap = contract('dba, dD, bB, aA -> DBA', ERI[v, j, v, v], QL[ki], QL[jj], QL[ik])
                    for n in range(no):
                        nn = n * no + n
                        nnki = nn*(no*no) + ki
                        Hvovv = Hvovv - contract('d, ab -> dab', t1[n] @ Sijmn[nnki], QL[ik].T @ ERI[n,j,v,v] @ QL[jj])
                        Hvovv_swap = Hvovv_swap - contract('d, ba -> dba', t1[n] @ Sijmn[nnki] , QL[jj].T @ ERI[n, j, v, v] @ QL[ik])
                    temp = contract('dab, b -> da', 2.0 * Hvovv - Hvovv_swap.swapaxes(1,2), X1_B[j])
                    temp = contract('da, ac -> cd', temp, X2_A[ik])
                    LHX1Y2 += contract('cd, cd', temp, Sijmn[ikki] @ l2[ki])
        # #
        # # temp = contract('dkab, ia -> ikdb', hbar.Hvovv, X1_B)
        # # temp = contract('ikdb, jkbc -> ijdc', temp, X2_A)
        # # LHX1Y2 -= contract('ijdc, ijdc', temp, l2)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                ij = i * no + j
                jj = j * no + j
                for k in range(no):
                    ki = k * no + i
                    ik = i * no + k
                    jk = j * no + k
                    kk = k * no + k
                    ikki = ik * (no * no) + ki
                    jkki = jk * (no * no) + ki
                    ijjk = ij * (no * no) + jk

                      #Hv_{ki}jv_{ii]v_{jk}
                    Hvovv = contract('dab, dD, aA, bB -> DAB', ERI[v, k, v, v], QL[ij], QL[ii], QL[jk])
                      #Hv_{ki}jv_{jk]v_{ii}
                    # Hvovv_swap = contract('dba, dD, bB, aA -> DBA', ERI[v, j, v, v], QL[ki], QL[jj], QL[ik])
                    for n in range(no):
                        nn = n * no + n
                        nnki = nn*(no*no) + ki
                        nnij = nn * (no*no) + ij
                        Hvovv = Hvovv - contract('d, ab -> dab', t1[n] @ Sijmn[nnij], QL[ii].T @ ERI[n,k,v,v] @ QL[jk])
                        # Hvovv_swap = Hvovv_swap - contract('d, ba -> dba', t1[n] @ Sijmn[nnki] , QL[jj].T @ ERI[n, j, v, v] @ QL[ik])
                    temp = contract('dab, a -> db', Hvovv, X1_B[i])
                    temp = contract('db, bc -> dc', temp, X2_A[jk])
                    LHX1Y2 -= contract('dc, dc', temp, l2[ij] @ Sijmn[ijjk])
        #
        # # temp = contract('dkab, jb -> kjad', hbar.Hvovv, X1_B)
        # # temp = contract('kjad, ikac -> ijdc', temp, X2_A)
        # # LHX1Y2 -= contract('ijdc, ijdc', temp, l2)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                ij = i * no + j
                jj = j * no + j
                for k in range(no):
                    ki = k * no + i
                    ik = i * no + k
                    jk = j * no + k
                    kk = k * no + k
                    ikki = ik * (no * no) + ki
                    jkki = jk * (no * no) + ki
                    ijik = ij * (no * no) + ik

                    # Hv_{ki}jv_{ii]v_{jk}
                    Hvovv = contract('dab, dD, aA, bB -> DAB', ERI[v, k, v, v], QL[ij], QL[ik], QL[jj])
                    # Hv_{ki}jv_{jk]v_{ii}
                    # Hvovv_swap = contract('dba, dD, bB, aA -> DBA', ERI[v, j, v, v], QL[ki], QL[jj], QL[ik])
                    for n in range(no):
                        nn = n * no + n
                        nnki = nn * (no * no) + ki
                        nnij = nn * (no * no) + ij
                        Hvovv = Hvovv - contract('d, ab -> dab', t1[n] @ Sijmn[nnij],QL[ik].T @ ERI[n, k, v, v] @ QL[jj])
                        # Hvovv_swap = Hvovv_swap - contract('d, ba -> dba', t1[n] @ Sijmn[nnki] , QL[jj].T @ ERI[n, j, v, v] @ QL[ik])
                    temp = contract('dab, b -> da', Hvovv, X1_B[j])
                    temp = contract('da, ac -> dc', temp, X2_A[ik])
                    LHX1Y2 -= contract('dc, dc', temp, l2[ij] @ Sijmn[ijik])
        #
        #
        # # temp = contract('dkab, kc -> abcd', hbar.Hvovv, X1_B)
        # # temp = contract('abcd, ijab -> ijdc', temp, X2_A)
        # # LHX1Y2 -= contract('ijdc, ijdc', temp, l2)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                ij = i * no + j
                jj = j * no + j
                for k in range(no):
                    ki = k * no + i
                    ik = i * no + k
                    jk = j * no + k
                    kk = k * no + k
                    ikki = ik * (no * no) + ki
                    jkki = jk * (no * no) + ki
                    ijik = ij * (no * no) + ik
                    ijkk = ij * (no * no) + kk

                    # Hv_{ki}jv_{ii]v_{jk}
                    Hvovv = contract('dab, dD, aA, bB -> DAB', ERI[v, k, v, v], QL[ij], QL[ij], QL[ij])
                    # Hv_{ki}jv_{jk]v_{ii}
                    # Hvovv_swap = contract('dba, dD, bB, aA -> DBA', ERI[v, j, v, v], QL[ki], QL[jj], QL[ik])
                    for n in range(no):
                        nn = n * no + n
                        nnki = nn * (no * no) + ki
                        nnij = nn * (no * no) + ij
                        Hvovv = Hvovv - contract('d, ab -> dab', t1[n] @ Sijmn[nnij],QL[ij].T @ ERI[n, k, v, v] @ QL[ij])
                        # Hvovv_swap = Hvovv_swap - contract('d, ba -> dba', t1[n] @ Sijmn[nnki] , QL[jj].T @ ERI[n, j, v, v] @ QL[ik])
                    temp = contract('dab, c -> abcd', Hvovv, X1_B[k])
                    temp = contract('abcd, ab -> dc', temp, X2_A[ij])
                    LHX1Y2 -= contract('dc, dc', temp, l2[ij] @ Sijmn[ijkk])
        # #
        # # temp = contract('jkla, ia -> ijkl', hbar.Hooov, X1_B)
        # # temp = contract('ijkl, jkbc -> libc', temp, X2_A)
        # # LHX1Y2 += contract('libc, libc', temp, l2)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                ij = i * no + j
                jj = j * no + j
                for k in range(no):
                    ik = i * no + k
                    jk = j * no + k
                    kk = k * no + k
                    for l in range(no):
                        li = l * no + i
                        jkli = jk * (no * no) + li
                        jjli = jj * (no * no) + li
                        ikli = ik * (no * no) + li
                        ijli = ij * (no * no) + li
                        likk = li * (no * no) + kk
                        temp = contract('a, a', hbar.Hooov[ii][j, k, l], X1_B[i])
                        temp = contract(', bc -> bc', temp, X2_A[jk])
                        LHX1Y2 += contract('bc, bc', temp, Sijmn[jkli] @ l2[li] @ Sijmn[jkli].T)
        # temp = contract('jkla, jb -> klab', hbar.Hooov, X1_B)
        # temp = contract('klab, ikac -> libc', temp, X2_A)
        # LHX1Y2 += contract('libc, libc -> ', temp, l2)
                        temp = contract('a, b -> ab', hbar.Hooov[ik][j, k, l], X1_B[j])
                        temp = contract('ab, ac -> bc', temp, X2_A[ik])
                        LHX1Y2 += contract('bc, bc', temp, Sijmn[jjli] @ l2[li] @ Sijmn[ikli].T)
        # temp = contract('jkla, kc -> jlac', hbar.Hooov, X1_B)
        # temp = contract('jlac, ijab -> libc', temp, X2_A)
        # LHX1Y2 += contract('libc, libc -> ', temp, l2)
                        temp = contract('a, c -> ac', hbar.Hooov[ij][j, k, l], X1_B[k])
                        temp = contract('ac, ab -> bc', temp, X2_A[ij])
                        LHX1Y2 += contract('bc, bc', temp, Sijmn[ijli] @ l2[li] @ Sijmn[likk])
        llhx1y2_end = process_time()
        self.lLHX1Y2_t += llhx1y2_end - llhx1y2_start
        return LHX1Y2

    def linresp_asym(self, pertkey_a, X1_B, X2_B, Y1_B, Y2_B):
        start_lr_asym = process_time()
        """
    Calculate the CC linear response function for polarizability at field-frequency omega(w1).

    The linear response function, <<A;B(w1)>> generally reuires the following perturbed wave functions and frequencies:

    Parameters
    ----------
    pertkey_a: string
        String identifying the one-electron perturbation, A along a cartesian axis


    Return
    ------
    polar: float
         A value of the chosen linear response function corresponding to compute polariazabiltity in a specified cartesian diresction.
        """

        contract = self.ccwfn.contract

        # Defining the l1 and l2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2

        # Please refer to eqn 78 of [Crawford: https://crawford.chem.vt.edu/wp-content/uploads/2022/06/cc_response.pdf].
        # Writing H(1)(omega) = B, T(1)(omega) = X, L(1)(omega) = y
        # <<A;B>> = <0|Y(B) * A_bar|0> + <0| (1 + L(0))[A_bar, X(B)}|0>
        #                 polar1                polar2
        polar1 = 0
        polar2 = 0
        pertbar_A = self.pertbar[pertkey_a]
        Avvoo = pertbar_A.Avvoo.swapaxes(0, 2).swapaxes(1, 3)

        # # <0|Y1(B) * A_bar|0>
        polar1 += contract("ai, ia -> ", pertbar_A.Avo, Y1_B)
        # # <0|Y2(B) * A_bar|0>
        polar1 += 0.5 * contract("abij, ijab -> ", Avvoo, Y2_B)
        polar1 += 0.5 * contract("baji, ijab -> ", Avvoo, Y2_B)
        # <0|[A_bar, X(B)]|0>
        polar2 += 2.0 * contract("ia, ia -> ", pertbar_A.Aov, X1_B)
        # <0|L1(0) [A_bar, X1(B)]|0>
        tmp = contract("ia, ic -> ac", l1, X1_B)
        polar2 += contract("ac, ac -> ", tmp, pertbar_A.Avv)
        tmp = contract("ia, ka -> ik", l1, X1_B)
        polar2 -= contract("ik, ki -> ", tmp, pertbar_A.Aoo)
        # # <0|L1(0)[a_bar, X2(B)]|0>
        tmp = contract("ia, jb -> ijab", l1, pertbar_A.Aov)
        polar2 += 2.0 * contract("ijab, ijab -> ", tmp, X2_B)
        polar2 += -1.0 * contract("ijab, ijba -> ", tmp, X2_B)
        # # <0|L2(0)[A_bar, X1(B)]|0>
        tmp = contract("ijbc, bcaj -> ia", l2, pertbar_A.Avvvo)
        polar2 += contract("ia, ia -> ", tmp, X1_B)
        tmp = contract("ijab, kbij -> ak", l2, pertbar_A.Aovoo)
        polar2 -= 0.5 * contract("ak, ka -> ", tmp, X1_B)
        tmp = contract("ijab, kaji -> bk", l2, pertbar_A.Aovoo)
        polar2 -= 0.5 * contract("bk, kb -> ", tmp, X1_B)
        # # <0|L2(0)[A_bar, X2(B)]|0>
        tmp = contract("ijab, kjab -> ik", l2, X2_B)
        polar2 -= 0.5 * contract("ik, ki -> ", tmp, pertbar_A.Aoo)
        tmp = contract("ijab, kiba-> jk", l2, X2_B)
        polar2 -= 0.5 * contract("jk, kj -> ", tmp, pertbar_A.Aoo)
        tmp = contract("ijab, ijac -> bc", l2, X2_B)
        polar2 += 0.5 * contract("bc, bc -> ", tmp, pertbar_A.Avv)
        tmp = contract("ijab, ijcb -> ac", l2, X2_B)
        polar2 += 0.5 * contract("ac, ac -> ", tmp, pertbar_A.Avv)

        end_lr_asym = process_time()
        self.linresp_asym_t = end_lr_asym - start_lr_asym
        return -1.0 * (polar1 + polar2)

    def local_linresp(self, axis, pertkey_a, X1_B, Y1_B, X2_B, Y2_B):

        """
         Calculate the CC linear response function for polarizability at field-frequency omega(w1).

         The linear response function, <<A;B(w1)>> generally reuires the following perturbed wave functions and frequencies:

         Parameters
         ----------
         pertkey_a: string
             String identifying the one-electron perturbation, A along a cartesian axis


         Return
         ------
         polar: float
              A value of the chosen linear response function corresponding to compute polariazabiltity in a specified cartesian diresction.
             """

        contract = self.ccwfn.contract
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2
        no = self.ccwfn.no
        v = self.ccwfn.v
        Sijmn = self.Local.Sijmn
        pertbar_A = self.lpertbar[pertkey_a]
        QL = self.Local.QL
        pert = self.H.mu[axis]
        Sijmi = self.ccwfn.Local.Sijmi
        Sijmm = self.ccwfn.Local.Sijmm

        # Please refer to eqn 78 of [Crawford: https://crawford.chem.vt.edu/wp-content/uploads/2022/06/cc_response.pdf].
        # Writing H(1)(omega) = B, T(1)(omega) = X, L(1)(omega) = y
        # <<A;B>> = <0|Y(B) * A_bar|0> + <0| (1 + L(0))[A_bar, X(B)}|0>
        #                 polar1                polar2
        polar1 = 0
        polar2 = 0
        quad_start = process_time()
        # Avvoo = pertbar_A.Avvoo.swapaxes(0, 2).swapaxes(1, 3)
        Avvoo = pertbar_A.Avvoo.copy()
        # <0|Y1(B) * A_bar|0>
        # polar1 += contract("ai, ia -> ", pertbar_A.Avo, Y1_B)
        for i in range(no):
            ii = i * no + i

            tmp = QL[ii].T @ pert[v, i].copy()
            tmp += t1[i] @ (QL[ii].T @ pert[v, v].copy() @ QL[ii]).T
          
             # Sijmi = self.ccwfn.Local.Sijmi
            for m in range(no):
                mi = m * no + i
                iim = ii * no + m
                tmp -= (t1[m] @ Sijmm[iim].T) * pert[m, i].copy()
                tmp1 = (2.0 * t2[mi] - t2[mi].swapaxes(0, 1)) @ Sijmi[iim].T
                tmp += contract('ea,e->a', tmp1, pert[m, v].copy() @ QL[mi])
                tmp -= contract('e,a,e->a', t1[i], t1[m] @ Sijmm[iim].T, pert[m, v].copy() @ QL[ii])
            polar1 += contract('a, a', tmp, Y1_B[i].copy())
        # <0|Y2(B) * A_bar|0>
        # polar1 += 0.5 * contract("abij, ijab -> ", Avvoo, Y2_B)
        # polar1 += 0.5 * contract("baji, ijab -> ", Avvoo, Y2_B)
            for j in range(no):
                ij = i * no + j
                ji = j * no + i
                polar1 += 0.5 * contract('ab, ab', Avvoo[ij].copy(), Y2_B[ij].copy())
                polar1 += 0.5 * contract('ba, ab', Avvoo[ji].copy(), Y2_B[ij].copy())
        quad_end = process_time()
        self.quad_terms = quad_end - quad_start
        # <0|[A_bar, X(B)]|0>
        # polar2 += 2.0 * contract("ia, ia -> ", pertbar_A.Aov, X1_B)
        llcx_start = process_time()
        for i in range(no):
            ii = i * no + i
            polar2 += 2.0 * contract('a, a', pertbar_A.Aov[ii][i], X1_B[i])
        # <0|L1(0) [A_bar, X1(B)]|0>
        # tmp = contract("ia, ic -> ac", l1, X1_B)
        # polar2 += contract("ac, ac -> ", tmp, pertbar_A.Avv)
        for i in range(no):
            ii = i * no + i
            temp = contract('ac, c -> a', pertbar_A.Avv[ii].copy(), X1_B[i])
            polar2 += contract('a, a', temp, l1[i])
        # tmp = contract("ia, ka -> ik", l1, X1_B)
        # polar2 -= contract("ik, ki -> ", tmp, pertbar_A.Aoo)
        for i in range(no):
            for k in range(no):
                ii = i * no + i
                kk = k * no + k
                kkii = kk * (no * no) + ii
                temp = contract('a, a', Sijmn[kkii] @ l1[i], X1_B[k])
                polar2 -= temp * pertbar_A.Aoo[k, i]
        # <0|L1(0)[a_bar, X2(B)]|0>
        # tmp = contract("ia, jb -> ijab", l1, pertbar_A.Aov)
        # polar2 += 2.0 * contract("ijab, ijab -> ", tmp, X2_B)
        # polar2 += -1.0 * contract("ijab, ijba -> ", tmp, X2_B)
        for i in range(no):
            ii = i * no + i
            for j in range(no):
                ij = i * no + j
                iiij = ii * (no * no) + ij
                temp = contract('a, b -> ab', l1[i], pertbar_A.Aov[ij][j])
                polar2 += 2.0 * contract('ab, ab', temp, Sijmn[iiij] @ X2_B[ij])
                polar2 += -1.0 * contract('ab, ba', temp, X2_B[ij] @ Sijmn[iiij].T)

        # <0|L2(0)[A_bar, X1(B)]|0>
        # tmp = contract("ijbc, bcaj -> ia", l2, pertbar_A.Avvvo)
        # polar2 += contract("ia, ia -> ", tmp, X1_B)
        for i in range(no):
            for j in range(no):
                ij = i * no + j
                temp = contract('bc, bca -> a', l2[ij], pertbar_A.Avvvj_ii[ij])
                polar2 += contract('a, a', temp, X1_B[i])
        # tmp = contract("ijab, kbij -> ak", l2, pertbar_A.Aovoo)
        # polar2 -= 0.5 * contract("ak, ka -> ", tmp, X1_B)
                for k in range(no):
                    kk = k * no + k
                    ijkk = ij * (no * no) + kk
                    temp = contract('ab, b -> a', l2[ij], pertbar_A.Aovoo[ij][k])
                    polar2 -= 0.5 * contract('a, a', temp, X1_B[k] @ Sijmn[ijkk].T)
        # tmp = contract("ijab, kaji -> bk", l2, pertbar_A.Aovoo)
        # polar2 -= 0.5 * contract("bk, kb -> ", tmp, X1_B)
                    temp = contract('ab, a -> b', l2[ij], pertbar_A.Aovoo_ji[ij][k])
                    polar2 -= 0.5 * contract('b, b', temp, X1_B[k] @ Sijmn[ijkk].T)
        # <0|L2(0)[A_bar, X2(B)]|0>
        # tmp = contract("ijab, kjab -> ik", l2, X2_B)
        # polar2 -= 0.5 * contract("ik, ki -> ", tmp, pertbar_A.Aoo)
        for i in range(no):
            for j in range(no):
                ij = i * no + j
                for k in range(no):
                    ki = k * no + i
                    kj = k * no + j
                    ijkj = ij * (no * no) + kj
                    ijki = ij * (no * no) + ki
                    temp = contract('ab, ab', l2[ij], Sijmn[ijkj] @ X2_B[kj] @ Sijmn[ijkj].T)
                    polar2 -= 0.5 * temp * pertbar_A.Aoo[k, i]
        # tmp = contract("ijab, kiba-> jk", l2, X2_B)
        # polar2 -= 0.5 * contract("jk, kj -> ", tmp, pertbar_A.Aoo)
                    temp = contract('ab, ba', l2[ij], Sijmn[ijki] @ X2_B[ki] @ Sijmn[ijki].T)
                    polar2 -= 0.5 * temp * pertbar_A.Aoo[k, j]
        # tmp = contract("ijab, ijac -> bc", l2, X2_B)
        # polar2 += 0.5 * contract("bc, bc -> ", tmp, pertbar_A.Avv)
                temp = contract('ab, ac -> bc', l2[ij], X2_B[ij])
                polar2 += 0.5 * contract('bc, bc', temp, pertbar_A.Avv[ij].copy())
        # tmp = contract("ijab, ijcb -> ac", l2, X2_B)
        # polar2 += 0.5 * contract("ac, ac -> ", tmp, pertbar_A.Avv)
                temp = contract('ab, cb -> ac', l2[ij], X2_B[ij])
                polar2 += 0.5 * contract('ac, ac', temp, pertbar_A.Avv[ij].copy())
        llcx_end = process_time()
        self.llcx_t = llcx_end - llcx_start
        return -1.0 * (polar1 + polar2)


    def solve_right(self, pertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1, pert_filter=False):
        solver_start = time.time()

        Dia = self.Dia
        Dijab = self.Dijab

        #need to update local.py for add in parameter omega
        if self.ccwfn.filter is True:
            w = omega
            X1, X2 = self.ccwfn.Local.filter_amps(pertbar.Avo.T,pertbar.Avvoo, omega = w)
        # initial guess, comment out omega
        else:
            X1 = pertbar.Avo.T /(Dia + omega)
            X2 = pertbar.Avvoo /(Dijab + omega)

        pseudo = self.pseudoresponse(pertbar, X1, X2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}") 

        #commenting this out for now
        #if self.ccwfn.local is not None and self.ccwfn.filter is True:
        #    X1, X2 = self.ccwfn.Local.filter_res(X1, X2)
        #pseudo = self.pseudoresponse(pertbar, X1, X2)
        #print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        #diis = helper_diis(X1, X2, max_diis)
        contract = self.ccwfn.contract

        self.X1 = X1
        self.X2 = X2

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            r1 = self.r_X1(pertbar, omega)
            r2 = self.r_X2(pertbar, omega)

            #comment out omega and not use eps_vir
            if self.ccwfn.local is not None:
                inc1, inc2 = self.ccwfn.Local.filter_amps(r1, r2) #, self.eps_occ, self.eps_vir, omega)
                self.X1 += inc1
                self.X2 += inc2
            
                rms = contract('ia,ia->', np.conj(inc1/(Dia)), inc1/(Dia))
                rms += contract('ijab,ijab->', np.conj(inc2/(Dijab)), inc2/(Dijab))
                rms = np.sqrt(rms)
            #if self.ccwfn.local is not None and pert_filter:
            #    inc1, inc2 = self.ccwfn.Local.filter_pertamps(r1, r2, self.eps_occ, self.eps_vir, omega)  
            #    self.X1 += inc1
            #    self.X2 += inc2 

            #    rms = contract('ia,ia->', np.conj(inc1/(Dia+omega)), inc1/(Dia+omega))
            #    rms += contract('ijab,ijab->', np.conj(inc2/(Dijab+omega)), inc2/(Dijab+omega))
            #    rms = np.sqrt(rms)        
            #elif self.ccwfn.local is not None:
            #    inc1, inc2 = self.ccwfn.Local.filter_amps(r1, r2)
            #    self.X1 += inc1
            #    self.X2 += inc2

            #    rms = contract('ia,ia->', np.conj(inc1/(Dia+omega)), inc1/(Dia+omega))
            #    rms += contract('ijab,ijab->', np.conj(inc2/(Dijab+omega)), inc2/(Dijab+omega))
            #    rms = np.sqrt(rms)
            else:
                self.X1 += r1/(Dia + omega)
                self.X2 += r2/(Dijab + omega)

                rms = contract('ia,ia->', np.conj(r1/(Dia+omega)), r1/(Dia+omega))
                rms += contract('ijab,ijab->', np.conj(r2/(Dijab+omega)), r2/(Dijab+omega))
                rms = np.sqrt(rms)

            rms = np.sqrt(rms)
            #end loop

            pseudo = self.pseudoresponse(pertbar, self.X1, self.X2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")

            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv) or maxiter == niter :
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.X1, self.X2, pseudo

            #diis.add_error_vector(self.X1, self.X2)
            #if niter >= start_diis:
            #    self.X1, self.X2 = diis.extrapolate(self.X1, self.X2)


    def local_solve_right(self, lpertbar, omega, conv_hbar, e_conv=1e-12, r_conv=1e-12, maxiter=200):#max_diis=7, start_diis=1):
        """
        For X1, only contains the first term -> requires implementation to the local basis
        """
        solver_start = time.time()

        no = self.no

        contract = self.contract

        Avo = lpertbar.Avo.copy()
        Avvoo = lpertbar.Avvoo.copy()
 
        print("only keeping the numerator terms")
        self.X1 = []
        self.X2 = []
        for i in range(no):
            ii = i * no + i

            #Xv{ii}
            lX1 = Avo[ii].copy() 
            lX1 = lX1/(self.eps_occ[i] - self.eps_lvir[ii].reshape(-1,) + omega)
            self.X1.append(2.0 * lX1)
            for j in range(no):
                ij = i * no + j
                lX2 = Avvoo[ij].copy()
                lX2 = lX2/(self.eps_occ[i] + self.eps_occ[j] - self.eps_lvir[ij].reshape(1,-1) - self.eps_lvir[ij].reshape(-1,1) + omega) #- eps_lvir[ij][a,a] - eps_lvir[ij][b,b] + omega)
                self.X2.append(2.0 * lX2)

        pseudo = self.local_pseudoresponse(lpertbar, self.X1, self.X2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        #diis = helper_diis(X1, X2, max_diis)
        contract = self.ccwfn.contract

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            r1 = self.lr_X1(lpertbar, omega)
            r2 = self.lr_X2(lpertbar, conv_hbar, omega)

            #start loop
            rms = 0
            for i in range(no):
                ii = i * no + i
                
                #swap the sign
                #for a in range(self.Local.dim[ii]):
                self.X1[i] -= r1[i] / (self.Local.eps[ii].reshape(-1,) - self.H.F[i,i])

                #(self.eps_occ[i] - self.eps_lvir[ii].reshape(-1,) + omega)#- eps_lvir[ii][a,a] + omega)#(eps_occ[i] - eps_lvir[ii].reshape(-1,) + omega)
                rms += contract('a,a->', np.conj(r1[i] / (self.eps_occ[i])), (r1[i] / (self.eps_occ[i])))

                for j in range(no):
                    ij = i*no + j

                    self.X2[ij] -= r2[ij] / (self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])


#(self.eps_occ[i] + self.eps_occ[j] - self.eps_lvir[ij].reshape(1,-1) - self.eps_lvir[ij].reshape(-1,1) + omega)# - eps_lvir[ij][a,a] - eps_lvir[ij][b,b] + omega)#(eps_occ[i] + eps_occ[j] - eps_lvir[ij].reshape(1,-1) - eps_lvir[ij].reshape(-1,1) + omega)
                    rms += contract('ab,ab->', np.conj(r2[ij]/(self.eps_occ[i] + self.eps_occ[j])), r2[ij]/(self.eps_occ[i] + self.eps_occ[j]))

            rms = np.sqrt(rms)
            #end loop

            pseudo = self.local_pseudoresponse(lpertbar, self.X1, self.X2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")

            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.X1, self.X2, pseudo

            if niter == maxiter:
                print("\nPerturbed wave function not fully converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.X1, self.X2, pseudo

            #diis.add_error_vector(self.X1, self.X2)
            #if niter >= start_diis:
            #    self.X1, self.X2 = diis.extrapolate(self.X1, self.X2)
    
    def solve_left(self, pertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
        '''
        Notes
        -----
        The first-order lambda equations are partition into two expressions: inhomogeneous (in_Y1 and in_Y2) and homogeneous terms (r_Y1 and r_Y2), 
        the inhomogenous terms contains only terms that are not changing over the iterative process of obtaining the solutions for these equations. Therefore, it is 
        computed only once and is called when solving for the homogenous terms.         
        '''
        solver_start = time.time()

        Dia = self.Dia
        Dijab = self.Dijab

        if self.ccwfn.filter is True:
            w = omega
            X1_guess, X2_guess = self.ccwfn.Local.filter_amps(pertbar.Avo.T,pertbar.Avvoo, omega = w)
        # initial guess, comment out omega for local 
        else:
            X1_guess = pertbar.Avo.T /(Dia + omega)
            X2_guess = pertbar.Avvoo /(Dijab + omega)

        #if self.ccwfn.local is not None and self.ccwfn.filter is True:
        #    X1_guess, X2_guess = self.ccwfn.Local.filter_res(X1_guess, X2_guess)

        # initial guess
        Y1 = 2.0 * X1_guess.copy()
        Y2 = 4.0 * X2_guess.copy()
        Y2 -= 2.0 * X2_guess.copy().swapaxes(2,3)              

        pseudo = self.pseudoresponse(pertbar, Y1, Y2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")
        
        #diis = helper_diis(Y1, Y2, max_diis)

        self.Y1 = Y1
        self.Y2 = Y2 
        
        ## uses updated X1 and X2
        self.im_Y1 = self.in_Y1(pertbar, self.X1, self.X2)
        self.im_Y2 = self.in_Y2(pertbar, self.X1, self.X2)

        ##adding filter here
        #if self.ccwfn.local is not None and self.ccwfn.filter is True:
        #    self.im_Y1, self.im_Y2 = self.ccwfn.Local.filter_res(self.im_Y1, self.im_Y2)

        #adding to validate imhomogenous terms
        pseudo = self.pseudoresponse(pertbar, self.im_Y1, self.im_Y2)
        print(f"Iter {0:3d}: CC Psuedoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo
            
            Y1 = self.Y1
            Y2 = self.Y2
            
            r1 = self.r_Y1(pertbar, omega)
            r2 = self.r_Y2(pertbar, omega)
           
            #comment out omega and eps_vir for local
            if self.ccwfn.local is not None:
                inc1, inc2 = self.ccwfn.Local.filter_amps(r1, r2) #, self.eps_occ, self.eps_vir, omega)
                self.Y1 += inc1
                self.Y2 += inc2
            
                rms = contract('ia,ia->', np.conj(inc1/(Dia)), inc1/(Dia))
                rms += contract('ijab,ijab->', np.conj(inc2/(Dijab)), inc2/(Dijab))
                rms = np.sqrt(rms)
            #if self.ccwfn.local is not None and pert_filter:
            #    inc1, inc2 = self.ccwfn.Local.filter_pertamps(r1, r2, self.eps_occ, self.eps_vir, omega)
            #    self.Y1 += inc1
            #    self.Y2 += inc2 

            #    rms = contract('ia,ia->', np.conj(inc1/(Dia+omega)), inc1/(Dia+omega))
            #    rms += contract('ijab,ijab->', np.conj(inc2/(Dijab+omega)), inc2/(Dijab+omega))
            #    rms = np.sqrt(rms)

            #elif self.ccwfn.local is not None:
            #    inc1, inc2 = self.ccwfn.Local.filter_amps(r1, r2)
            #    self.Y1 += inc1
            #    self.Y2 += inc2

            #    rms = contract('ia,ia->', np.conj(inc1/(Dia+omega)), inc1/(Dia+omega))
            #    rms += contract('ijab,ijab->', np.conj(inc2/(Dijab+omega)), inc2/(Dijab+omega))
            #    rms = np.sqrt(rms)

            else:
                self.Y1 += r1/(Dia + omega)
                self.Y2 += r2/(Dijab + omega)
            
                rms = contract('ia,ia->', np.conj(r1/(Dia+omega)), r1/(Dia+omega))
                rms += contract('ijab,ijab->', np.conj(r2/(Dijab+omega)), r2/(Dijab+omega))
                rms = np.sqrt(rms)
            
            pseudo = self.pseudoresponse(pertbar, self.Y1, self.Y2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")
                
            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.Y1, self.Y2 , pseudo

            if niter == maxiter:
                print("\nPerturbed wave function not fully converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.Y1, self.Y2, pseudo

            #diis.add_error_vector(self.Y1, self.Y2)
            #if niter >= start_diis:
            #    self.Y1, self.Y2 = diis.extrapolate(self.Y1, self.Y2)

    def local_solve_left(self, lpertbar, omega, e_conv=1e-12, r_conv=1e-12, maxiter=200): #, max_diis=7, start_diis=1):
        """
        For Y1, only evaluates the first term of inhomogenous terms as well as the first term of homogenous terms
        """
        solver_start = time.time()
        no = self.no
        eps_occ = np.diag(self.cchbar.Hoo)
        eps_lvir = []
        for i in range(no):
            #ii = i *no + i
           for j in range(no):
                ij = i*no + j
                eps_lvir.append(np.diag(self.cchbar.Hvv[ij]))
                #print("eps_lvir_ij", ij, self.cchbar.Hvv[ij])
        contract =self.contract

        Q = self.Local.Q
        L = self.Local.L

        QL = self.Local.QL
        Avo = lpertbar.Avo.copy()
        Avvoo = lpertbar.Avvoo.copy()

        #initial guess for Y 
        self.Y1 = []
        self.Y2 = []

        for i in range(no):
            ii = i * no + i
            QL_ii = Q[ii] @ L[ii]

            #Xv{ii}
            lX1 = Avo[ii].copy()
            lX1 /= (self.Local.eps[ii].reshape(-1,) - self.H.F[i,i] + omega)
            self.Y1.append(2.0 * lX1.copy())

            for j in range(no):
                ij = i * no + j

                #temporary removing the virtual orbital energies
                lX2 = Avvoo[ij].copy()/(self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j] + omega)
                self.Y2.append((4.0 * lX2.copy()) - (2.0 * lX2.copy().swapaxes(0,1)))

        pseudo = self.local_pseudoresponse(lpertbar, self.Y1, self.Y2)
        print(f"Iter {0:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        ## uses updated X1 and X2
        self.im_Y1 = self.in_lY1(lpertbar, self.X1, self.X2)
        self.im_Y2 = self.in_lY2(lpertbar, self.X1, self.X2)

        #adding to validate imhomogenous terms
        pseudo = self.local_pseudoresponse(lpertbar, self.im_Y1, self.im_Y2)
        print(f"Iter {0:3d}: CC Psuedoresponse = {pseudo.real:.15f} dP = {pseudo.real:.5E}")

        #diis = helper_diis(X1, X2, max_diis)
        contract = self.ccwfn.contract

        for niter in range(1, maxiter+1):
            pseudo_last = pseudo

            r1 = self.lr_Y1(lpertbar, omega)
            r2 = self.lr_Y2(lpertbar, omega)

            #start loop
            rms = 0
            for i in range(no):
                ii = i * no + i

                #commented out error prone component
                self.Y1[i] -= r1[i] / (self.Local.eps[ii].reshape(-1,) - self.H.F[i,i])#(eps_occ[i] - eps_lvir[ii].reshape(-1,) + omega)
                rms += contract('a,a->', np.conj(r1[i] / (eps_occ[i])), (r1[i] / (eps_occ[i])))

                for j in range(no):
                    ij = i*no + j

                    self.Y2[ij] -= r2[ij] / (self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])#(eps_occ[i] + eps_occ[j] - eps_lvir[ij].reshape(1,-1) - eps_lvir[ij].reshape(-1,1) + omega)
                    rms += contract('ab,ab->', np.conj(r2[ij]/(eps_occ[i] + eps_occ[j])), r2[ij]/(eps_occ[i] + eps_occ[j]))

            rms = np.sqrt(rms)
            #end loop

            pseudo = self.local_pseudoresponse(lpertbar, self.Y1, self.Y2)
            pseudodiff = np.abs(pseudo - pseudo_last)
            print(f"Iter {niter:3d}: CC Pseudoresponse = {pseudo.real:.15f} dP = {pseudodiff:.5E} rms = {rms.real:.5E}")

            if ((abs(pseudodiff) < e_conv) and abs(rms) < r_conv):
                print("\nPerturbed wave function converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.Y1, self.Y2, pseudo

            if niter == maxiter:
                print("\nPerturbed wave function not fully converged in %.3f seconds.\n" % (time.time() - solver_start))
                self.psuedoresponse.append(pseudo)
                return self.Y1, self.Y2, pseudo

        #    #diis.add_error_vector(self.X1, self.X2)
        #    #if niter >= start_diis:
        #        #self.X1, self.X2 = diis.extrapolate(self.X1, self.X2)

    def r_X1(self, pertbar, omega):
        start_r_x1 = process_time()
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        X1 = self.X1
        X2 = self.X2
        hbar = self.hbar
        ERI = self.H.ERI

        r_X1 = (pertbar.Avo.T - omega * X1).copy()
        r_X1 += contract('ie,ae->ia', X1, hbar.Hvv)
        r_X1 -= contract('ma,mi->ia', X1, hbar.Hoo)
        r_X1 += 2.0*contract('me,maei->ia', X1, hbar.Hovvo)
        r_X1 -= contract('me,maie->ia', X1, hbar.Hovov)
        r_X1 += contract('me,miea->ia', hbar.Hov, (2.0*X2 - X2.swapaxes(0,1)))
        r_X1 += contract('imef,amef->ia', X2, (2.0* hbar.Hvovv - hbar.Hvovv.swapaxes(2,3)))
        r_X1 -= contract('mnae,mnie->ia', X2, (2.0* hbar.Hooov - hbar.Hooov.swapaxes(0,1)))

        end_r_X1 = process_time()
        # self.time_X1 += end_r_X1 - start_r_x1
        return r_X1

    def lr_X1(self, lpertbar, omega):
        lX1_start = process_time()
        contract = self.contract
        no = self.ccwfn.no
        v = self.ccwfn.v
        hbar = self.hbar
        Avo = lpertbar.Avo
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2 
        ERI = self.H.ERI
        L = self.H.L 
        Sijmn = self.Local.Sijmn
        QL = self.Local.QL


        lr_X1_all = []
        for i in range(no):
            ii = i*no + i

            lr_X1 = (Avo[ii] - omega * self.X1[i]).copy()
            lr_X1 = lr_X1 + contract('e, ae ->a', self.X1[i], hbar.Hvv[ii]) 
            for m in range(no):
                mm = m*no + m 
                mi = m*no + i 
                im = i*no + m
                iimm = ii*(no*no) + mm
                iimi = ii*(no*no) + mi
 
                lr_X1 = lr_X1 - ((self.X1[m] @ Sijmn[iimm].T) * hbar.Hoo[m,i]) 
                
                Hovvo = QL[ii].T @ ERI[m,v,v,i] @ QL[mm]
                Hovov = QL[ii].T @ ERI[m,v,i,v] @ QL[mm]

                ERIovvv = contract('aef, aA -> Aef', ERI[m,v,v,v], QL[ii]) 
                ERIovvv = contract('Aef, eE -> AEf', ERIovvv, QL[mm]) 
                ERIovvv = contract('AEf, fF -> AEF', ERIovvv, QL[ii])  
                Hovvo = Hovvo + contract('f,aef -> ae', t1[i], ERIovvv) 

                ERIvovv = contract('aef, aA -> Aef', ERI[v,m,v,v], QL[ii])
                ERIvovv = contract('Aef, eE -> AEf', ERIvovv, QL[mm]) 
                ERIvovv = contract('AEf, fF -> AEF', ERIvovv, QL[ii])  
                Hovov = Hovov + contract('f,aef -> ae', t1[i], ERIvovv)

                for n in range(no): 
                    nn = n*no + n 
                    _in = i*no + n
                    ni = n*no + i 
                    iinn = ii*(no*no) + nn
                    iiin = ii*(no*no) + _in
                    iini = ii*(no*no) + ni

                    Hovvo = Hovvo - contract('a, e -> ae', Sijmn[iinn] @ t1[n], ERI[m,n,v,i] @ QL[mm]) 
                    Hovvo = Hovvo - contract('fa, ef -> ae', t2[_in] @ Sijmn[iiin].T, QL[mm].T @ ERI[m,n,v,v] @ QL[_in]) 
                    tmp =  contract('a, ef -> aef', Sijmn[iinn] @ t1[n], QL[mm].T  @ ERI[m,n,v,v] @ QL[ii]) 
                    Hovvo = Hovvo - contract('f, aef ->ae', t1[i], tmp) 
                    Hovvo = Hovvo + contract('fa, ef -> ae', t2[ni] @ Sijmn[iini].T, QL[mm].T @ L[m,n,v,v] @ QL[ni])
 
                    Hovov = Hovov - contract('a, e -> ae', Sijmn[iinn] @ t1[n], ERI[m,n,i,v] @ QL[mm]) 
                    Hovov = Hovov - contract('fa, ef -> ae', t2[_in] @ Sijmn[iiin].T, QL[mm].T @ ERI[n,m,v,v] @ QL[_in]) 
                    tmp =  contract('a, ef -> aef', Sijmn[iinn] @ t1[n], QL[mm].T  @ ERI[n,m,v,v] @ QL[ii]) 
                    Hovov = Hovov - contract('f, aef ->ae', t1[i], tmp) 

                lr_X1 = lr_X1 + contract('e, ae -> a', self.X1[m], 2.0 * Hovvo - Hovov) 
       
                lr_X1 = lr_X1 + 2.0 * contract('e, ea -> a', hbar.Hov[mi][m], self.X2[mi] @ Sijmn[iimi].T) 
                lr_X1 = lr_X1 - contract('e, ae -> a', hbar.Hov[mi][m], Sijmn[iimi] @ self.X2[mi]) 
                
                Hvovv = contract('aef, aA -> Aef', ERI[v,m,v,v], QL[ii])
                Hvovv_34swap = contract('Afe, fF -> AFe', Hvovv, QL[im])
                Hvovv_34swap = contract('AFe, eE -> AFE', Hvovv_34swap, QL[im])
                Hvovv = contract('Aef, eE -> AEf', Hvovv, QL[im]) 
                Hvovv = contract('AEf, fF -> AEF', Hvovv, QL[im]) 
       
                for n in range(no): 
                    nn = n*no + n 
                    iinn = ii*(no*no) + nn
 
                    Hvovv = Hvovv - contract('a, ef -> aef', Sijmn[iinn] @ t1[n], QL[im].T @ ERI[n,m,v,v] @ QL[im])
                    Hvovv_34swap = Hvovv_34swap - contract('a, fe -> afe', Sijmn[iinn] @ t1[n], QL[im].T @ ERI[n,m,v,v] @ QL[im])
                
                lr_X1 = lr_X1 + contract('ef, aef -> a', self.X2[im], 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2))   

                for n in range(no):
                    mn = m*no + n
                    iimn = ii*(no*no) + mn

                    Hooov = ERI[m,n,i,v] @ QL[mn]
                    Hooov_12swap = ERI[n,m,i,v] @ QL[mn]    
                    Hooov = Hooov + contract('f, ef -> e', t1[i], QL[mn].T @ ERI[n,m,v,v] @ QL[ii]) 
                    Hooov_12swap = Hooov_12swap + contract('f,ef-> e', t1[i], QL[mn].T @ ERI[m,n,v,v] @ QL[ii])
  
                    lr_X1 = lr_X1 - contract('ae, e -> a', Sijmn[iimn] @ self.X2[mn], 2.0 * Hooov - Hooov_12swap) 
            lr_X1_all.append(lr_X1)

        lX1_end = process_time()
        self.lX1_t += lX1_end - lX1_start
        return lr_X1_all

    def r_X2(self, pertbar, omega):
        start_r_X2 = process_time()
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        X1 = self.X1
        X2 = self.X2
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L

        Zvv = contract('amef,mf->ae', (2.0*hbar.Hvovv - hbar.Hvovv.swapaxes(2,3)), X1)
        #Zvv = contract('amef,mf->ae', (2.0*hbar.Hvovv), X1)
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

        end_r_X2 = process_time()
        self.time_X2 += end_r_X2 - start_r_X2
        return r_X2

    def lr_X2(self, lpertbar, conv_hbar, omega):
        lX2_start = process_time()
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        X1 = self.X1
        X2 = self.X2
        t2 = self.lccwfn.t2
        hbar = self.hbar
        L = self.H.L

        dim = self.Local.dim
        QL = self.Local.QL

        Zoo = np.zeros((no,no))
        for i in range(no):
            for m in range(no):
                im = i*no + m
                for n in range(no):
                    imn = im*no + n
                    _in = i*no + n
                    Zoo[m,i] -= contract('n,n->', (2.0 * hbar.Hmnie[imn] - hbar.Hnmie[imn]), X1[n]) 
                    tmp = contract('ef, eE, fF->EF', L[m,n, v, v], QL[_in], QL[_in])  
                    Zoo[m,i] -= contract('ef,ef->', tmp, X2[_in])

        Zvv = []
        Sijmn = self.Local.Sijmn
        for i in range(no):
            for j in range(no):
                ij = i*no + j
                lZvv = np.zeros((dim[ij], dim[ij])) 
                for m in range(no):
                    mm = m*no + m
                    ijm = ij*no + m

                    lZvv += contract('aef,f->ae', (2.0*hbar.Hamef[ijm] - hbar.Hamfe[ijm].swapaxes(1,2)), X1[m]) 
                    #lZvv += contract('aef,f->ae', (2.0*hbar.Hamef[ijm] - hbar.Hamef[ijm]), X1[m]) 
                    for n in range(no):
                        mn = m*no + n
                        ijmn = ijm * no + n
                        tmp = contract('ef, eE, fF->EF', L[m,n,v,v], QL[ij], QL[mn])
                        lZvv -= contract('ef, af->ae', tmp, Sijmn[ijmn] @ X2[mn]) 
                Zvv.append(lZvv) 
 
        lr2 = []
        tmp_r2 = []
        Sijmj = self.Local.Sijmj 
        Sijim = self.Local.Sijim
        Sijmi = self.Local.Sijmi
        Sijmn = self.Local.Sijmn
        for i in range(no):
            ii = i*no + i
            for j in range(no):
                ij = i*no + j 
                jj = j*no + j
            
                r2 = np.zeros(dim[ij],dim[ij])
     
                #first term
                r2 = lpertbar.Avvoo[ij] - 0.5 *omega *X2[ij] 
  
                #second term
                r2 = r2 + contract('e, abe ->ab', X1[i], hbar.Hvvvo_ij[ij])

                #fifth term
                r2 = r2 + contract('eb,ae->ab', t2[ij], Zvv[ij])
    
                #sixth term 
                r2 = r2 + contract('eb, ae->ab', X2[ij], hbar.Hvv[ij]) 

                #ninth term 
                r2 = r2 + 0.5 * contract('ef,abef->ab', X2[ij], hbar.Hvvvv[ij])
                   
                for m in range(no): 
                    ijm = ij*no + m 
                    mj = m*no + j 
                    im = i*no + m
                    mi = m*no + i 

                    #third term
                    r2 = r2 - contract('a,b->ab', X1[m] @ self.Local.Sijmm[ijm].T, hbar.Hovoo_ij[ijm]) 
 
                    #fourth term
                    r2 = r2 + Zoo[m,i] * self.Local.Sijmj[ijm] @ t2[mj] @ self.Local.Sijmj[ijm].T 

                    #seventh term 
                    r2 = r2 - ((Sijmj[ijm] @ X2[mj] @Sijmj[ijm].T) * hbar.Hoo[m,i]) 

                    #tenth term 
                    r2 = r2 - contract('eb,ae->ab', X2[im] @ Sijim[ijm].T, hbar.Hovov_im[ijm])   

                    #eleventh term
                    #Hmbej = hbar.Hovvo_mi[ijm].transpose() 
                    r2 = r2 - contract('ea,be->ab', X2[im] @ Sijim[ijm].T, hbar.Hovvo_im[ijm]) 

                    #twelveth term
                    r2 = r2 + 2.0 * contract('ea, be->ab', X2[mi] @ Sijmi[ijm].T, hbar.Hmvvj_mi[ijm])

                    #thirteenth term
                    r2 = r2 - contract('ea, be->ab', X2[mi] @ Sijmi[ijm].T, hbar.Hovov_im[ijm]) 

                    for n in range(no):
                        mn = m*no +n 
                        ijmn = ijm*no +n

                        #eight term 
                        r2 = r2 + (0.5 * (Sijmn[ijmn] @ X2[mn] @ Sijmn[ijmn].T) * hbar.Hoooo[m,n,i,j]) 
                tmp_r2.append(r2)

        for ij in range(no*no):
            i = ij // no 
            j = ij % no 
            ji = j*no + i 
   
            lr2.append(tmp_r2[ij].copy() + tmp_r2[ji].copy().transpose())
        lX2_end = process_time()
        self.lX2_t += lX2_end - lX2_start
        return lr2    

    def in_Y1(self, pertbar, X1, X2):
        start_inY1 = process_time()
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

        # <O|A_bar|phi^a_i> good
        r_Y1 = 2.0 * pertbar.Aov.copy()

        # <O|L1(0)|A_bar|phi^a_i> good
        r_Y1 -= contract('im,ma->ia', pertbar.Aoo, l1)
        r_Y1 += contract('ie,ea->ia', l1, pertbar.Avv)

        # <O|L2(0)|A_bar|phi^a_i>
        r_Y1 += contract('imfe,feam->ia', l2, pertbar.Avvvo)
   
        ##can combine the next two to swapaxes type contraction
        r_Y1 -= 0.5 * contract('ienm,mnea->ia', pertbar.Aovoo, l2)
        r_Y1 -= 0.5 * contract('iemn,mnae->ia', pertbar.Aovoo, l2)

        # <O|[Hbar(0), X1]|phi^a_i> good
        r_Y1 +=  2.0 * contract('imae,me->ia', L[o,o,v,v], X1)

        # <O|L1(0)|[Hbar(0), X1]|phi^a_i>
        tmp  = -1.0 * contract('ma,ie->miae', hbar.Hov, l1)
        tmp -= contract('ma,ie->miae', l1, hbar.Hov)
        tmp -= 2.0 * contract('mina,ne->miae', hbar.Hooov, l1)
        tmp += contract('imna,ne->miae', hbar.Hooov, l1)

        ##can combine the next two to swapaxes type contraction
        tmp -= 2.0 * contract('imne,na->miae', hbar.Hooov, l1)
        tmp += contract('mine,na->miae', hbar.Hooov, l1)

        ##can combine the next two to swapaxes type contraction
        tmp += 2.0 * contract('fmae,if->miae', hbar.Hvovv, l1)
        tmp -= contract('fmea,if->miae', hbar.Hvovv, l1)

        ##can combine the next two to swapaxes type contraction
        tmp += 2.0 * contract('fiea,mf->miae', hbar.Hvovv, l1)
        tmp -= contract('fiae,mf->miae', hbar.Hvovv , l1)
        r_Y1 += contract('miae,me->ia', tmp, X1)

        ## <O|L1(0)|[Hbar(0), X2]|phi^a_i> good

        ##can combine the next two to swapaxes type contraction
        tmp  = 2.0 * contract('mnef,nf->me', X2, l1)
        tmp  -= contract('mnfe,nf->me', X2, l1)
        r_Y1 += contract('imae,me->ia', L[o,o,v,v], tmp)
        r_Y1 -= contract('ni,na->ia', cclambda.build_Goo(X2, L[o,o,v,v]), l1)
        r_Y1 += contract('ie,ea->ia', l1, cclambda.build_Gvv(L[o,o,v,v], X2))

        ## <O|L2(0)|[Hbar(0), X1]|phi^a_i> good
        ## can reorganize thesenext four to two swapaxes type contraction
        tmp   = -1.0 * contract('nief,mfna->iema', l2, hbar.Hovov)
        tmp  -= contract('ifne,nmaf->iema', hbar.Hovov, l2)

        t1 = self.ccwfn.t1
        Hovvo = ERI[o,v,v,o].copy()
        Hovvo = Hovvo + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
        Hovvo = Hovvo - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
        Hovvo = Hovvo - contract('jnfb,mnef->mbej', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v]) #self.ccwfn.build_tau(t1, t2)
        Hovvo = Hovvo + contract('njfb,mnef->mbej', t2, L[o,o,v,v])

        tmp  -= contract('inef,mfan->iema', l2, Hovvo)
        tmp  -= contract('ifen,nmfa->iema', Hovvo, l2)

        Hvvvv = ERI[v,v,v,v].copy()
        tmp1 = contract('mb,amef->abef', t1, ERI[v,o,v,v])
        Hvvvv = Hvvvv - (tmp1 + tmp1.swapaxes(0,1).swapaxes(2,3))
        Hvvvv = Hvvvv + contract('mnab,mnef->abef', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])

        ##can combine the next two to swapaxes type contraction
        tmp  += 0.5 * contract('imfg,fgae->iema', l2, Hvvvv)
        tmp  += 0.5 * contract('imgf,fgea->iema', l2, Hvvvv)

        ##can combine the next two to swapaxes type contraction
        tmp  += 0.5 * contract('imno,onea->iema', hbar.Hoooo, l2)
        tmp  += 0.5 * contract('mino,noea->iema', hbar.Hoooo, l2)
        r_Y1 += contract('iema,me->ia', tmp, X1)

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

        ##can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('giea,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))
        r_Y1 += contract('giae,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))
        tmp   = contract('oief,mnef->oimn', l2, X2)
        r_Y1 += contract('oimn,mnoa->ia', tmp, hbar.Hooov)
        tmp   = contract('mofa,mnef->oane', l2, X2)
        r_Y1 += contract('inoe,oane->ia', hbar.Hooov, tmp)
        tmp   = contract('onea,mnef->oamf', l2, X2)
        r_Y1 += contract('miof,oamf->ia', hbar.Hooov, tmp)

        ##can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('mioa,mo->ia', hbar.Hooov, cclambda.build_Goo(X2, l2))
        r_Y1 += contract('imoa,mo->ia', hbar.Hooov, cclambda.build_Goo(X2, l2))

        ##can combine the next two to swapaxes type contraction
        tmp   = -2.0 * contract('imoe,mnef->ionf', hbar.Hooov, X2)
        tmp  += contract('mioe,mnef->ionf', hbar.Hooov, X2)
        r_Y1 += contract('ionf,nofa->ia', tmp, l2)

        end_inY1 = process_time()
        self.time_inY1 = end_inY1 - start_inY1
        return r_Y1

    def in_lY1(self, lpertbar, X1, X2):
        lY1_start = process_time()
        contract = self.contract
        no = self.ccwfn.no
        v = self.ccwfn.v
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2
        hbar = self.hbar
        ERI = self.H.ERI
        L = self.H.L
        Sijmn = self.Local.Sijmn
        QL = self.Local.QL
        mu = lpertbar.pert
        ERIoovv = self.Local.ERIoovv

        in_Y1 = []
        for i in range(no):
            ii = i * no + i

            # <O|A_bar|phi^a_i> good
            r_Y1 = 2.0 * lpertbar.Aov[ii][i].copy()

            # collecting Gvv terms here
            for m in range(no):
                for n in range(no):
                    nn = n * no + n
                    mn = m * no + n
                    iimn = ii * (no * no) + mn

                    # read the resulting index more carefully in Gvv its ae but need ea
                    Gvv = -1.0 * contract('ab,eb -> ea', QL[ii].T @ L[m, n, v, v] @ QL[mn], X2[mn])
                    r_Y1 = r_Y1 + contract('e, ea ->a', Sijmn[iimn].T @ l1[i], Gvv)

            for n in range(no):
                nn = n * no + n
                for m in range(no):
                    for _o in range(no):
                        mo = m * no + _o
                        nnmo = nn * (no * no) + mo

                        Gvv = -1.0 * contract('bc,fc -> fb', Sijmn[nnmo] @ l2[mo], t2[mo])
                        tmp = contract('b, fb ->f', X1[n], Gvv)
                        r_Y1 = r_Y1 + contract('af, f -> a', QL[ii].T @ L[i, n, v, v] @ QL[mo], tmp)

            for m in range(no):
                mm = m * no + m
                for n in range(no):
                    for _o in range(no):
                        _no = n * no + _o
                        iino = ii * (no * no) + _no

                        Gvv = -1.0 * contract('ac,fc -> fa', Sijmn[iino] @ l2[_no], t2[_no])
                        tmp = contract('e, fa -> efa', X1[m], Gvv)
                        r_Y1 = r_Y1 + contract('ef, efa -> a', QL[mm].T @ L[m, i, v, v] @ QL[_no], tmp)

            for m in range(no):
                for n in range(no):
                    mn = m * no + n
                    iimn = ii * (no * no) + mn

                    Gvv = -1.0 * contract('ab, eb -> ea', Sijmn[iimn] @ l2[mn], X2[mn])
                    r_Y1 = r_Y1 + contract('e, ea -> a', hbar.Hov[mn][i], Gvv)

            for m in range(no):
                for n in range(no):
                    mn = m * no + n
                    imn = i * (no * no) + mn

                    Gvv = -1.0 * contract('eb, gb -> ge', X2[mn], l2[mn])
                    r_Y1 = r_Y1 + contract('gea, ge -> a',
                                           -2.0 * hbar.Hvovv_imn[imn] + hbar.Hvovv_imns[imn].swapaxes(1, 2), Gvv)

                    # Goo terms
            for n in range(no):
                for _o in range(no):
                    _no = n * no + _o
                    io = i * no + _o
                    iono = io * (no * no) + _no
                    Goo = contract('ab, ab->', Sijmn[iono] @ t2[_no] @ Sijmn[iono].T, l2[io])

                    for m in range(no):
                        mm = m * no + m
                        tmp_X = X1[m] * Goo
                        r_Y1 = r_Y1 - contract('e, ea -> a', tmp_X, QL[mm].T @ L[m, n, v, v] @ QL[ii])

            for n in range(no):
                for m in range(no):
                    mm = m * no + m
                    X_tmp = contract('e,ea ->a', X1[m], QL[mm].T @ L[m, n, v, v] @ QL[ii])
                    for _o in range(no):
                        _no = n * no + _o
                        io = i * no + _o
                        iono = io * (no * no) + _no
                        Goo = contract('ab, ab ->', Sijmn[iono] @ t2[_no] @ Sijmn[iono].T, l2[io])
                        # r_Y1 = r_Y1 - (Goo * X_tmp)

            for j in range(no):
                jj = j * no + j
                for n in range(no):
                    for m in range(no):
                        nm = n * no + m
                        jm = j * no + m
                        nmjm = nm * (no * no) + jm

                        Goo = contract('ab, ab->', Sijmn[nmjm].T @ t2[nm] @ Sijmn[nmjm], l2[jm])
                        tmp = X1[j] * Goo
                        r_Y1 = r_Y1 - contract('f, af ->a', tmp, QL[ii].T @ L[i, n, v, v] @ QL[jj])

            for m in range(no):
                for n in range(no):
                    mn = m * no + n
                    _in = i * no + n
                    mnin = mn * (no * no) + _in

                    Goo = contract('ab, ab ->', X2[mn], Sijmn[mnin] @ l2[_in] @ Sijmn[mnin].T)
                    r_Y1 = r_Y1 - (Goo * hbar.Hov[ii][m])

            for m in range(no):
                for _o in range(no):
                    oo = _o * no + _o
                    for n in range(no):
                        mn = m * no + n
                        on = _o * no + n
                        mnon = mn * (no * no) + on

                        Goo = contract('ab, ab ->', X2[mn], Sijmn[mnon] @ l2[on] @ Sijmn[mnon].T)
                        # Hooov = ERI[m,i,_o,v] @ QL[ii]
                        # Hooov = Hooov + contract('f, af -> a', t1[_o], QL[ii].T @ ERI[i,m,v,v] @ QL[oo])
                        # Hooov_12swap = ERI[i,m,_o,v] @ QL[ii]
                        # Hooov_12swap = Hooov_12swap + contract('f, af -> a', t1[_o], QL[ii].T @ ERI[m,i,v,v] @ QL[oo])
                        r_Y1 = r_Y1 + ((-2.0 * hbar.Hooov[ii][m, i, _o] + hbar.Hooov[ii][i, m, _o]) * Goo)

                        # <O|L1(0)|A_bar|phi^a_i> good
            for m in range(no):
                mm = m * no + m
                iimm = ii * (no * no) + mm

                r_Y1 = r_Y1 - (lpertbar.Aoo[i, m] * l1[m] @ Sijmn[iimm].T)

            r_Y1 = r_Y1 + contract('e, ea -> a', l1[i], lpertbar.Avv[ii])

            # <O|L2(0)|A_bar|phi^a_i>
            for m in range(no):
                im = i * no + m
                mi = m * no + i
                mm = m * no + m
                iimm = ii * (no * no) + mm
                miim = mi * (no * no) + im
                immm = im * (no * no) + mm
                iim = ii * no + m
                mmi = mm * no + i

                Avvvo = 0
                # for m sum in Avvvo becomes n since m is being used for the og terms
                for n in range(no):
                    nm = n * no + m
                    nmim = nm * (no * no) + im
                    Avvvo = Avvvo - contract('fe,a -> fea', Sijmn[nmim].T @ t2[nm] @ Sijmn[nmim],
                                             mu[n, v].copy() @ QL[ii])
                r_Y1 = r_Y1 + contract('fe, fea -> a', l2[im], Avvvo)

                for n in range(no):
                    nm = n * no + m
                    mn = m * no + n
                    mnii = nm * (no * no) + ii
                    nmmn = nm * (no * no) + mn

                    Aovoo = contract('fe, f->e', t2[nm] @ Sijmn[nmmn], mu[i, v] @ QL[nm])
                    r_Y1 = r_Y1 - 0.5 * contract('e, ea -> a', Aovoo, l2[mn] @ Sijmn[mnii])

                    Aovoo = contract('fe, f->e', t2[mn], mu[i, v] @ QL[mn])
                    r_Y1 = r_Y1 - 0.5 * contract('e, ae -> a', Aovoo, Sijmn[mnii].T @ l2[mn])

                    # <O|[Hbar(0), X1]|phi^a_i>
                Loovv = QL[ii].T @ L[i, m, v, v] @ QL[mm]
                r_Y1 = r_Y1 + 2.0 * contract('ae, e ->a', Loovv, X1[m])

                # <O|L1(0)|[Hbar(0), X1]|phi^a_i>
                tmp = -1.0 * contract('a, e -> ae', hbar.Hov[ii][m], l1[i] @ Sijmn[iimm])
                tmp = tmp - contract('a, e -> ae', Sijmn[iimm] @ l1[m], hbar.Hov[mm][i])

                for n in range(no):
                    nn = n * no + n
                    nnmm = nn * (no * no) + mm
                    nnii = nn * (no * no) + ii

                    # Hooov = ERI[m,i,n,v] @ QL[ii]
                    # Hooov_12swap = ERI[i,m,n,v] @ QL[ii]
                    # Hooov = Hooov + contract('f, af -> a', t1[n], QL[ii].T @ ERI[i,m,v,v] @ QL[nn])
                    # Hooov_12swap = Hooov_12swap + contract('f, af -> a', t1[n], QL[ii].T @ ERI[m,i,v,v] @ QL[nn])

                    tmp = tmp + contract('a,e -> ae', -2.0 * hbar.Hooov[ii][m, i, n] + hbar.Hooov[ii][i, m, n],
                                         l1[n] @ Sijmn[nnmm])

                    # Hooov = ERI[i,m,n,v] @ QL[mm]
                    # Hooov_12swap = ERI[m,i,n,v] @ QL[mm]
                    # Hooov = Hooov + contract('f, ef -> e', t1[n], QL[mm].T @ ERI[m,i,v,v] @ QL[nn])
                    # Hooov_12swap = Hooov_12swap + contract('f, af -> a', t1[n], QL[mm].T @ ERI[i,m,v,v] @ QL[nn])

                    tmp = tmp + contract('e,a -> ae', -2.0 * hbar.Hooov[mm][i, m, n] + hbar.Hooov[mm][m, i, n],
                                         l1[n] @ Sijmn[nnii])

                tmp = tmp + contract('fae, f -> ae', 2.0 * hbar.Hamef[iim] - hbar.Hamfe[iim].swapaxes(1, 2),
                                     l1[i])  # Hvovv_34swap.swapaxes(1,2), l1[i])

                tmp = tmp + contract('fea, f -> ae', 2.0 * hbar.Hamef[mmi] - hbar.Hamfe[mmi].swapaxes(1, 2), l1[m])
                r_Y1 = r_Y1 + contract('ae, e -> a', tmp, X1[m])

                # <O|L1(0)|[Hbar(0), X2]|phi^a_i>
                for n in range(no):
                    nn = n * no + n
                    mn = m * no + n
                    nm = n * no + m
                    ni = n * no + i
                    _in = i * no + n
                    imn = im * no + n
                    _min = mi * no + n
                    nimm = ni * (no * no) + mm
                    nnmm = nn * (no * no) + mm
                    nmii = nm * (no * no) + ii
                    inmm = _in * (no * no) + mm
                    nmmm = nm * (no * no) + mm
                    nnmn = nn * (no * no) + mn
                    iini = ii * (no * no) + ni
                    iinn = ii * (no * no) + nn
                    iimn = ii * (no * no) + mn

                    tmp = 2.0 * contract('ef, f -> e', X2[mn], l1[n] @ Sijmn[nnmn])
                    tmp = tmp - contract('fe, f -> e', X2[mn], l1[n] @ Sijmn[nnmn])
                    Loovv = QL[ii].T @ L[i, m, v, v] @ QL[mn]
                    r_Y1 = r_Y1 + contract('ae, e -> a', Loovv, tmp)

                    Goo = contract('ab, ab ->', X2[nm], self.Local.Loovv[nm][i, m])
                    r_Y1 = r_Y1 - (Goo * l1[n] @ Sijmn[iinn].T)

                    # <O|L2(0)|[Hbar(0), X1]|phi^a_i>
                    # e_mm a_ii
                    tmp1 = -1.0 * contract('ef, fa -> ea', Sijmn[nimm].T @ l2[ni], hbar.Hovov_ni[imn])

                    tmp1 = tmp1 - contract('fe, af -> ea', hbar.Hovov_ni[_min], Sijmn[nmii].T @ l2[nm])

                    tmp1 = tmp1 - contract('ef,fa -> ea', Sijmn[inmm].T @ l2[_in], hbar.Hovvo_ni[imn])

                    tmp1 = tmp1 - contract('fe,fa -> ea', hbar.Hovvo_ni[_min], l2[nm] @ Sijmn[nmii])

                    for _o in range(no):
                        oo = _o * no + _o
                        on = _o * no + n
                        _no = n * no + _o
                        nomm = _no * (no * no) + mm
                        noii = _no * (no * no) + ii
                        onmm = on * (no * no) + mm
                        onii = on * (no * no) + ii

                        tmp1 = tmp1 + 0.5 * hbar.Hoooo[i, m, n, _o] * (Sijmn[onmm].T @ l2[on] @ Sijmn[onii])

                        tmp1 = tmp1 + 0.5 * hbar.Hoooo[m, i, n, _o] * (Sijmn[nomm].T @ l2[_no] @ Sijmn[noii])

                    r_Y1 = r_Y1 + contract('ea,e ->a', tmp1, X1[m])

                tmp1 = 0.5 * contract('fg, fgae -> ea', l2[im], hbar.Hvvvv_im[im])

                tmp1 = tmp1 + 0.5 * contract('gf, fgea -> ea', l2[im], hbar.Hvvvv_im[mi])

                r_Y1 = r_Y1 + contract('ea,e ->a', tmp1, X1[m])

                # for n in range(no):
                # for _o in range(no):
                # _no = n*no + _o
                # io = i*no + _o
                # iono = io*(no*no) + _no

                # Goo = contract('ab, ab ->', Sijmn[iono] @ t2[_no] @ Sijmn[iono].T, l2[io])
                # tmp = X1[m] * Goo
                # r_Y1 = r_Y1 - contract('e, ea ->', tmp, QL[mm].T @ L[m,n,v,v] @ QL[ii])

                # tmp  =  contract('me,ni->meni', X1, cclambda.build_Goo(t2, l2))
                # r_Y1 -= contract('meni,mnea->ia', tmp, L[o,o,v,v])
                # tmp  =  contract('jf,nj->fn', X1, cclambda.build_Goo(t2, l2))
                # r_Y1 -= contract('inaf,fn->ia', L[o,o,v,v], tmp)

                # i__
                # m__
                ## <O|L2(0)|[Hbar(0), X2]|phi^a_i>
                # r_Y1 -= contract('mi,ma->ia', cclambda.build_Goo(X2, l2), hbar.Hov)
                # r_Y1 += contract('ie,ea->ia', hbar.Hov, cclambda.build_Gvv(l2, X2))
                for n in range(no):
                    mn = m * no + n
                    ni = n * no + i
                    immn = im * (no * no) + mn
                    mimn = mi * (no * no) + mn
                    iimn = ii * (no * no) + mn
                    mnni = mn * (no * no) + ni
                    nm = n * no + m
                    nmi = nm * no + i
                    nim = ni * no + m
                    imn = i * (no * no) + mn
                    inm = i * (no * no) + nm
                    # g_im e_mn
                    tmp = contract('fg,ef->ge', Sijmn[immn].T @ l2[im], X2[mn])
                    r_Y1 = r_Y1 - contract('ge, gea -> a', tmp, hbar.Hgnea[imn])  # hbar.Hfobe[nim][:,n,:,:])

                    # g_mi e_mn
                    tmp = contract('fg,ef->ge', Sijmn[mimn].T @ l2[mi], X2[mn])
                    r_Y1 = r_Y1 - contract('ge, gae -> a', tmp, hbar.Hgnae[imn])

                    # g_mn a_ii e_mn f_mn
                    # v^4
                    tmp = contract('ga,ef->gaef', l2[mn] @ Sijmn[iimn].T, X2[mn])
                    r_Y1 = r_Y1 - contract('gef, gaef -> a', hbar.Hvovv_ij[mn][:, i], tmp)

                    # g_ni e_mn f_mn a_ii
                    tmp = contract('gae, ef -> gaf', 2.0 * hbar.Hgnae[inm] - hbar.Hgnea[inm].swapaxes(1, 2), X2[mn])
                    r_Y1 = r_Y1 + contract('fg, gaf -> a', Sijmn[mnni] @ l2[ni], tmp)

                    ##can combine the next two to swapaxes type contraction
                    # r_Y1 -= 2.0 * contract('giea,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))
                    # r_Y1 += contract('giae,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))

                    for _o in range(no):
                        oi = _o * no + i
                        oo = _o * no + _o
                        mo = m * no + _o
                        on = _o * no + n
                        _no = n * no + _o
                        oimn = oi * (no * no) + mn
                        momn = mo * (no * no) + mn
                        moii = mo * (no * no) + ii
                        onmn = on * (no * no) + mn
                        noii = _no * (no * no) + ii
                        onii = on * (no * no) + ii
                        nomn = _no * (no * no) + mn

                        tmp = contract('ef, ef ->', Sijmn[oimn].T @ l2[oi] @ Sijmn[oimn], X2[mn])
                        # Hooov = ERI[m,n,_o,v] @ QL[ii]
                        # Hooov = Hooov + contract('f, af -> a', t1[_o], QL[ii].T @ ERI[n,m,v,v] @ QL[oo])
                        r_Y1 = r_Y1 + (tmp * hbar.Hooov[ii][m, n, _o])

                        tmp = contract('fa, ef -> ae', Sijmn[momn].T @ l2[mo] @ Sijmn[moii], X2[mn])
                        # Hooov = ERI[i,n,_o,v] @ QL[mn]
                        # Hooov = Hooov + contract('f, ef -> e', t1[_o], QL[mn].T @ ERI[n,i,v,v] @ QL[oo])
                        r_Y1 = r_Y1 + contract('e, ae -> a', hbar.Hooov[mn][i, n, _o], tmp)

                        # a_ii f_mn
                        tmp = contract('ea, ef -> af', Sijmn[onmn].T @ l2[on] @ Sijmn[onii], X2[mn])
                        # Hooov = ERI[m,i,_o,v] @ QL[mn]
                        # Hooov = Hooov + contract('e,fe  -> f', t1[_o], QL[mn].T @ ERI[i,m,v,v] @ QL[oo])
                        r_Y1 = r_Y1 + contract('f, af -> a', hbar.Hooov[mn][m, i, _o], tmp)

                        # Hooov = ERI[i,m,_o,v] @ QL[mn]
                        # Hooov = Hooov + contract('f, ef -> e', t1[_o], QL[mn].T @ ERI[m,i,v,v] @ QL[oo])
                        # Hooov_12swap = ERI[m,i,_o,v] @ QL[mn]
                        # Hooov_12swap = Hooov_12swap + contract('f, ef -> e', t1[_o], QL[mn].T @ ERI[i,m,v,v] @ QL[oo])
                        tmp = contract('e, ef -> f', -2.0 * hbar.Hooov[mn][i, m, _o] + hbar.Hooov[mn][m, i, _o], X2[mn])
                        r_Y1 = r_Y1 + contract('f, fa -> a', tmp, Sijmn[nomn].T @ l2[_no] @ Sijmn[noii])
                        # r_Y1 = r_Y1 + contract('e, ea->a', l1[i], Gvv)
            in_Y1.append(r_Y1)
        lY1_end = process_time()
        self.lY1_t += lY1_end - lY1_start
        return in_Y1

    #def lr_Y1(self, lpertbar, omega):
    #    contract = self.contract 
    #    o = self.ccwfn.o
    #    v = self.ccwfn.v
    #  
    #    #imhomogenous terms
    #    r_Y1 = self.im_Y1.copy()
    #    
    #    return r_Y1 

    #    #for i in range(self.ccwfn.no):
    #        #ii = i* self.ccwfn.no + i 
    #        #QL = self.ccwfn.Local.Q[ii] @ self.ccwfn.Local.L[ii]
    #        #print("R-Y1", i, r_Y1[i] @ QL) 
    #    return r_Y1
    #
    # def in_lY1(self, lpertbar, X1, X2):
    #     contract = self.contract
    #     no = self.ccwfn.no

    #     l1 = self.cclambda.l1
    #     l2 = self.cclambda.l2
    #     cclambda = self.cclambda
    #     t2 = self.ccwfn.t2
    #     hbar = self.hbar
    #     L = self.H.L

    #     # Inhomogenous terms appearing in Y1 equations
    #     #seems like these imhomogenous terms are computing at the beginning and not involve in the iteration itself
    #     #may require moving to a sperate function
    #     
    #     in_Y1 = []
    #     for i in range(no): 
    #         ii = i * no + i 

    #         # <O|A_bar|phi^a_i> good
    #         r_Y1 = 2.0 * lpertbar.Aov[ii][i].copy()
    #         #print("r_Y1", i, r_Y1)
    #         in_Y1.append(r_Y1)
 
    #     return in_Y1

    def lr_Y1(self, lpertbar, omega):
        lY1_start = process_time()
        contract = self.contract
        hbar = self.cchbar
        no = self.ccwfn.no
        o = self.ccwfn.o
        v = self.ccwfn.v
        F = self.H.F
        ERI = self.H.ERI
        L = self.H.L
        QL = self.Local.QL
        Sijmn = self.Local.Sijmn
        t1 = self.lccwfn.t1
        t2 = self.lccwfn.t2
        Y1 = self.Y1
        Y2 = self.Y2
        #imhomogenous terms
        r_Y1 = self.im_Y1.copy()

        for i in range(no):
            ii = i*no + i

            r_Y1[i] = r_Y1[i] + (omega * Y1[i])
            r_Y1[i] = r_Y1[i] + contract('e, ea -> a', Y1[i], hbar.Hvv[ii])
            r_lY1 = 0

            #collecting Gvv terms here
            for m in range(no):
                for n in range(no):
                    mn = m*no + n
                    iimn = ii*(no*no) + mn

                    Hvovv = contract('efa, eE -> Efa', ERI[v,i,v,v], QL[mn])
                    Hvovv_34swap = contract('Eaf, aA -> EAf', Hvovv, QL[ii])
                    Hvovv_34swap = contract('EAf, fF -> EAF', Hvovv_34swap, QL[mn])
                    Hvovv = contract('Efa, fF -> EFa', Hvovv, QL[mn])
                    Hvovv = contract('EFa, aA -> EFA', Hvovv, QL[ii])

                    for _o in range(no):
                        oo = _o*no +_o
                        mnoo = mn*(no*no) + oo

                        Hvovv = Hvovv - contract('g, ea ->gea', Sijmn[mnoo] @ t1[_o], QL[mn].T @ ERI[_o,i,v,v] @ QL[ii])
                        Hvovv_34swap = Hvovv_34swap - contract('g, ea ->gea', Sijmn[mnoo] @ t1[_o], QL[ii].T @ ERI[_o,i,v,v] @ QL[mn])

                    Gvv = -1.0 * contract('fb, eb -> ef', t2[mn], Y2[mn])
                    r_lY1 = r_lY1 + contract('efa, ef -> a', -2.0 * Hvovv + Hvovv_34swap.swapaxes(1,2), Gvv)

            for m in range(no):
                for _o in range(no):
                    mo = m*no + _o
                    for n in range(no):
                        nn = n*no + n
                        _no = n*no + _o
                        mono = mo*(no*no) + _no

                        Goo = contract('ab, ab ->', Sijmn[mono].T @ t2[mo] @ Sijmn[mono], Y2[_no])
                        Hooov = ERI[m,i,n,v] @ QL[ii]
                        Hooov = Hooov + contract('f, af -> a', t1[n], QL[ii].T @ ERI[i,m,v,v] @ QL[nn])
                        Hooov_12swap = ERI[i,m,n,v] @ QL[ii]
                        Hooov_12swap = Hooov_12swap + contract('f, af -> a', t1[n], QL[ii].T @ ERI[m,i,v,v] @ QL[nn])
                        r_lY1 = r_lY1 + ((-2.0 * Hooov + Hooov_12swap) * Goo)

            for m in range(no):
                mm = m*no + m
                im = i*no + m
                iimm = ii*(no*no) + mm
                mmim = mm*(no*no) + im

                Hoo = F[i,m].copy()
                Hoo = Hoo + contract('e, e ->',  t1[m], self.Local.Fov[mm][i])
                for n in range(no):
                    nn = n*no + n
                    mn = m*no + n
                    Hoo = Hoo + contract('e, e ->', t1[n], L[i,n,m,v] @ QL[nn])
                    Hoo = Hoo + contract('ef, ef ->', t2[mn], self.Local.Loovv[mn][i,n])
                    tmp = contract('f, ef -> e', t1[n], QL[mm].T @ L[i,n,v,v] @ QL[nn])
                    Hoo = Hoo + contract('e, e->', t1[m], tmp)

                r_lY1 = r_lY1 - (Hoo * Y1[m] @ Sijmn[iimm].T)

                #e_mm a_ii
                Hovvo = QL[mm].T @ ERI[i,v,v,m] @ QL[ii]
                ERIovvv = contract('eaf, eE -> Eaf', ERI[i,v,v,v], QL[mm])
                ERIovvv = contract('Eaf, aA-> EAf', ERIovvv, QL[ii])
                ERIovvv = contract('EAf, fF -> EAF', ERIovvv, QL[mm])
                Hovvo = Hovvo + contract('f, eaf -> ea', t1[m], ERIovvv)

                for n in range(no):
                    nn = n*no + n
                    mn = m*no + n
                    nm = n*no + m
                    mmnn = mm*(no*no) + nn
                    mmmn = mm*(no*no) + mn
                    mmnm = mm*(no*no) + nm

                    Hovvo = Hovvo - contract('e, a-> ea', Sijmn[mmnn] @ t1[n], ERI[i,n,v,m] @ QL[ii])
                    Hovvo = Hovvo - contract('fe, af -> ea', t2[mn] @ Sijmn[mmmn].T, QL[ii].T @ ERI[i,n,v,v] @ QL[mn])
                    tmp = contract('e, af -> eaf', Sijmn[mmnn] @ t1[n], QL[ii].T @ ERI[i,n,v,v] @ QL[mm])
                    Hovvo = Hovvo - contract('f, eaf -> ea', t1[m], tmp)
                    Hovvo = Hovvo + contract('fe, af -> ea', t2[nm] @ Sijmn[mmnm].T, QL[ii].T @ L[i,n,v,v] @ QL[nm])

                Hovov = QL[mm].T @ ERI[i,v,m,v] @ QL[ii]
                ERIvovv = contract('eaf, eE -> Eaf', ERI[v,i,v,v], QL[mm])
                ERIvovv = contract('Eaf, aA-> EAf', ERIvovv, QL[ii])
                ERIvovv = contract('EAf, fF -> EAF', ERIvovv, QL[mm])

                Hovov = Hovov + contract('f, eaf -> ea', t1[m], ERIvovv)

                for n in range(no):
                    nn = n*no + n
                    mn = m*no + n
                    nm = n*no + m
                    mmnn = mm*(no*no) + nn
                    mmmn = mm*(no*no) + mn
                    mmnm = mm*(no*no) + nm

                    Hovov = Hovov - contract('e, a-> ea', Sijmn[mmnn] @ t1[n], ERI[i,n,m,v] @ QL[ii])
                    Hovov = Hovov - contract('fe, af -> ea', t2[mn] @ Sijmn[mmmn].T, QL[ii].T @ ERI[n,i,v,v] @ QL[mn])
                    tmp = contract('e, af -> eaf', Sijmn[mmnn] @ t1[n], QL[ii].T @ ERI[n,i,v,v] @ QL[mm])
                    Hovov = Hovov - contract('f, eaf -> ea', t1[m], tmp)

                r_lY1 = r_lY1 + contract('ea,e -> a', 2.0 * Hovvo - Hovov, Y1[m])

                ##e_im f_im a_ii
                ##amp priority of e_im instead of e_mm
                r_lY1 = r_lY1 + contract('ef, efa -> a', Y2[im], hbar.Hvvvo_im[im])

                for n in range(no):
                    mn = m*no + n
                    iimn = ii*(no*no) + mn
                    mmmn = mm*(no*no) + mn
                    imn = im*no + n

                    Hovoo = QL[mn].T @ ERI[i,v,m,n]
                    r_lY1 = r_lY1 - contract('e, ae -> a', hbar.Hovoo_mn[imn], Sijmn[iimn] @ Y2[mn])
            r_Y1[i] = r_Y1[i] + r_lY1
        lY1_end = process_time()
        self.lY1_t += lY1_end - lY1_start
        return r_Y1

    def r_Y1(self, pertbar, omega):
        start_r_Y1 = process_time()
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
        F = self.H.F
        ERI = self.H.ERI
        #imhomogenous terms
        r_Y1 = self.im_Y1.copy()
        
        #homogenous terms appearing in Y1 equations
       
        r_Y1 += omega * Y1
        r_Y1 += contract('ie,ea->ia', Y1, hbar.Hvv)
        r_Y1 -= contract('im,ma->ia', hbar.Hoo, Y1)
        r_Y1 += 2.0 * contract('ieam,me->ia', hbar.Hovvo, Y1)
       
        t1 = self.ccwfn.t1 
        Hovov = ERI[o,v,o,v].copy()
        Hovov = Hovov + contract('jf,bmef->mbje', t1, ERI[v,o,v,v])
        Hovov = Hovov - contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
        Hovov = Hovov - contract('jnfb,nmef->mbje', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,v])

        r_Y1 -= contract('iema,me->ia', Hovov, Y1)

        t2 = self.ccwfn.t2
        Hvvvo = ERI[v,v,v,o].copy()
        Hvvvo = Hvvvo - contract('me,miab->abei', hbar.Hov, t2)
        Hvvvo = Hvvvo + contract('if,abef->abei', t1, hbar.Hvvvv)
        Hvvvo = Hvvvo + contract('mnab,mnei->abei', self.ccwfn.build_tau(t1, t2), ERI[o,o,v,o])
        Hvvvo = Hvvvo - contract('imfa,bmfe->abei', t2, ERI[v,o,v,v])
        Hvvvo = Hvvvo - contract('imfb,amef->abei', t2, ERI[v,o,v,v])
        Hvvvo = Hvvvo + contract('mifb,amef->abei', t2, L[v,o,v,v])

        tmp1 = ERI[v,o,v,o].copy()
        tmp1 = tmp1 - contract('infa,mnfe->amei', t2, ERI[o,o,v,v])
        Hvvvo = Hvvvo - contract('mb,amei->abei', t1, tmp1)

        tmp1 = ERI[v,o,o,v].copy()
        tmp1 = tmp1 - contract('infb,mnef->bmie', t2, ERI[o,o,v,v])
        tmp1 = tmp1 + contract('nifb,mnef->bmie', t2, L[o,o,v,v])
        Hvvvo = Hvvvo - contract('ma,bmie->abei', t1, tmp1)

        r_Y1 += contract('imef,efam->ia', Y2, hbar.Hvvvo)
        r_Y1 -= contract('iemn,mnae->ia', hbar.Hovoo, Y2)

        ##can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('eifa,ef->ia', hbar.Hvovv, cclambda.build_Gvv(t2, Y2))
        r_Y1 += contract('eiaf,ef->ia', hbar.Hvovv, cclambda.build_Gvv(t2, Y2))

        ##can combine the next two to swapaxes type contraction
        r_Y1 -= 2.0 * contract('mina,mn->ia', hbar.Hooov, cclambda.build_Goo(t2, Y2))
        r_Y1 += contract('imna,mn->ia', hbar.Hooov, cclambda.build_Goo(t2, Y2))

        end_r_Y1 = process_time()
        # self.time_Y1 += end_r_Y1 - start_r_Y1
        return r_Y1
   
    def in_Y2(self, pertbar, X1, X2):
        start_inY2 = process_time()
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

        ## <O|L2(0)|[Hbar(0), X1]|phi^ab_ij> 
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

        ##swapaxes contraction
        tmp   = 2.0 * contract('me,fiea->mfia', X1, hbar.Hvovv)
        tmp  -= contract('me,fiae->mfia', X1, hbar.Hvovv)
        r_Y2 += contract('mfia,jmbf->ijab', tmp, l2)
        tmp   = contract('me,jmna->ejna', X1, hbar.Hooov)
        r_Y2 += contract('ineb,ejna->ijab', l2, tmp)

        tmp   = contract('me,mjna->ejna', X1, hbar.Hooov)
        r_Y2 += contract('nieb,ejna->ijab', l2, tmp)
        tmp   = contract('me,nmba->enba', X1, l2)
        r_Y2 += contract('jine,enba->ijab', hbar.Hooov, tmp)

        ##swapaxes
        tmp   = 2.0 * contract('me,mina->eina', X1, hbar.Hooov)
        tmp  -= contract('me,imna->eina', X1, hbar.Hooov)
        r_Y2 -= contract('eina,njeb->ijab', tmp, l2)

        ##swapaxes
        tmp   = 2.0 * contract('me,imne->in', X1, hbar.Hooov)
        tmp  -= contract('me,mine->in', X1, hbar.Hooov)
        r_Y2 -= contract('in,jnba->ijab', tmp, l2)

        ## <O|L2(0)|[Hbar(0), X2]|phi^ab_ij>
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

        ##temp intermediate?
        r_Y2 -= contract('in,jnba->ijab', cclambda.build_Goo(L[o,o,v,v], X2), l2)
        r_Y2 += contract('ijfb,af->ijab', l2, cclambda.build_Gvv(X2, L[o,o,v,v]))
        r_Y2 += contract('ijae,be->ijab', L[o,o,v,v], cclambda.build_Gvv(X2, l2))
        r_Y2 -= contract('imab,jm->ijab', L[o,o,v,v], cclambda.build_Goo(l2, X2))
        tmp   = contract('nifb,mnef->ibme', l2, X2)
        r_Y2 -= contract('ibme,mjea->ijab', tmp, L[o,o,v,v])
        tmp   = 2.0 * contract('njfb,mnef->jbme', l2, X2)
        r_Y2 += contract('imae,jbme->ijab', L[o,o,v,v], tmp)

        end_inY2 = process_time()
        self.time_inY2 = end_inY2 - start_inY2
        return r_Y2

    def in_lY2(self, lpertbar, X1, X2):
        lY2_start = process_time()
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        Y1 = self.Y1
        Y2 = self.Y2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.ccwfn.t2
        hbar = self.hbar
        L = self.H.L
        ERI = self.H.ERI

        in_Y2 = []

        QL = self.Local.QL
        Sijii = self.Local.Sijii
        Sijjj = self.Local.Sijjj
        Sijmj = self.Local.Sijmj
        Sijmm = self.Local.Sijmm
        Sijim = self.Local.Sijim
        Sijmn = self.Local.Sijmn

        # G_in = np.zeros((no,no))
        # for i in range(self.no):
        # for j in range(self.no):
        # ij = i*self.no + j

        # for n in range(self.no):
        # nj = n*self.no + j
        # ijn = ij*self.no + n

        # tmp = self.Local.Loovv[nj][i,j]
        # G_in[i,n] += contract('ab,ab->',tmp,l2[nj])

        # Goo_LX = np.zeros((self.no,self.no))
        # for i in range(self.no):
        # for j in range(self.no):
        # ij = i*self.no + j

        # for m in range(self.no):
        # mj = m*self.no + j
        # ijm = ij*self.no + m

        # tmp = Sijmj[ijm] @ X2[mj]
        # tmp = tmp @ Sijmj[ijm].T
        # Goo_LX[i,m] += contract('ab,ab->',self.Local.Loovv[ij][i,j], tmp)

        # Gvv terms needed for Expression 5, Term 7
        # for i in range(no):
        # for j in range(no):
        # ij = i*no + j
        # Gvv term needed for Expression 5, Term 7
        # self.Gaf.append(-1.0 * contract('fb, ab-> af', X2[ij], self.Local.Loovv[ij][i,j]))
        # print("Gaf", self.Gaf[ij].shape)

        # Gvv term needed for Expression 5, Term 8
        # self.Gae.append(-1.0 * contract('eb, ab->ae', X2[ij], l2[ij]))

        for i in range(no):
            ii = i * no + i
            for j in range(no):
                ij = i * no + j
                jj = j * no + j

                # <O|L1(0)|A_bar|phi^ab_ij>, Eqn 162
                r_Y2 = 2.0 * contract('a,b->ab', l1[i] @ Sijii[ij].T, lpertbar.Aov[ij][j].copy())
                r_Y2 = r_Y2 - contract('a,b->ab', l1[j] @ Sijjj[ij].T, lpertbar.Aov[ij][i].copy())

                # <O|L2(0)|A_bar|phi^ab_ij>, Eqn 163
                r_Y2 += contract('eb,ea->ab', l2[ij], lpertbar.Avv[ij])

                # collecting Gvv terms here
                for m in range(no):
                    for n in range(no):
                        mn = m * no + n
                        ijmn = ij * (no * no) + mn

                        Gvv = -1.0 * contract('fe, ae ->af', X2[mn], QL[ij].T @ L[m, n, v, v] @ QL[mn])
                        r_Y2 = r_Y2 + contract('fb, af -> ab', Sijmn[ijmn].T @ l2[ij], Gvv)

                for m in range(no):
                    for n in range(no):
                        mn = m * no + n
                        ijmn = ij * (no * no) + mn

                        Gvv = -1.0 * contract('ef, bf -> be', X2[mn], Sijmn[ijmn] @ l2[mn])
                        r_Y2 = r_Y2 + contract('ae, be -> ab', QL[ij].T @ L[i, j, v, v] @ QL[mn], Gvv)

                        # last Goo here
                for m in range(no):
                    for n in range(no):
                        mn = m * no + n
                        jn = j * no + n
                        mnjn = mn * (no * no) + jn

                        Goo = contract('ab, ab ->', Sijmn[mnjn] @ l2[jn] @ Sijmn[mnjn].T, X2[mn])
                        r_Y2 = r_Y2 - (self.Local.Loovv[ij][i, m] * Goo)

                for m in range(no):
                    mj = m * no + j
                    ijm = ij * no + m

                    tmp = Sijmj[ijm] @ l2[mj] @ Sijmj[ijm].T
                    r_Y2 = r_Y2 - lpertbar.Aoo[i, m] * tmp

                # <O|L1(0)|[Hbar(0), X1]|phi^ab_ij>, Eqn 164
                for m in range(no):
                    ijm = ij * no + m
                    mm = m * no + m
                    iim = ii * no + m

                    tmp = contract('e,a->ea', X1[m], (l1[j] @ Sijjj[ij].T))
                    tmp1 = contract('eb, eE, bB ->EB', L[m, i, v, v], QL[mm], QL[ij])
                    r_Y2 = r_Y2 - contract('eb, ea-> ab', tmp1, tmp)

                    tmp = contract('e,b->eb', X1[m], (l1[m] @ Sijmm[ijm].T))
                    tmp1 = contract('ae, aA, eE ->AE', L[i, j, v, v], QL[ij], QL[mm])
                    r_Y2 = r_Y2 - contract('ae, eb-> ab', tmp1, tmp)

                    tmp = contract('e,e->', X1[m], (l1[i] @ Sijmm[iim]))
                    r_Y2 = r_Y2 - tmp * self.Local.Loovv[ij][j, m].swapaxes(0, 1)

                    tmp = 2.0 * contract('e,b ->eb', X1[m], (l1[j] @ Sijjj[ij].T))
                    tmp1 = contract('ae, aA, eE ->AE', L[i, m, v, v], QL[ij], QL[mm])
                    r_Y2 = r_Y2 + contract('ae, eb-> ab', tmp1, tmp)

                # <O|L2(0)|[Hbar(0), X1]|phi^ab_ij>, Eqn 165
                for m in range(no):
                    mm = m * no + m
                    ijm = ij * no + m
                    jm = j * no + m
                    im = i * no + m

                    tmp = contract('e,a-> ea', X1[m], hbar.Hov[ij][m])
                    r_Y2 = r_Y2 - contract('eb,ea->ab', Sijmm[ijm].T @ l2[ij], tmp)

                    tmp = contract('e,e->', X1[m], hbar.Hov[mm][i])
                    r_Y2 = r_Y2 - tmp * (Sijmj[ijm] @ l2[jm] @ Sijmj[ijm].T).swapaxes(0, 1)

                    # may need to double-check this one
                    tmp = contract('e,ef->f', X1[m], Sijmm[ijm].T @ l2[ij])
                    r_Y2 = r_Y2 - contract('f, fba -> ab', tmp, hbar.Hvovv_ij[ij][:, m, :, :])

                    tmp = contract('e,bf->ebf', X1[m], Sijim[ijm] @ l2[im])
                    r_Y2 = r_Y2 - contract('ebf, fea->ab', tmp, hbar.Hfjea[ijm])

                    tmp = contract('e,fa->efa', X1[m], l2[jm] @ Sijmj[ijm].T)
                    r_Y2 = r_Y2 - contract('fbe, efa->ab', hbar.Hfibe[ijm], tmp)

                    tmp = contract('e, fae -> fa', X1[m], 2.0 * hbar.Hfmae[ijm] - hbar.Hfmea[ijm].swapaxes(1, 2))
                    r_Y2 = r_Y2 + contract('fb,fa->ab', l2[ij], tmp)

                    tmp = contract('e, fea -> fa', X1[m], 2.0 * hbar.Hfieb[ijm] - hbar.Hfibe[ijm].swapaxes(1, 2))
                    r_Y2 = r_Y2 + contract('fa,bf->ab', tmp, Sijmj[ijm] @ l2[jm])

                    for n in range(no):
                        ijmn = ijm * no + n
                        _in = i * no + n
                        inm = _in * no + m
                        ijn = ij * no + n
                        ni = n * no + i
                        nim = ni * no + m
                        nm = n * no + m
                        ijnm = ij * (no * no) + nm
                        nj = n * no + j
                        njm = nj * no + m
                        jn = j * no + n

                        imn = im * no + n

                        tmp = contract('e,a -> ea', X1[m], hbar.Hjmna[ijmn])
                        tmp1 = Sijmm[inm].T @ l2[_in] @ Sijim[ijn].T
                        r_Y2 = r_Y2 + contract('eb,ea->ab', tmp1, tmp)

                        tmp = contract('e,a -> ea', X1[m], hbar.Hmjna[ijmn])
                        tmp1 = Sijmm[nim].T @ l2[ni] @ Sijim[ijn].T
                        r_Y2 = r_Y2 + contract('eb,ea->ab', tmp1, tmp)

                        tmp = Sijmn[ijnm] @ l2[nm] @ Sijmn[ijnm].T
                        tmp = contract('e,ba->eba', X1[m], tmp)
                        r_Y2 = r_Y2 + contract('eba,e->ab', tmp, hbar.Hjine[ijmn])

                        tmp = contract('e,a->ea', X1[m], 2.0 * hbar.Hmine[ijmn] - hbar.Himne[ijmn])
                        tmp1 = Sijmm[njm].T @ l2[nj] @ Sijmj[ijn].T
                        r_Y2 = r_Y2 - contract('ea, eb->ab', tmp, tmp1)

                        tmp = contract('e,e->', X1[m], 2.0 * hbar.Himne_mm[imn] - hbar.Hmine_mm[imn])
                        tmp1 = Sijmj[ijn] @ l2[jn] @ Sijmj[ijn].T
                        r_Y2 = r_Y2 - tmp * tmp1.swapaxes(0, 1)

                # <O|L2(0)|[Hbar(0), X2]|phi^ab_ij>, Eqn 174
                Gin = np.zeros((no, no))
                for m in range(no):
                    ijm = ij * no + m
                    mi = m * no + i
                    im = i * no + m
                    mj = m * no + j
                    for n in range(no):
                        mn = m * no + n
                        ijmn = ijm * (no) + n
                        imn = i * (no * no) + mn
                        min = mi * no + n
                        mni = mn * no + i
                        nm = n * no + m
                        inm = i * (no * no) + nm
                        mjn = mj * no + n
                        jn = j * no + j
                        ijn = i * (no * no) + jn
                        ni = n * no + i
                        nim = ni * no + m
                        nj = n * no + j
                        njm = nj * no + m
                        mnni = mn * (no * no) + ni
                        ijni = ij * (no * no) + ni
                        ijnj = ij * (no * no) + nj
                        mnnj = mn * (no * no) + nj

                        tmp = Sijmn[ijmn].T @ l2[ij] @ Sijmn[ijmn]
                        tmp = 0.5 * contract('ef,ef->', tmp, X2[mn])
                        r_Y2 = r_Y2 + tmp * self.Local.ERIoovv[ij][m, n]

                        tmp = Sijmn[ijmn] @ l2[mn] @ Sijmn[ijmn].T
                        tmp1 = 0.5 * contract('fe,ef->', self.Local.ERIoovv[mn][i, j], X2[mn])
                        r_Y2 = r_Y2 + tmp1 * tmp.swapaxes(0, 1)

                        tmp = Sijim[min].T @ l2[mi] @ Sijim[ijm].T
                        tmp = contract('fb, ef-> be', tmp, X2[mn])
                        r_Y2 = r_Y2 + contract('be, ae->ab', tmp, QL[ij].T @ ERI[j, n, v, v] @ QL[mn])

                        tmp = Sijim[min].T @ l2[im] @ Sijim[ijm].T
                        tmp = contract('fb, ef-> be', tmp, X2[mn])
                        r_Y2 = r_Y2 + contract('be, ae->ab', tmp, QL[ij].T @ ERI[n, j, v, v] @ QL[mn])

                        tmp = Sijim[mjn].T @ l2[mj] @ Sijmj[ijm].T
                        tmp = contract('fb, ef-> be', tmp, X2[mn])
                        r_Y2 = r_Y2 - contract('be, ae->ab', tmp, QL[ij].T @ L[i, n, v, v] @ QL[mn])

                        # Expression 5, Term 10
                        tmp = Sijmn[mnni] @ l2[ni] @ Sijmn[ijni].T
                        tmp = contract('fb, ef-> be', tmp, X2[mn])
                        r_Y2 = r_Y2 - contract('be,ea->ab', tmp, QL[mn].T @ L[m, j, v, v] @ QL[ij])

                        # Expression 5, Term 11
                        tmp = Sijmn[mnnj] @ l2[nj] @ Sijmn[ijnj].T
                        tmp = 2.0 * contract('fb, ef-> be', tmp, X2[mn])
                        r_Y2 = r_Y2 + contract('ae, be-> ab', QL[ij].T @ L[i, m, v, v] @ QL[mn], tmp)

                        # Goo term for Term 6
                        Gin[i, n] += contract('ef,ef->', QL[nm].T @ L[i, m, v, v] @ QL[nm], X2[nm])

                for n in range(no):
                    ijn = ij * no + n
                    jn = j * no + n

                    # Term 6
                    tmp = Sijmj[ijn] @ l2[jn] @ Sijmj[ijn].T
                    r_Y2 = r_Y2 - Gin[i, n] * tmp.swapaxes(0, 1)

                in_Y2.append(r_Y2)
        lY2_end = process_time()
        self.lY2_t += lY2_end - lY2_start
        return in_Y2


    def r_Y2(self, pertbar, omega):
        start_r_Y2 = process_time()
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

        end_r_Y2 = process_time()
        self.time_Y2 = end_r_Y2 - start_r_Y2
        return r_Y2

    def lr_Y2(self, lpertbar, omega):
        lY2_start = process_time()
        contract = self.contract
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        Y1 = self.Y1
        Y2 = self.Y2
        l1 = self.cclambda.l1
        l2 = self.cclambda.l2
        cclambda = self.cclambda
        t2 = self.lccwfn.t2
        hbar = self.hbar
        L = self.H.L
        ERI = self.H.ERI

        in_Y2 = []

        QL = self.Local.QL 
        Sijii = self.Local.Sijii
        Sijjj = self.Local.Sijjj
        Sijmj = self.Local.Sijmj
        Sijmm = self.Local.Sijmm
        Sijim = self.Local.Sijim
        Sijmn = self.Local.Sijmn

        tmp_Y2 = []
        lr_Y2 = []

        #build Goo and Gvv here
        Goo = self.cclambda.build_lGoo(t2, Y2)
        Gvv = self.cclambda.build_lGvv(t2, Y2)
 
        for i in range(no):
            for j in range(no):
                ij = i*no + j 
           
                #first term
                r_Y2 = self.im_Y2[ij].copy()

                #second term
                r_Y2 = r_Y2 + 0.5 * omega * self.Y2[ij].copy()

                #third term
                tmp1 = 2.0 * Sijii[ij] @ Y1[i]  
                r_Y2 = r_Y2 + contract('a,b->ab', tmp1, hbar.Hov[ij][j])
  
                #fourth term
                tmp = Sijjj[ij] @ Y1[j] 
                r_Y2 = r_Y2 - contract('a,b->ab', tmp, hbar.Hov[ij][i]) 

                #fifth term 
                r_Y2 = r_Y2 + contract('eb, ea -> ab', Y2[ij], hbar.Hvv[ij]) 

                #eigth term 
                r_Y2 = r_Y2 + 0.5 * contract('ef,efab->ab', Y2[ij], hbar.Hvvvv[ij])
 
                #ninth term 
                r_Y2 = r_Y2 + 2.0 * contract('e,eab->ab', Y1[i], hbar.Hvovv_ii[ij][:,j,:,:]) 
                
                #tenth term 
                r_Y2 = r_Y2 - contract('e,eba->ab', Y1[i], hbar.Hvovv_ii[ij][:,j,:,:])

                for m in range(no):
                    mi = m*no + i
                    mj = m*no + j 
                    ijm = ij*no + m
 
                    #sixth term
                    tmp = Sijmj[ijm] @ Y2[mj] @ Sijmj[ijm].T 
                    r_Y2 = r_Y2 - hbar.Hoo[i,m] * tmp  

                    #eleventh term and twelve term  
                    r_Y2 = r_Y2 - contract('b,a->ab', Sijmm[ijm] @ Y1[m], 2.0 * hbar.Hjiov[ij][m] - hbar.Hijov[ij][m]) 
        
                    #thirteenth term and fourteenth term  
                    r_Y2 = r_Y2 + contract('ea,eb -> ab', 2.0 * hbar.Hovvo_mj[ijm] - hbar.Hovov_mj[ijm], Y2[mj] @ Sijmj[ijm].T)  
 
                    #fifteenth term
                    tmp = Sijim[ijm] @ Y2[mi]
                    r_Y2 = r_Y2 - contract('be, ea->ab', Sijim[ijm] @ Y2[mi], hbar.Hovov_mi[ijm])  

                    #sixteenth term 
                    r_Y2 = r_Y2 - contract('eb, ea -> ab',  Y2[mi] @ Sijim[ijm].T, hbar.Hovvo_mi[ijm])  
                    
                    #eighteenth term
                    r_Y2 = r_Y2 - Goo[m,i] * self.Local.Loovv[ij][m,j] 

                    for n in range(no):
                        mn = m*no + n
                        ijmn = ij*(no*no) + mn 
 
                        #seventh term
                        tmp = Sijmn[ijmn] @ Y2[mn] @ Sijmn[ijmn].T 
                        r_Y2 = r_Y2 + 0.5 * hbar.Hoooo[i,j,m,n] * tmp 
                         
                        #seventeenth term
                        tmp = QL[mn].T @ L[i,j,v,v] @ QL[ij]
                        r_Y2 = r_Y2 + contract('eb,ae->ab', tmp, Sijmn[ijmn] @ Gvv[mn])   

                tmp_Y2.append(r_Y2)

        for ij in range(no*no):
            i = ij // no
            j = ij % no
            ji = j*no + i

            lr_Y2.append(tmp_Y2[ij].copy() + tmp_Y2[ji].copy().transpose())
        lY2_end = process_time()
        self.lY2_t += lY2_end - lY2_start
        return lr_Y2

    def pseudoresponse(self, pertbar, X1, X2):
        # pseudoresponse_start = process_time()
        contract = self.ccwfn.contract
        polar1 = 2.0 * contract('ai,ia->', np.conj(pertbar.Avo), X1)
        polar2 = 2.0 * contract('ijab,ijab->', np.conj(pertbar.Avvoo), (2.0*X2 - X2.swapaxes(2,3)))
        pseudoresponse_end = process_time()
        # self.pseudoresponse_t += pseudoresponse_end - pseudoresponse_start
        return -2.0*(polar1 + polar2)

    def local_pseudoresponse(self, lpertbar, X1, X2):
        lpseudoresponse_start = process_time()
        contract = self.ccwfn.contract
        no = self.no
        Avo = lpertbar.Avo.copy()
        Avvoo = lpertbar.Avvoo.copy()
        polar1 = 0
        polar2 = 0
        for i in range(no):
            ii = i*no +i 
            polar1 += 2.0 * contract('a,a->', Avo[ii].copy(), X1[i].copy())
            for j in range(no):
                ij = i*no + j 
                #need to split this in two separate line slike X1 terms 
                polar2 += 2.0 * contract('ab,ab->', Avvoo[ij], (2.0*X2[ij] - X2[ij].transpose()))
        lpseudoresponse_end = process_time()
        self.lpseudoresponse_t += lpseudoresponse_end - lpseudoresponse_start
        return -2.0*(polar1 + polar2)
        
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

        #saving H.mu[axis] here for on the fly generation of pertbar in the in_Y1 eqns
        self.pert = pert
        self.Aov = []
        self.Avv = []
        self.Avo = []
        self.Aovoo = []
        lAvvoo = []
        self.Avvoo = []
        self.Avvvo = []

        self.Avvvj_ii = []

        self.Aoo = pert[o,o].copy()
        for i in range(no):
            ii = i*no + i
            for m in range(no):
                self.Aoo[m,i] += contract('e,e->',t1[i], (pert[m,v].copy() @ QL[ii]))

        norm = 0
        Sijmn = ccwfn.Local.Sijmn
        self.Aovoo_ji = []
        for ij in range(no*no):
            i = ij // no
            j = ij % no
            ii = i*no + i
            ji = j*no + i
            ijji = ij*(no*no) + ji

            #Aov
            self.Aov.append(pert[o,v].copy() @ QL[ij])
            # print((pert[o,v].copy() @ QL[ij]).shape)
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

            #Aovoo_ji
            Sijmn = ccwfn.Local.Sijmn

            tmp = contract('eb, me -> mb', t2[ji] @ Sijmn[ijji].T, pert[o,v].copy() @ QL[ji])
            self.Aovoo_ji.append(tmp)

            #Avvvo -> Avvvi
            tmp = 0
            for m in range(no):
                mi = m*no + i
                ijm = ij*no + m
                tmp -= contract('ab,e->abe', Sijmi[ijm] @ t2[mi] @ Sijmi[ijm].T, pert[m,v] @ QL[ij])
            self.Avvvo.append(tmp)

            #Avvv_{ii}j
            tmp = 0
            Sijmj = ccwfn.Local.Sijmj
            for m in range(no):
                mj = m*no + j
                ijm = ij*no + m
                tmp -= contract('ab,e->abe', Sijmj[ijm] @ t2[mj] @ Sijmj[ijm].T, pert[m,v] @ QL[ii])
            self.Avvvj_ii.append(tmp)


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
                norm += np.linalg.norm(0.5 * (lAvvoo[ij].copy() + lAvvoo[ji].copy().transpose()))
