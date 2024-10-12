"""
ccresponse.py: CC Response Functions
"""
import itertools

from opt_einsum import contract

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
        """
        LCX: Is the first second term contributions to the linear response symmetric function.
        <0|(1 + L^(0)){ [\bar{A}^(0), X^(1)(B)] + [\bar{B}^(1), X^(1)(B)]}|0>
        """
        LCX_start = process_time()
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

        polar1 += self.lLHX1Y1(X1_B, X1_A)
        polar1 += self.lLHX1Y1(X1_A, X1_B)
        # #
        # # # <0|L2[[H, X2], Y2]|0>
        # # temp = contract("ikac, ijab -> kjbc", L[o, o, v, v], X2_A)
        # # temp = contract("kjbc, klcd -> jlbd", temp, X2_B)
        # # LHX2Y2 = 2 * contract("jlbd, jlbd -> ", temp, l2)
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
        #
        polar1 += self.lLHX2Y2(X2_A, X2_B)
        polar1 += self.lLHX2Y2(X2_B, X2_A)
        # #
        polar1 += self.lLHX1Y2(X1_B, X2_A)
        polar1 += self.lLHX1Y2(X1_A, X2_B)
        # #
        polar1 += LHX1Y1 + LHX2Y2

        return -1.0 * polar1

    def lLCX(self, pertkey_a, X1_B, X2_B):
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

        return LCX

    def lLHX1Y1(self, X1_B, X1_A):
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

        return LHX1Y1

    def lLHX2Y2(self, X2_A, X2_B):
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

        return LHX2Y2

    def lLHX1Y2(self, X1_B, X2_A):
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
        # <0|[A_bar, X(B)]|0>
        # polar2 += 2.0 * contract("ia, ia -> ", pertbar_A.Aov, X1_B)
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
        return -1.0 * (polar1 + polar2)

    # def pert_quadresp(self, omega1, omega2, e_conv=1e-08, r_conv=1e-08, maxiter=200, max_diis=7, start_diis=1):
    #     """
    #     Build first-order perturbed wave functions (left- and right-hand) for the electric dipole operator (Mu)
    #
    #     Parameters
    #     ----------
    #     omega1: float
    #         First external field frequency.
    #     omega2: float
    #         Second external field frequency.
    #     e_conv : float
    #         convergence condition for the pseudoresponse value (default if 1e-13)
    #     r_conv : float
    #         convergence condition for perturbed wave function rmsd (default if 1e-13)
    #     maxiter : int
    #         maximum allowed number of iterations of the wave function equations (default is 100)
    #     max_diis : int
    #         maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
    #     start_diis : int
    #         earliest iteration to start DIIS extrapolations (default is 1)
    #
    #     To Do
    #     -----
    #     Organize to only compute the neccesary perturbed wave functions.
    #     """
    #
    #     #dictionaries for perturbed waves functions
    #     self.ccpert_om1_X = {}
    #     self.ccpert_om2_X = {}
    #     self.ccpert_om_sum_X = {}
    #
    #     self.ccpert_om1_2nd_X = {}
    #     self.ccpert_om2_2nd_X = {}
    #     self.ccpert_om_sum_2nd_X = {}
    #
    #     self.ccpert_om1_Y = {}
    #     self.ccpert_om2_Y = {}
    #     self.ccpert_om_sum_Y = {}
    #
    #     self.ccpert_om1_2nd_Y = {}
    #     self.ccpert_om2_2nd_Y = {}
    #     self.ccpert_om_sum_2nd_Y = {}
    #
    #     omega_sum = -(omega1 + omega2)
    #
    #     for axis in range(0, 3):
    #
    #         pertkey = "MU_" + self.cart[axis]
    #
    #         print("Solving right-hand perturbed wave function for omega1 %s:" % (pertkey))
    #         self.ccpert_om1_X[pertkey] = self.solve_right(self.pertbar[pertkey], omega1, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.ccpert_om1_Y[pertkey] = self.solve_left(self.pertbar[pertkey], omega1, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for omega2 %s:" % (pertkey))
    #         self.ccpert_om2_X[pertkey] = self.solve_right(self.pertbar[pertkey], omega2, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.ccpert_om2_Y[pertkey] = self.solve_left(self.pertbar[pertkey], omega2, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for omega_sum %s:" % (pertkey))
    #         self.ccpert_om_sum_X[pertkey] = self.solve_right(self.pertbar[pertkey], omega_sum, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.ccpert_om_sum_Y[pertkey] = self.solve_left(self.pertbar[pertkey], omega_sum, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for -omega1 %s:" % (pertkey))
    #         self.ccpert_om1_2nd_X[pertkey] = self.solve_right(self.pertbar[pertkey], -omega1, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.ccpert_om1_2nd_Y[pertkey] = self.solve_left(self.pertbar[pertkey], -omega1, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for -omega2 %s:" % (pertkey))
    #         self.ccpert_om2_2nd_X[pertkey] = self.solve_right(self.pertbar[pertkey], -omega2, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.ccpert_om2_2nd_Y[pertkey] = self.solve_left(self.pertbar[pertkey], -omega2, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for -omega_sum %s:" % (pertkey))
    #         self.ccpert_om_sum_2nd_X[pertkey] = self.solve_right(self.pertbar[pertkey], -omega_sum, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.ccpert_om_sum_2nd_Y[pertkey] = self.solve_left(self.pertbar[pertkey], -omega_sum, e_conv, r_conv, maxiter, max_diis, start_diis)
    #
    # def pert_lquadresp(self, omega1, omega2, e_conv=1e-12, r_conv=1e-12, maxiter=200, max_diis=7, start_diis=1):
    #     """
    #     Build first-order perturbed wave functions (left- and right-hand) for the electric dipole operator (Mu)
    #
    #     Parameters
    #     ----------
    #     omega1: float
    #         First external field frequency.
    #     omega2: float
    #         Second external field frequency.
    #     e_conv : float
    #         convergence condition for the pseudoresponse value (default if 1e-13)
    #     r_conv : float
    #         convergence condition for perturbed wave function rmsd (default if 1e-13)
    #     maxiter : int
    #         maximum allowed number of iterations of the wave function equations (default is 100)
    #     max_diis : int
    #         maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
    #     start_diis : int
    #         earliest iteration to start DIIS extrapolations (default is 1)
    #
    #     To Do
    #     -----
    #     Organize to only compute the neccesary perturbed wave functions.
    #     """
    #
    #     #dictionaries for perturbed waves functions
    #     self.lccpert_om1_X = {}
    #     self.lccpert_om2_X = {}
    #     self.lccpert_om_sum_X = {}
    #
    #     self.lccpert_om1_2nd_X = {}
    #     self.lccpert_om2_2nd_X = {}
    #     self.lccpert_om_sum_2nd_X = {}
    #
    #     self.lccpert_om1_Y = {}
    #     self.lccpert_om2_Y = {}
    #     self.lccpert_om_sum_Y = {}
    #
    #     self.lccpert_om1_2nd_Y = {}
    #     self.lccpert_om2_2nd_Y = {}
    #     self.lccpert_om_sum_2nd_Y = {}
    #
    #     omega_sum = -(omega1 + omega2)
    #
    #     for axis in range(0, 3):
    #
    #         pertkey = "MU_" + self.cart[axis]
    #
    #         print("Solving right-hand perturbed wave function for omega1 %s:" % (pertkey))
    #         self.lccpert_om1_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], omega1, e_conv, r_conv, maxiter) #, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.lccpert_om1_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], omega1, e_conv, r_conv, maxiter) #, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for omega2 %s:" % (pertkey))
    #         self.lccpert_om2_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], omega2, e_conv, r_conv, maxiter) #, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.lccpert_om2_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], omega2, e_conv, r_conv, maxiter) #, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for omega_sum %s:" % (pertkey))
    #         self.lccpert_om_sum_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], omega_sum, e_conv, r_conv, maxiter) #, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.lccpert_om_sum_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], omega_sum, e_conv, r_conv, maxiter) #, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for -omega1 %s:" % (pertkey))
    #         self.lccpert_om1_2nd_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], -omega1, e_conv, r_conv, maxiter) # , max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.lccpert_om1_2nd_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], -omega1, e_conv, r_conv, maxiter) #, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for -omega2 %s:" % (pertkey))
    #         self.lccpert_om2_2nd_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], -omega2, e_conv, r_conv, maxiter) # , max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.lccpert_om2_2nd_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], -omega2, e_conv, r_conv, maxiter) #, max_diis, start_diis)
    #
    #         print("Solving right-hand perturbed wave function for -omega_sum %s:" % (pertkey))
    #         self.lccpert_om_sum_2nd_X[pertkey] = self.local_solve_right(self.lpertbar[pertkey], -omega_sum, e_conv, r_conv, maxiter) #, max_diis, start_diis)
    #
    #         print("Solving left-hand perturbed wave function for %s:" % (pertkey))
    #         self.lccpert_om_sum_2nd_Y[pertkey] = self.local_solve_left(self.lpertbar[pertkey], -omega_sum, e_conv, r_conv, maxiter) # , max_diis, start_diis)
    #
    #
    # def lquadraticresp(self, pertkey_a, pertkey_b, pertkey_c, ccpert_X_A, ccpert_X_B, ccpert_X_C, ccpert_Y_A, ccpert_Y_B, ccpert_Y_C):
    #     contract = self.contract
    #     o = self.ccwfn.o
    #     v = self.ccwfn.v
    #     no = self.ccwfn.no
    #     t1 = self.lccwfn.t1
    #     t2 = self.lccwfn.t2
    #     l1 = self.cclambda.l1
    #     l2 = self.cclambda.l2
    #     # Grab X and Y amplitudes corresponding to perturbation A, omega_sum
    #     X1_A = ccpert_X_A[0]
    #     X2_A = ccpert_X_A[1]
    #     Y1_A = ccpert_Y_A[0]
    #     Y2_A = ccpert_Y_A[1]
    #     # Grab X and Y amplitudes corresponding to perturbation B, omega1
    #     X1_B = ccpert_X_B[0]
    #     X2_B = ccpert_X_B[1]
    #     Y1_B = ccpert_Y_B[0]
    #     Y2_B = ccpert_Y_B[1]
    #     # Grab X and Y amplitudes corresponding to perturbation C, omega2
    #     X1_C = ccpert_X_C[0]
    #     X2_C = ccpert_X_C[1]
    #     Y1_C = ccpert_Y_C[0]
    #     Y2_C = ccpert_Y_C[1]
    #     # Grab pert integrals
    #     pertbar_A = self.lpertbar[pertkey_a]
    #     pertbar_B = self.lpertbar[pertkey_b]
    #     pertbar_C = self.lpertbar[pertkey_c]
    #
    #     #Grab H_bar, L and ERI
    #     hbar = self.hbar
    #     L = self.H.L
    #     ERI = self.H. ERI
    #
    #     self.hyper = 0.0
    #     self.LAX = 0.0
    #     self.LAX2 = 0.0
    #     self.LAX3 = 0.0
    #     self.LAX4 = 0.0
    #     self.LAX5 = 0.0
    #     self.LAX6 = 0.0
    #     self.Fz1 = 0.0
    #     self.Fz2 = 0.0
    #     self.Fz3 = 0.0
    #
    #
    #     #LAX expressions
    #     # BAC
    #     self.LAX = self.comp_lLAX(X1_C, X2_C, Y1_B, Y2_B, pertbar_A)
    #     # CAB
    #     self.LAX2 = self.comp_lLAX(X1_B, X2_B, Y1_C, Y2_C, pertbar_A)
    #     # ABC
    #     self.LAX3 = self.comp_lLAX(X1_C, X2_C, Y1_A, Y2_A, pertbar_B)
    #     # CBA
    #     self.LAX4 = self.comp_lLAX(X1_A, X2_A, Y1_C, Y2_C, pertbar_B)
    #     # ACB
    #     self.LAX5 = self.comp_lLAX(X1_B, X2_B, Y1_A, Y2_A, pertbar_C)
    #     # BCA
    #     self.LAX6 = self.comp_lLAX(X1_A, X2_A, Y1_B, Y2_B, pertbar_C)
    #
    #     self.hyper += self.LAX + self.LAX2 + self.LAX3 + self.LAX4 + self.LAX5 + self.LAX6
    #
    #     #Fz expressions
    #     #L1, X1, X1
    #     self.Fz1 = self.comp_lFz(X1_B, X1_C, X2_B, X2_C, pertbar_A)
    #     #L2, X1, X2
    #     self.Fz2 = self.comp_lFz(X1_A, X1_C, X2_A, X2_C, pertbar_B)
    #     #L2, X2, X1
    #     self.Fz3 = self.comp_lFz(X1_A, X1_B, X2_A, X2_B, pertbar_C)
    #
    #     self.hyper += self.Fz1 + self.Fz2 + self.Fz3
    #
    #     #Generate Hbars for Bcon here
    #     #self.lHooov()
    #     #self.lHvovv()
    #
    #     #Bcon expressions
    #     #ABC
    #     self.Bcon1 = 0
    #     self.Bcon2 = 0
    #     self.Bcon3 = 0
    #
    #     #ABC
    #     self.Bcon1 = self.comp_lBcon(Y1_A, X1_B, X1_C, Y2_A, X2_B, X2_C, hbar)
    #     #BAC
    #     self.Bcon2 = self.comp_lBcon(Y1_B, X1_A, X1_C, Y2_B, X2_A, X2_C, hbar)
    #     #CAB
    #     self.Bcon3 = self.comp_lBcon(Y1_C, X1_A, X1_B, Y2_C, X2_A, X2_B, hbar)
    #
    #     self.hyper += self.Bcon1 + self.Bcon2 + self.Bcon3
    #
    #     self.G = 0
    #
    #     QL = self.Local.QL
    #     Sijmn = self.Local.Sijmn
    #     #<L1(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>
    #     for i in range(no):
    #         ii= i*no + i
    #
    #         for j in range(no):
    #             jj = j*no + j
    #             ij = i*no + j
    #             iijj = ii*(no*no) + jj
    #
    #             for k in range(no):
    #                 kk = k*no + k
    #                 ik = i*no + k
    #                 jk = j*no + k
    #                 jjkk = jj*(no*no) + kk
    #                 iikk = ii*(no*no) + kk
    #                 ijkk = ij*(no*no) + kk
    #                 ikjj = ik*(no*no) + jj
    #                 jkkk = jk*(no*no) + kk
    #
    #                 tmp = contract('a,ac->c', X1_A[i], QL[ii].T @ L[i,j,v,v] @ QL[kk])
    #                 tmp = contract('c,c->', X1_C[k], tmp)
    #                 tmp2 = contract('b,b->', X1_B[j], l1[k] @ Sijmn[jjkk].T)
    #                 self.G -= tmp2 * tmp
    #
    #                 tmp = contract('a,ab->b', X1_A[i], QL[ii].T @ L[i,k,v,v] @ QL[jj])
    #                 tmp = contract('b,b->', X1_B[j], tmp)
    #                 tmp2 = contract('c,c->', l1[j] @ Sijmn[jjkk], X1_C[k])
    #                 self.G -= tmp2 * tmp
    #
    #                 tmp = contract('b,ba->a', X1_B[j], QL[jj].T @ L[j,k,v,v] @ QL[ii])
    #                 tmp = contract('a,a->', X1_A[i], tmp)
    #                 tmp2 = contract('c,c->', X1_C[k], l1[i] @ Sijmn[iikk])
    #                 self.G -= tmp2 * tmp
    #
    #                 tmp = contract('b,bc->c', X1_B[j], QL[jj].T @ L[j,i,v,v] @ QL[kk])
    #                 tmp = contract('c,c->', X1_C[k], tmp)
    #                 tmp2 = contract('a,a->', l1[k] @ Sijmn[iikk].T , X1_A[i])
    #                 self.G -= tmp2 * tmp
    #
    #                 tmp = contract('c,cb->b', X1_C[k], QL[kk].T @ L[k,i,v,v] @ QL[jj])
    #                 tmp = contract('b,b->', X1_B[j], tmp)
    #                 tmp2 = contract('a,a->', l1[j] @ Sijmn[iijj].T , X1_A[i])
    #                 self.G -= tmp2 * tmp
    #
    #                 tmp = contract('c,ca->a', X1_C[k], QL[kk].T @ L[k,j,v,v] @ QL[ii])
    #                 tmp = contract('a,a->', X1_A[i], tmp)
    #                 tmp2 = contract('b,b->', X1_B[j], l1[i] @ Sijmn[iijj])
    #                 self.G -= tmp2 * tmp
    #
    #                 for l in range(no):
    #                     ll = l*no + l
    #                     il = i*no + l
    #                     ijll = ij*(no*no) + ll
    #                     ikll = ik*(no*no) + ll
    #                     iljj = il*(no*no) + jj
    #                     ilkk = il*(no*no) + kk
    #                     jkll = jk*(no*no) + ll
    #
    #                     # <L2(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>
    #                     Hooov = ERI[k,l,i,v] @ QL[jj]
    #                     Hooov = Hooov + contract('f,bf->b', t1[i], QL[jj].T @ ERI[l,k,v,v] @ QL[ii])
    #
    #                     tmp = contract('b,b->', X1_A[j], Hooov)
    #                     tmp2  = contract('d,cd->c', X1_C[l], Sijmn[ijkk].T @ l2[ij] @ Sijmn[ijll])
    #                     tmp2  = contract('c,c->', X1_B[k], tmp2)
    #                     self.G = self.G + (tmp2 * tmp)
    #
    #                     Hooov = ERI[l,k,i,v] @ QL[jj]
    #                     Hooov = Hooov + contract('f,bf->b', t1[i], QL[jj].T @ ERI[k,l,v,v] @ QL[ii])
    #
    #                     tmp = contract('b,b->', X1_A[j], Hooov)
    #                     tmp2 = contract('d,dc->c', X1_C[l], Sijmn[ijll].T @ l2[ij] @ Sijmn[ijkk])
    #                     tmp2 = contract('c,c->', X1_B[k], tmp2)
    #                     self.G = self.G + (tmp2 * tmp)
    #
    #                     Hooov = ERI[j,l,i,v] @ QL[kk]
    #                     Hooov = Hooov + contract('f,bf->b', t1[i], QL[kk].T @ ERI[l,j,v,v] @ QL[ii])
    #
    #                     tmp = contract('c,c->', X1_B[k], Hooov)
    #                     tmp2  = contract('b,bd->d', X1_A[j], Sijmn[ikjj].T @ l2[ik] @ Sijmn[ikll])
    #                     tmp2  = contract('d,d->', X1_C[l], tmp2)
    #                     self.G = self.G + (tmp2 * tmp)
    #
    #                     Hooov = ERI[l,j,i,v] @ QL[kk]
    #                     Hooov = Hooov + contract('f,bf->b', t1[i], QL[kk].T @ ERI[j,l,v,v] @ QL[ii])
    #
    #                     tmp = contract('c,c->', X1_B[k], Hooov)
    #                     tmp2  = contract('b,db->d', X1_A[j], Sijmn[ikll].T @ l2[ik] @ Sijmn[ikjj])
    #                     tmp2  = contract('d,d->', X1_C[l], tmp2)
    #                     self.G = self.G + (tmp2 * tmp)
    #
    #                     Hooov = ERI[j,k,i,v] @ QL[ll]
    #                     Hooov = Hooov + contract('f,bf->b', t1[i], QL[ll].T @ ERI[k,j,v,v] @ QL[ii])
    #
    #                     tmp = contract('d,d->', X1_C[l], Hooov)
    #                     tmp2  = contract('b,bc->c', X1_A[j], Sijmn[iljj].T @ l2[il] @ Sijmn[ilkk])
    #                     tmp2  = contract('c,c->', X1_B[k], tmp2)
    #                     self.G = self.G + (tmp2 * tmp)
    #
    #                     Hooov = ERI[k,j,i,v] @ QL[ll]
    #                     Hooov = Hooov + contract('f,bf->b', t1[i], QL[ll].T @ ERI[j,k,v,v] @ QL[ii])
    #
    #                     tmp = contract('d,d->', X1_C[l], Hooov)
    #                     tmp2  = contract('b,cb->c', X1_A[j], Sijmn[ilkk].T @ l2[il] @ Sijmn[iljj])
    #                     tmp2  = contract('c,c->', X1_B[k], tmp2)
    #                     self.G = self.G + (tmp2 * tmp)
    #
    #     for j in range(no):
    #         jj = j*no + j
    #
    #         for k in range(no):
    #             kk = k*no + k
    #             jk = j*no + k
    #
    #             for l in range(no):
    #                 ll = l*no + l
    #                 jl = j*no + l
    #                 kl = k*no + l
    #                 jkll = jk*(no*no) + ll
    #                 jlll = jl*(no*no) + ll
    #                 jlkk = jl*(no*no) + kk
    #                 jjkl = jj*(no*no) + kl
    #
    #                 Hvovv = contract('aA, abc -> Abc', QL[jk], ERI[v,l,v,v])
    #                 Hvovv = contract('bB, Abc -> ABc', QL[jj] , Hvovv)
    #                 Hvovv = contract('cC, ABc -> ABC', QL[kk], Hvovv)
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     jknn = jk*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc ->abc', (Sijmn[jknn] @ t1[n]), QL[jj].T @ ERI[n,l,v,v] @ QL[kk])
    #
    #                 tmp = contract('b,abc->ac', X1_A[j], Hvovv)
    #                 tmp = contract('c,ac->a', X1_B[k], tmp)
    #                 tmp2  = contract('d,ad->a', X1_C[l], l2[jk] @ Sijmn[jkll])
    #                 self.G = self.G - contract('a,a->', tmp2, tmp)
    #
    #                 #can I just do previous Hvovv.swapaxes(1,2)?
    #                 Hvovv = contract('aA, abc -> Abc', QL[jk], ERI[v,l,v,v])
    #                 Hvovv = contract('bB, Abc -> ABc', QL[kk] , Hvovv)
    #                 Hvovv = contract('cC, ABc -> ABC', QL[jj], Hvovv)
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     jknn = jk*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc ->abc', (Sijmn[jknn] @ t1[n]), QL[kk].T @ ERI[n,l,v,v] @ QL[jj])
    #
    #                 #Hvovv = Hvovv.swapaxes(1,2) - answer is no
    #                 tmp = contract('b,acb->ac', X1_A[j], Hvovv)
    #                 tmp = contract('c,ac->a', X1_B[k], tmp)
    #                 tmp2  = contract('d,da->a', X1_C[l], Sijmn[jkll].T @ l2[jk])
    #                 self.G = self.G - contract('a,a->', tmp2, tmp)
    #
    #                 Hvovv = contract('aA, abc -> Abc', QL[jl], ERI[v,k,v,v])
    #                 Hvovv = contract('bB, Abc -> ABc', QL[jj] , Hvovv)
    #                 Hvovv = contract('cC, ABc -> ABC', QL[ll], Hvovv)
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     jlnn = jl*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc ->abc', (Sijmn[jlnn] @ t1[n]), QL[jj].T @ ERI[n,k,v,v] @ QL[ll])
    #
    #                 tmp = contract('b,abd->ad', X1_A[j], Hvovv)
    #                 tmp = contract('d,ad->a', X1_C[l], tmp)
    #                 tmp2  = contract('c,ac->a', X1_B[k], l2[jl] @ Sijmn[jlkk])
    #                 self.G = self.G - contract('a,a->', tmp2, tmp)
    #
    #                 Hvovv = contract('aA, abc -> Abc', QL[jl], ERI[v,k,v,v])
    #                 Hvovv = contract('bB, Abc -> ABc', QL[ll] , Hvovv)
    #                 Hvovv = contract('cC, ABc -> ABC', QL[jj], Hvovv)
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     jlnn = jl*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc ->abc', (Sijmn[jlnn] @ t1[n]), QL[ll].T @ ERI[n,k,v,v] @ QL[jj])
    #
    #                 tmp = contract('b,adb->ad', X1_A[j], Hvovv)
    #                 tmp = contract('d,ad->a', X1_C[l], tmp)
    #                 tmp2  = contract('c,ca->a', X1_B[k], Sijmn[jlkk].T @ l2[jl])
    #                 self.G = self.G - contract('a,a->', tmp2, tmp)
    #
    #                 Hvovv = contract('aA, abc -> Abc', QL[kl], ERI[v,j,v,v])
    #                 Hvovv = contract('bB, Abc -> ABc', QL[kk] , Hvovv)
    #                 Hvovv = contract('cC, ABc -> ABC', QL[ll], Hvovv)
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     klnn = kl*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc ->abc', (Sijmn[klnn] @ t1[n]), QL[kk].T @ ERI[n,j,v,v] @ QL[ll])
    #
    #                 tmp = contract('c,acd->ad', X1_B[k], Hvovv)
    #                 tmp = contract('d,ad->a', X1_C[l], tmp)
    #                 tmp2  = contract('b,ab->a', X1_A[j], l2[kl] @ Sijmn[jjkl].T)
    #                 self.G = self.G - contract('a,a->', tmp2, tmp)
    #
    #                 Hvovv = contract('aA, abc -> Abc', QL[kl], ERI[v,j,v,v])
    #                 Hvovv = contract('bB, Abc -> ABc', QL[ll] , Hvovv)
    #                 Hvovv = contract('cC, ABc -> ABC', QL[kk], Hvovv)
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     klnn = kl*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc ->abc', (Sijmn[klnn] @ t1[n]), QL[ll].T @ ERI[n,j,v,v] @ QL[kk])
    #
    #                 tmp = contract('c,adc->ad', X1_B[k], Hvovv)
    #                 tmp = contract('d,ad->a', X1_C[l], tmp)
    #                 tmp2  = contract('b,ba->a', X1_A[j], Sijmn[jjkl] @ l2[kl])
    #                 self.G = self.G - contract('a,a->', tmp2, tmp)
    #
    #     #LHX2Y1Z1
    #     self.G += self.comp_lLHXYZ(X2_A, X1_B, X1_C)
    #     #LHX1Y2Z1
    #     self.G += self.comp_lLHXYZ(X2_B, X1_A, X1_C)
    #     #LHX1Y1Z2
    #     self.G += self.comp_lLHXYZ(X2_C, X1_A, X1_B)
    #
    #     self.hyper += self.G
    #     return self.hyper
    #
    # def comp_lLAX(self, X1_C, X2_C, Y1_B, Y2_B, pertbar_A):
    #     contract = self.contract
    #     no = self.ccwfn.no
    #
    #     LAX = 0
    #     Sijmn = self.Local.Sijmn
    #     for i in range(no):
    #         ii = i*no + i
    #
    #         # <0|L1(B)[A_bar, X1(C)]|0>
    #         tmp = contract('a,c->ac', Y1_B[i], X1_C[i])
    #         LAX += contract('ac,ac->',tmp, pertbar_A.Avv[ii].copy())
    #         for m in range(no):
    #             mm = m*no + m
    #             iimm = ii*(no*no) + mm
    #
    #             tmp = contract('a,a->', Y1_B[i], Sijmn[iimm] @ X1_C[m])
    #             LAX -= tmp * pertbar_A.Aoo[m,i]
    #
    #         for j in range(no):
    #             ij = i*no + j
    #             ijii = ij*(no*no) + ii
    #
    #             # <0|L1(B)[A_bar, X2(C)]|0>
    #             tmp = contract('a,b->ab', Sijmn[ijii] @ Y1_B[i], pertbar_A.Aov[ij][j])
    #             LAX += contract('ab,ab->', tmp, 2.0 * X2_C[ij] - X2_C[ij].swapaxes(0,1))
    #
    #         for j in range(no):
    #             ij = i*no + j
    #             ijii = ij*(no*no) + ii
    #             tmp = contract('bc,bca->a', Y2_B[ij], pertbar_A.Avvvj_ii[ij])
    #             LAX += contract('a,a->', tmp, X1_C[i])
    #
    #             for m in range(no):
    #                 mm = m*no + m
    #                 ijmm = ij*(no*no)+mm
    #
    #                 tmp = contract('ab,b->a', Y2_B[ij], pertbar_A.Aovoo[ij][m])
    #                 LAX -= contract('a,a->', tmp, X1_C[m] @ Sijmn[ijmm].T)
    #
    #         for j in range(no):
    #             ij = i*no + j
    #
    #             #Second Term
    #             tmp = contract('ab,ac->bc', Y2_B[ij], X2_C[ij])
    #             LAX += contract('bc,bc->', tmp, pertbar_A.Avv[ij])
    #             for m in range(no):
    #                 mj = m*no + j
    #                 ijmj = ij*(no*no) + mj
    #
    #                 tmp = contract('ab, ab->', Y2_B[ij], Sijmn[ijmj] @ X2_C[mj] @ Sijmn[ijmj].T)
    #                 LAX -= tmp * pertbar_A.Aoo[m,i]
    #
    #     return LAX
    #
    # def comp_LAX(self, X1_C, X2_C, Y1_B, Y2_B, pertbar_A):
    #     contract = self.contract
    #
    #     LAX = 0
    #     # <0|L1(B)[A_bar, X1(C)]|0>
    #     tmp = contract('ia,ic->ac', Y1_B, X1_C)
    #     LAX += contract('ac,ac->',tmp, pertbar_A.Avv)
    #     tmp = contract('ia,ka->ik', Y1_B, X1_C)
    #     LAX -= contract('ik,ki->', tmp, pertbar_A.Aoo)
    #
    #     # <0|L1(B)[A_bar, X2(C)]|0>
    #     tmp = contract('ia,jb->ijab', Y1_B, pertbar_A.Aov)
    #
    #     #swapaxes
    #     LAX += 2.0 * contract('ijab,ijab->', tmp, X2_C)
    #     LAX -= contract('ijab,ijba->',tmp, X2_C)
    #
    #     # <0|L2(B)[A_bar, X1(C)]|0>
    #     tmp = contract('ijbc,bcaj->ia', Y2_B, pertbar_A.Avvvo)
    #     LAX += contract('ia,ia->', tmp, X1_C)
    #     tmp = contract('ijab,kbij->ak', Y2_B, pertbar_A.Aovoo)
    #     LAX -= contract('ak,ka->', tmp, X1_C)
    #     ## <0|L2(B)[A_bar, X2(C)]|0>
    #     tmp = contract('ijab,kjab->ik', Y2_B, X2_C)
    #     LAX -= contract('ik,ki->', tmp, pertbar_A.Aoo)
    #     tmp = contract('ijab,ijac->bc', Y2_B, X2_C)
    #     LAX += contract('bc,bc->', tmp, pertbar_A.Avv)
    #
    #     return LAX
    #
    # def comp_lFz(self, X1_B, X1_C, X2_B, X2_C, pertbar_A):
    #     contract = self.contract
    #     no = self.ccwfn.no
    #     l1 = self.cclambda.l1
    #     l2 = self.cclambda.l2
    #
    #     Fz = 0
    #     Sijmn = self.Local.Sijmn
    #     for i in range(no):
    #         ii = i*no + i
    #         for j in range(no):
    #             ij = i*no + j
    #             jj = j*no + j
    #             iijj = ii*(no*no)+jj
    #
    #             # <0|L1(0)[[A_bar,X1(B)],X1(C)]|0>
    #             tmp2 =  contract('a,a->', X1_B[i], pertbar_A.Aov[ii][j])
    #             tmp1 = contract('b,b->', l1[i] @ Sijmn[iijj], X1_C[j])
    #             Fz -= tmp2 * tmp1
    #
    #             tmp = contract('b,b->', X1_C[j], pertbar_A.Aov[jj][i])
    #             tmp1 = contract('a,a->', X1_B[i], Sijmn[iijj] @ l1[j])
    #             Fz -= tmp * tmp1
    #
    #             # <0|L2(0)[[A_bar,X1(B)],X2(C)]|0>
    #             tmp1 = 0
    #             tmp = 0
    #
    #             for m in range(no):
    #                 jm = j*no + m
    #                 im = i*no + m
    #                 jmim = jm*(no*no) + im
    #                 jmii = jm*(no*no) + ii
    #
    #                 #why the need for 2.0
    #                 tmp1 = 2.0 * contract('bc,bc->', X2_C[jm], Sijmn[jmim] @ l2[im] @ Sijmn[jmim].T)
    #
    #                 #second term
    #                 tmp = contract('a, ac->c', X1_B[i], Sijmn[jmii].T @ l2[jm])
    #                 tmp = contract('bc, c->b', X2_C[jm], tmp)
    #                 Fz -= contract('b,b->', tmp, pertbar_A.Aov[jm][i])
    #
    #             #first term
    #             tmp2 = contract('a,a->', X1_B[i], pertbar_A.Aov[ii][j])
    #             Fz -= tmp2*tmp1
    #
    #             # <0|L2(0)[[A_bar,X2(B)],X1(C)]|0>
    #             tmp1 = 0
    #             tmp = 0
    #             tmp2 = contract('a,a->', X1_C[i], pertbar_A.Aov[ii][j])
    #             for m in range(no):
    #                 jm = j*no + m
    #                 im = i*no + m
    #                 jmii = jm*(no*no) + ii
    #                 jmim = jm*(no*no) + im
    #
    #                 #why the need for 2.0?
    #                 tmp1 = 2.0 * contract('bc,bc->', X2_B[jm], Sijmn[jmim] @ l2[im] @ Sijmn[jmim].T)
    #
    #                 #second term
    #                 tmp = contract('a, ac->c', X1_C[i], Sijmn[jmii].T @ l2[jm])
    #                 tmp = contract('bc, c->b', X2_B[jm], tmp)
    #                 Fz -= contract('b,b->', tmp, pertbar_A.Aov[jm][i])
    #
    #             #first term
    #             Fz -= tmp2*tmp1
    #     return Fz
    #
    # def comp_Fz(self, X1_B, X1_C, X2_B, X2_C, pertbar_A):
    #     contract = self.contract
    #     l1 = self.cclambda.l1
    #     l2 = self.cclambda.l2
    #
    #     Fz = 0
    #     # <0|L1(0)[[A_bar,X1(B)],X1(C)]|0>
    #     tmp = contract('ia,ja->ij', X1_B, pertbar_A.Aov)
    #     tmp2 = contract('ib,jb->ij', l1, X1_C)
    #     Fz -= contract('ij,ij->', tmp2, tmp)
    #
    #     tmp = contract('jb,ib->ij', X1_C, pertbar_A.Aov)
    #     tmp2 = contract('ia,ja->ij', X1_B, l1)
    #     Fz -= contract('ij,ij->', tmp2, tmp)
    #
    #     # <0|L2(0)[[A_bar,X1(B)],X2(C)]|0>
    #     tmp = contract('ia,ja->ij', X1_B, pertbar_A.Aov)
    #     tmp2 = contract('jkbc,ikbc->ij', X2_C, l2)
    #     Fz -= contract('ij,ij->',tmp2,tmp)
    #
    #     tmp = contract('ia,jkac->jkic', X1_B, l2)
    #     tmp = contract('jkbc,jkic->ib', X2_C, tmp)
    #     Fz -= contract('ib,ib->', tmp, pertbar_A.Aov)
    #
    #     # <0|L2(0)[[A_bar,X2(B)],X1(C)]|0>
    #     tmp = contract('ia,ja->ij', X1_C, pertbar_A.Aov)
    #     tmp2 = contract('jkbc,ikbc->ij', X2_B, l2)
    #     Fz -= contract('ij,ij->', tmp2, tmp)
    #
    #     tmp = contract('ia,jkac->jkic', X1_C, l2)
    #     tmp = contract('jkbc,jkic->ib', X2_B, tmp)
    #     Fz -= contract('ib,ib->', tmp, pertbar_A.Aov)
    #
    #     return Fz
    #
    # #hbar blocks for lBcon
    # def lHooov(self):
    #     contract = self.contract
    #     o = self.ccwfn.o
    #     v = self.ccwfn.v
    #     no = self.ccwfn.no
    #     t1 = self.lccwfn.t1
    #     ERIooov = self.Local.ERIooov
    #     ERI = self.H.ERI
    #     QL = self.Local.QL
    #
    #     self.Hkjib_jj = []
    #     self.Hjkib_jj = []
    #
    #     for i in range(no):
    #         ii = i*no + i
    #         for j in range(no):
    #             ij = i*no + j
    #             jj = j*no + j
    #             #convert this k loop to m loop
    #             for k in range(no):
    #                 kk = k*no + k
    #
    #                 Hkjib = ERIooov[jj][k,j,i].copy()
    #
    #                 tmp = contract('ef,eE ->Ef', ERI[j,k,v,v], QL[jj])
    #                 tmp = contract('Ef, fF -> EF', tmp, QL[ii])
    #                 Hkjib = Hkjib + contract('f,ef->e', t1[i], tmp)
    #
    #                 self.Hkjib_jj.append(Hkjib)
    #
    #                 Hjkib = ERIooov[jj][j,k,i].copy()
    #
    #                 tmp = contract('ef, eE- >Ef', ERI[k,j,v,v], QL[jj])
    #                 tmp = contract('Ef, fF -> EF', tmp, QL[ii])
    #                 Hjkib = Hjkib + contract('f,ef->e', t1[i], tmp)
    #
    #                 self.Hjkib_jj.append(Hjkib)
    #
    #     self.Hkjic_ii = []
    #     self.Hjkic_ii = []
    #
    #     for i in range(no):
    #         ii = i*no + i
    #         for j in range(no):
    #             ij = i*no + j
    #             jj = j*no + j
    #             #convert this k loop to m loop
    #             for k in range(no):
    #                 kk = k*no + k
    #
    #                 Hkjic = ERIooov[ii][k,j,i].copy()
    #
    #                 tmp = contract('ef,eE ->Ef', ERI[j,k,v,v], QL[ii])
    #                 tmp = contract('Ef, fF -> EF', tmp, QL[ii])
    #                 Hkjic = Hkjic + contract('f,ef->e', t1[i], tmp)
    #
    #                 self.Hkjic_ii.append(Hkjic)
    #
    #                 Hjkic = ERIooov[ii][j,k,i].copy()
    #
    #                 tmp = contract('ef, eE- >Ef', ERI[k,j,v,v], QL[ii])
    #                 tmp = contract('Ef, fF -> EF', tmp, QL[ii])
    #                 Hjkic = Hjkic + contract('f,ef->e', t1[i], tmp)
    #
    #                 self.Hjkic_ii.append(Hjkic)
    #
    # def lHvovv(self):
    #     contract = self.contract
    #     o = self.ccwfn.o
    #     v = self.ccwfn.v
    #     no = self.ccwfn.no
    #     t1 = self.lccwfn.t1
    #     ERI = self.H. ERI
    #     QL = self.Local.QL
    #     Sijmn = self.Local.Sijmn
    #
    #     self.Hajcb_kj = []
    #     self.Hajbc_kj = []
    #
    #     #Hv_kk j v_kk v_jj
    #     for i in range(no):
    #         ii = i*no + i
    #         for j in range(no):
    #             jj = j*no + j
    #
    #             #change k loop to m loop
    #             for k in range(no):
    #                 kk = k*no + k
    #
    #                 tmp = contract('afe, aA -> Afe', ERI[v,j,v,v], QL[kk])
    #                 tmp = contract('Afe, fF -> AFe', tmp, QL[kk])
    #                 Hajcb = contract('AFe, eE -> AFE', tmp, QL[jj])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     kknn = kk*(no*no) + nn
    #
    #                     tmp = t1[n] @ Sijmn[kknn].T
    #                     tmp1 = contract('fe,fF->Fe', ERI[n,i,v,v], QL[kk])
    #                     tmp1 = contract('Fe, eE -> FE', tmp1, QL[jj])
    #                     Hajcb = Hajcb - contract('a,fe->afe', tmp, tmp1)
    #                 self.Hajcb_kj.append(Hajcb)
    #
    #     #Hv_kk j v_jj v_kk
    #     for i in range(no):
    #         ii = i*no + i
    #         for j in range(no):
    #             jj = j*no + j
    #
    #             #change k loop to m loop
    #             for k in range(no):
    #                 kk = k*no + k
    #
    #                 tmp = contract('afe, aA -> Afe', ERI[v,j,v,v], QL[kk])
    #                 tmp = contract('Afe, fF -> AFe', tmp, QL[jj])
    #                 Hajbc = contract('AFe, eE -> AFE', tmp, QL[kk])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     kknn = kk*(no*no) + nn
    #
    #                     tmp = t1[n] @ Sijmn[kknn].T
    #                     tmp1 = contract('fe,fF->Fe', ERI[n,i,v,v], QL[jj])
    #                     tmp1 = contract('Fe, eE -> FE', tmp1, QL[kk])
    #                     Hajbc = Hajbc - contract('a,fe->afe', tmp, tmp1)
    #                 self.Hajbc_kj.append(Hajbc)
    #
    #     self.Hakbc_jk = []
    #     self.Hakcb_jk = []
    #
    #     #Hv_jj k v_jj v_kk
    #     for i in range(no):
    #         ii = i*no + i
    #         for j in range(no):
    #             jj = j*no + j
    #
    #             #change k loop to m loop
    #             for k in range(no):
    #                 kk = k*no + k
    #
    #                 tmp = contract('afe, aA -> Afe', ERI[v,k,v,v], QL[jj])
    #                 tmp = contract('Afe, fF -> AFe', tmp, QL[jj])
    #                 Hakbc = contract('AFe, eE -> AFE', tmp, QL[kk])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     jjnn = jj*(no*no) + nn
    #
    #                     tmp = t1[n] @ Sijmn[jjnn].T
    #                     tmp1 = contract('fe,fF->Fe', ERI[n,i,v,v], QL[jj])
    #                     tmp1 = contract('Fe, eE -> FE', tmp1, QL[kk])
    #                     Hakbc = Hakbc - contract('a,fe->afe', tmp, tmp1)
    #                 self.Hakbc_jk.append(Hakbc)
    #
    #     #Hv_jj k v_kk v_jj
    #     for i in range(no):
    #         ii = i*no + i
    #         for j in range(no):
    #             jj = j*no + j
    #
    #             #change k loop to m loop
    #             for k in range(no):
    #                 kk = k*no + k
    #
    #                 tmp = contract('afe, aA -> Afe', ERI[v,k,v,v], QL[jj])
    #                 tmp = contract('Afe, fF -> AFe', tmp, QL[kk])
    #                 Hakcb = contract('AFe, eE -> AFE', tmp, QL[jj])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     kknn = kk*(no*no) + nn
    #
    #                     tmp = t1[n] @ Sijmn[kknn].T
    #                     tmp1 = contract('fe,fF->Fe', ERI[n,i,v,v], QL[kk])
    #                     tmp1 = contract('Fe, eE -> FE', tmp1, QL[jj])
    #                     Hakcb = Hakcb - contract('a,fe->afe', tmp, tmp1)
    #                 self.Hakcb_jk.append(Hakcb)
    #
    # def comp_lBcon(self, Y1_A, X1_B, X1_C, Y2_A, X2_B, X2_C, hbar):
    #     contract = self.contract
    #     no = self.ccwfn.no
    #     Bcon = 0
    #     t1 = self.lccwfn.t1
    #     t2 = self.lccwfn.t2
    #     l1 = self.cclambda.l1
    #     Sijmn = self.Local.Sijmn
    #     QL = self.Local.QL
    #     ERIoovv = self.Local.ERIoovv
    #     ERI = self.H.ERI
    #     L = self.H.L
    #     v = self.ccwfn.v
    #
    #     # <O|L1(A)[[Hbar(0),X1(B),X1(C)]]|0>
    #     for j in range(no):
    #         jj = j*no + j
    #
    #         for k in range(no):
    #             kk = k*no + k
    #             jk = j*no + k
    #             jjkk = jj*(no*no) + kk
    #
    #             tmp = -1.0 * contract('c,b->cb', hbar.Hov[kk][j], Y1_A[k])
    #             tmp2 = contract('cb, b -> c', tmp, Sijmn[jjkk].T @ X1_B[j])
    #             Bcon = Bcon + contract('c,c->', tmp2, X1_C[k])
    #
    #             tmp = -1.0 * contract('c, b -> cb', Y1_A[j], hbar.Hov[jj][k])
    #             tmp2 = contract('cb, b ->c', tmp, X1_B[j])
    #             Bcon = Bcon + contract('c,c->', tmp2, Sijmn[jjkk] @ X1_C[k])
    #
    #             for i in range(no):
    #                 ii = i*no + i
    #                 iikk = ii*(no*no) + kk
    #                 iijj = ii*(no*no) + jj
    #
    #                 ##
    #                 Hooov = ERI[k,j,i,v].copy() @ QL[jj]
    #                 Hooov = Hooov + contract('f,bf->b', t1[i], QL[jj].T @ ERI[j,k,v,v] @ QL[ii])
    #
    #                 Hooov_12swap = ERI[j,k,i,v].copy() @ QL[jj]
    #                 Hooov_12swap = Hooov_12swap + contract('f,bf->b', t1[i], QL[jj].T @ ERI[k,j,v,v] @ QL[ii])
    #
    #                 tmp = contract('b,c->cb', -2.0 * Hooov + Hooov_12swap, Y1_A[i])
    #                 tmp2 = contract('cb,b ->c', tmp, X1_B[j])
    #                 Bcon = Bcon + contract('c,c->', tmp2, Sijmn[iikk] @ X1_C[k])
    #
    #                 ##
    #                 Hooov = ERI[j,k,i,v].copy() @ QL[kk]
    #                 Hooov = Hooov + contract('f,bf->b', t1[i], QL[kk].T @ ERI[k,j,v,v] @ QL[ii])
    #
    #                 Hooov_12swap = ERI[k,j,i,v].copy() @ QL[kk]
    #                 Hooov_12swap = Hooov_12swap + contract('f,bf->b', t1[i], QL[kk].T @ ERI[j,k,v,v] @ QL[ii])
    #
    #                 tmp = contract('c,b->cb', -2.0 * Hooov + Hooov_12swap, Y1_A[i])
    #                 tmp2 = contract('cb,b ->c', tmp, Sijmn[iijj] @ X1_B[j])
    #                 Bcon = Bcon + contract('c,c->', tmp2, X1_C[k])
    #
    #             #jk loop only
    #             Hvovv = contract('acb, aA -> Acb', ERI[v,j,v,v].copy(), QL[kk])
    #             Hvovv_34swap = contract('Abc, bB -> ABc', Hvovv.copy(), QL[jj])
    #             Hvovv_34swap = contract('ABc, cC -> ABC', Hvovv_34swap, QL[kk])
    #             Hvovv = contract('Acb, cC -> ACb', Hvovv, QL[kk])
    #             Hvovv = contract('ACb, bB -> ACB', Hvovv, QL[jj])
    #
    #             for i in range(no):
    #                 ii = i*no + i
    #                 iikk = ii*(no*no) + kk
    #
    #                 Hvovv = Hvovv - contract('a,cb->acb', Sijmn[iikk].T @ t1[i], QL[kk].T @ ERI[i,j,v,v].copy() @ QL[jj])
    #                 Hvovv_34swap = Hvovv_34swap - contract('a, bc->abc', Sijmn[iikk].T @ t1[i], QL[jj].T @ ERI[i,j,v,v] @ QL[kk])
    #
    #             tmp = contract('acb, a -> cb', 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2), Y1_A[k])
    #             tmp2 = contract('cb, b ->c', tmp, X1_B[j])
    #             Bcon = Bcon + contract('c,c->', tmp2, X1_C[k])
    #
    #             Hvovv = contract('abc, aA -> Abc', ERI[v,k,v,v].copy(), QL[jj])
    #             Hvovv_34swap = contract('Acb, cC -> ACb', Hvovv.copy(), QL[kk])
    #             Hvovv_34swap = contract('ACb, bB -> ACB', Hvovv_34swap, QL[jj])
    #             Hvovv = contract('Abc, bB -> ABc', Hvovv, QL[jj])
    #             Hvovv = contract('ABc, cC -> ABC', Hvovv, QL[kk])
    #
    #             for i in range(no):
    #                 ii = i*no + i
    #                 iijj = ii*(no*no) + jj
    #
    #                 Hvovv = Hvovv - contract('a,bc->abc', Sijmn[iijj].T @ t1[i], QL[jj].T @ ERI[i,k,v,v].copy() @ QL[kk])
    #                 Hvovv_34swap = Hvovv_34swap - contract('a, cb->acb', Sijmn[iijj].T @ t1[i], QL[kk].T @ ERI[i,k,v,v] @ QL[jj])
    #
    #             tmp = contract('abc, a -> cb', 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2), Y1_A[j])
    #             tmp2 = contract('cb, b ->c', tmp, X1_B[j])
    #             Bcon = Bcon + contract('c,c->', tmp2, X1_C[k])
    #
    #     # # <O|L2(A)|[[Hbar(0),X1(B)],X1(C)]|0>
    #     for j in range(no):
    #         jj = j* no + j
    #
    #         for n in range(no):
    #             nn = n*no + n
    #             nj = n*no + j
    #
    #             for k in range(no):
    #                 kk = k*no + k
    #                 nk = n*no + k
    #                 nj = n*no + j
    #                 kj = k*no + j
    #                 jk = j*no + k
    #                 jjnk = jj*(no*no) + nk
    #                 njkk = nj*(no*no) + kk
    #
    #
    #                 Hovov = QL[nk].T @ ERI[j,v,n,v].copy() @ QL[kk]
    #                 #Hovov = contract('ac, aA -> Ac', ERI[j,v,n,v], QL[nk])
    #                 #Hovov = contract('Ac, cC -> AC', Hovov, QL[kk])
    #                 ERIvovv = contract('acf, aA -> Acf', ERI[v,j,v,v], QL[nk])
    #                 ERIvovv = contract('Acf, cC -> ACf', ERIvovv, QL[kk])
    #                 ERIvovv = contract('ACf, fF -> ACF', ERIvovv, QL[nn])
    #
    #                 Hovov = Hovov + contract('f, acf -> ac', t1[n], ERIvovv)
    #                 for _o in range(no):
    #                     oo = _o*no + _o
    #                     _no  = n*no + _o
    #                     nkoo = nk*(no*no) + oo
    #                     nkno = nk*(no*no) + _no
    #
    #                     Hovov = Hovov - contract('a, c -> ac', Sijmn[nkoo] @ t1[_o], ERI[j,_o,n,v] @ QL[kk])
    #                     Hovov = Hovov - contract('fa, cf -> ac', t2[_no] @ Sijmn[nkno].T, QL[kk].T @ ERI[_o,j,v,v] @ QL[_no])
    #                     tmp = contract('a, cf -> acf', Sijmn[nkoo] @ t1[_o], QL[kk].T @ ERI[_o,j,v,v] @ QL[nn])
    #                     Hovov = Hovov - contract('f,acf -> ac', t1[n], tmp)
    #
    #                 #jnk loop only
    #                 tmp  = -1.0 * contract('ac,ba->cb', Hovov, Y2_A[nk])
    #                 tmp2 = contract('cb, c -> b', tmp, X1_B[k])
    #                 Bcon = Bcon + contract('b,b->', tmp2, Sijmn[jjnk].T @ X1_C[j])
    #
    #                 #tmp  -= contract('kanb,njca->jckb', hbar.Hovov, Y2_A)
    #                 Hovov = contract('ab, aA -> Ab', ERI[k,v,n,v], QL[nj])
    #                 Hovov = contract('Ab, bB -> AB', Hovov, QL[jj])
    #
    #                 ERIvovv = contract('abf, aA -> Abf', ERI[v,k,v,v], QL[nj])
    #                 ERIvovv = contract('Abf, bB -> ABf', ERIvovv, QL[jj])
    #                 ERIvovv = contract('ABf, fF -> ABF', ERIvovv, QL[nn])
    #
    #                 Hovov = Hovov + contract('f, abf -> ab', t1[n], ERIvovv)
    #                 for _o in range(no):
    #                     oo = _o*no + _o
    #                     _no  = n*no + _o
    #                     njoo = nj*(no*no) + oo
    #                     njno = nj*(no*no) + _no
    #
    #                     Hovov = Hovov - contract('a, b -> ab', Sijmn[njoo] @ t1[_o], ERI[k,_o,n,v] @ QL[jj])
    #                     Hovov = Hovov - contract('fa, bf -> ab', t2[_no] @ Sijmn[njno].T, QL[jj].T @ ERI[_o,k,v,v] @ QL[_no])
    #                     tmp = contract('a, bf -> abf', Sijmn[njoo] @ t1[_o], QL[jj].T @ ERI[_o,k,v,v] @ QL[nn])
    #                     Hovov = Hovov - contract('f,abf -> ab', t1[n], tmp)
    #
    #                 tmp = -1.0 * contract('ab, ca -> cb', Hovov, Y2_A[nj])
    #                 tmp2 = contract('cb, c -> b', tmp, Sijmn[njkk] @ X1_B[k])
    #                 Bcon = Bcon + contract('b,b->', tmp2, X1_C[j])
    #
    #                 # tmp  -= contract('jacn,nkab->jckb', hbar.Hovvo, Y2_A)
    #                 Hovvo = contract('ac, aA -> Ac', ERI[j,v,v,n], QL[nk])
    #                 Hovvo = contract('Ac, cC -> AC', Hovvo, QL[kk])
    #
    #                 ERIovvv = contract('acf, aA -> Acf', ERI[j,v,v,v], QL[nk])
    #                 ERIovvv = contract('Acf, cC -> ACf', ERIovvv, QL[kk])
    #                 ERIovvv = contract('ACf, fF -> ACF', ERIovvv, QL[nn])
    #
    #                 Hovvo = Hovvo + contract('f, acf -> ac', t1[n], ERIovvv)
    #                 for _o in range(no):
    #                     oo = _o*no + _o
    #                     _no  = n*no + _o
    #                     on = _o*no + n
    #                     nkoo = nk*(no*no) + oo
    #                     nkno = nk*(no*no) + _no
    #                     nkon = nk*(no*no) + on
    #
    #                     Hovvo = Hovvo - contract('a, c -> ac', Sijmn[nkoo] @ t1[_o], ERI[j,_o,v,n] @ QL[kk])
    #                     Hovvo = Hovvo - contract('fa, cf -> ac', t2[_no] @ Sijmn[nkno].T, QL[kk].T @ ERI[j,_o,v,v] @ QL[_no])
    #                     tmp = contract('a, cf -> acf', Sijmn[nkoo] @ t1[_o], QL[kk].T @ ERI[j,_o,v,v] @ QL[nn])
    #                     Hovvo = Hovvo - contract('f,acf -> ac', t1[n], tmp)
    #                     Hovvo = Hovvo + contract('fa, cf -> ac', t2[on] @ Sijmn[nkon].T, QL[kk].T @ L[j,_o,v,v] @ QL[_no])
    #
    #                 tmp = -1.0 * contract('ac, ab -> cb', Hovvo, Y2_A[nk])
    #                 tmp2 = contract('cb, c -> b', tmp, X1_B[k])
    #                 Bcon = Bcon + contract('b,b->', tmp2, Sijmn[jjnk].T @ X1_C[j])
    #
    #                 # tmp  -= contract('kabn,njac->jckb', hbar.Hovvo, Y2_A)
    #                 Hovvo = contract('ab, aA -> Ab', ERI[k,v,v,n], QL[nj])
    #                 Hovvo = contract('Ab, bB -> AB', Hovvo, QL[jj])
    #
    #                 ERIovvv = contract('abf, aA -> Abf', ERI[k,v,v,v], QL[nj])
    #                 ERIovvv = contract('Abf, bB -> ABf', ERIovvv, QL[jj])
    #                 ERIovvv = contract('ABf, fF -> ABF', ERIovvv, QL[nn])
    #
    #                 Hovvo = Hovvo + contract('f, abf -> ab', t1[n], ERIovvv)
    #                 for _o in range(no):
    #                     oo = _o*no + _o
    #                     _no  = n*no + _o
    #                     on = _o*no + n
    #                     njoo = nj*(no*no) + oo
    #                     njno = nj*(no*no) + _no
    #                     njon = nj*(no*no) + on
    #
    #                     Hovvo = Hovvo - contract('a, b -> ab', Sijmn[njoo] @ t1[_o], ERI[k,_o,v,n] @ QL[jj])
    #                     Hovvo = Hovvo - contract('fa, bf -> ab', t2[_no] @ Sijmn[njno].T, QL[jj].T @ ERI[k,_o,v,v] @ QL[_no])
    #                     tmp = contract('a, bf -> abf', Sijmn[njoo] @ t1[_o], QL[jj].T @ ERI[k,_o,v,v] @ QL[nn])
    #                     Hovvo = Hovvo - contract('f,abf -> ab', t1[n], tmp)
    #                     Hovvo = Hovvo + contract('fa, bf -> ab', t2[on] @ Sijmn[njon].T, QL[jj].T @ L[k,_o,v,v] @ QL[_no])
    #
    #                 tmp = -1.0 * contract('ab, ac -> cb', Hovvo, Y2_A[nj])
    #                 tmp2 = contract('cb, c -> b', tmp, Sijmn[njkk] @ X1_B[k])
    #                 Bcon = Bcon + contract('b,b->', tmp2, X1_C[j])
    #
    #                 for i in range(no):
    #                     ii = i*no + i
    #                     ni = n*no + i
    #                     _in = i*no + n
    #                     nijj = ni*(no*no) + jj
    #                     nikk = ni*(no*no) + kk
    #                     # tmp  += 0.5* contract('kjin,nibc->jckb', hbar.Hoooo, Y2_A)
    #
    #                     Hoooo = ERI[k,j,i,n].copy()
    #                     tmp_oooo = contract('e,e ->', t1[n], ERI[k,j,i,v] @ QL[nn])
    #                     tmp_oooo = tmp_oooo + contract('e,e ->', t1[i], ERI[j,k,n,v] @ QL[ii])
    #                     Hoooo = Hoooo + tmp_oooo
    #                     Hoooo = Hoooo + contract('ef, ef->', t2[_in], ERIoovv[_in][k,j])
    #                     tmp =  contract('f, ef -> e', t1[n], QL[ii].T @ ERI[k,j,v,v] @ QL[nn])
    #                     Hoooo = Hoooo + contract('e, e->', t1[i], tmp)
    #
    #                     tmp = 0.5 * Hoooo * Y2_A[ni].swapaxes(0,1)
    #                     tmp2 = contract('cb, c -> b', tmp, Sijmn[nikk] @ X1_B[k])
    #                     Bcon = Bcon + contract('b,b->', tmp2, Sijmn[nijj] @ X1_C[j])
    #
    #                     # tmp  += 0.5* contract('jkin,nicb->jckb', hbar.Hoooo, Y2_A)
    #
    #                     Hoooo = ERI[j,k,i,n]
    #                     tmp_oooo = contract('e,e ->', t1[n], ERI[j,k,i,v] @ QL[nn])
    #                     tmp_oooo = tmp_oooo + contract('e,e ->', t1[i], ERI[k,j,n,v] @ QL[ii])
    #                     Hoooo = Hoooo + tmp_oooo
    #                     Hoooo = Hoooo + contract('ef, ef->', t2[_in], ERIoovv[_in][j,k])
    #                     tmp =  contract('f, ef -> e', t1[n], QL[ii].T @ ERI[j,k,v,v] @ QL[nn])
    #                     Hoooo = Hoooo + contract('e, e->', t1[i], tmp)
    #
    #                     tmp = 0.5 * Hoooo * Y2_A[ni]
    #                     tmp2 = contract('cb, c -> b', tmp, Sijmn[nikk] @ X1_B[k])
    #                     Bcon = Bcon + contract('b,b->', tmp2, Sijmn[nijj] @ X1_C[j])
    #
    #     for j in range(no):
    #         jj = j*no + j
    #
    #         for k in range(no):
    #             kk = k*no + k
    #             jk = j*no + k
    #             kj = k*no + j
    #
    #             # tmp  += 0.5* contract('fabc,jkfa->jckb', hbar.Hvvvv, Y2_A)
    #             Hvvvv = contract('fabc, fF -> Fabc', ERI[v,v,v,v], QL[jk])
    #             Hvvvv = contract('Fabc, aA -> FAbc', Hvvvv, QL[jk])
    #             Hvvvv = contract('FAbc, bB -> FABc', Hvvvv, QL[jj])
    #             Hvvvv = contract('FABc, cC -> FABC', Hvvvv, QL[kk])
    #
    #             for n in range(no):
    #                 nn = n*no + n
    #                 jknn = jk*(no*no) + nn
    #
    #                 ERIvovv = contract('fbc, fF -> Fbc', ERI[v,n,v,v], QL[jk])
    #                 ERIvovv = contract('Fbc, bB -> FBc', ERIvovv, QL[jj])
    #                 ERIvovv = contract('FBc, cC -> FBC', ERIvovv, QL[kk])
    #
    #                 Hvvvv = Hvvvv - contract('a, fbc -> fabc', Sijmn[jknn] @ t1[n], ERIvovv)
    #
    #                 ERIvovv = contract('acb, aA -> Acb', ERI[v,n,v,v], QL[jk])
    #                 ERIvovv = contract('Acb, cC -> ACb', ERIvovv, QL[kk])
    #                 ERIvovv = contract('ACb, bB -> ACB', ERIvovv, QL[jj])
    #
    #                 tmp = contract('f, acb -> afcb', Sijmn[jknn] @ t1[n], ERIvovv)
    #                 Hvvvv = Hvvvv - tmp.swapaxes(0,1).swapaxes(2,3)
    #
    #                 for _o in range(no):
    #                     _no = n*no + _o
    #                     oo = _o*no + _o
    #                     jkno = jk*(no*no) + _no
    #                     jkoo = jk*(no*no) + oo
    #
    #                     Hvvvv = Hvvvv + contract('fa, bc -> fabc', Sijmn[jkno] @ t2[_no] @ Sijmn[jkno].T, QL[jj].T @ ERI[n,_o,v,v] @ QL[kk])
    #                     tmp = contract('a, bc-> abc', Sijmn[jkoo] @ t1[_o], QL[jj].T @ ERI[n,_o,v,v] @ QL[kk])
    #                     Hvvvv = Hvvvv + contract('f, abc -> fabc', Sijmn[jknn] @ t1[n], tmp)
    #
    #             tmp = 0.5 * contract('fabc, fa -> cb', Hvvvv, Y2_A[jk])
    #             tmp2 = contract('cb, c -> b', tmp, X1_B[k])
    #             Bcon = Bcon + contract('b,b->', tmp2, X1_C[j])
    #
    #             # tmp  += 0.5* contract('facb,kjfa->jckb', hbar.Hvvvv, Y2_A)
    #             Hvvvv = contract('facb, fF -> Facb', ERI[v,v,v,v], QL[kj])
    #             Hvvvv = contract('Facb, aA -> FAcb', Hvvvv, QL[kj])
    #             Hvvvv = contract('FAcb, cC -> FACb', Hvvvv, QL[kk])
    #             Hvvvv = contract('FACb, bB -> FACB', Hvvvv, QL[jj])
    #
    #             for n in range(no):
    #                 nn = n*no + n
    #                 kjnn = kj*(no*no) + nn
    #
    #                 ERIvovv = contract('fcb, fF -> Fcb', ERI[v,n,v,v], QL[kj])
    #                 ERIvovv = contract('Fcb, cC -> FCb', ERIvovv, QL[kk])
    #                 ERIvovv = contract('FCb, bB -> FCB', ERIvovv, QL[jj])
    #
    #                 Hvvvv = Hvvvv - contract('a, fcb -> facb', Sijmn[kjnn] @ t1[n], ERIvovv)
    #
    #                 ERIvovv = contract('abc, aA -> Abc', ERI[v,n,v,v], QL[kj])
    #                 ERIvovv = contract('Abc, bB -> ABc', ERIvovv, QL[jj])
    #                 ERIvovv = contract('ABc, cC -> ABC', ERIvovv, QL[kk])
    #
    #                 tmp = contract('f, abc -> afbc', Sijmn[kjnn] @ t1[n], ERIvovv)
    #                 Hvvvv = Hvvvv - tmp.swapaxes(0,1).swapaxes(2,3)
    #
    #                 for _o in range(no):
    #                     _no = n*no + _o
    #                     oo = _o*no + _o
    #                     kjno = kj*(no*no) + _no
    #                     kjoo = kj*(no*no) + oo
    #
    #                     Hvvvv = Hvvvv + contract('fa, cb -> facb', Sijmn[kjno] @ t2[_no] @ Sijmn[kjno].T, QL[kk].T @ ERI[n,_o,v,v] @ QL[jj])
    #                     tmp = contract('a, cb-> acb', Sijmn[kjoo] @ t1[_o], QL[kk].T @ ERI[n,_o,v,v] @ QL[jj])
    #                     Hvvvv = Hvvvv + contract('f, acb -> facb', Sijmn[kjnn] @ t1[n], tmp)
    #
    #             tmp = 0.5 * contract('facb, fa -> cb', Hvvvv, Y2_A[kj])
    #             tmp2 = contract('cb, c -> b', tmp, X1_B[k])
    #             Bcon = Bcon + contract('b,b->', tmp2, X1_C[j])
    #
    #     L = self.H.L
    #     ERIoovv = self.Local.ERIoovv
    #     Loovv = self.Local.Loovv
    #
    #     for i in range(no):
    #         ii = i*no + i
    #
    #         for j in range(no):
    #             jj = j*no + j
    #             ij = i*no + j
    #
    #             for k in range(no):
    #                 kk = k*no + k
    #                 jk = j*no + k
    #                 ik = i*no + k
    #                 jkkk = jk*(no*no) + kk
    #                 iikk = ii*(no*no) + kk
    #                 iijk = ii*(no*no) + jk
    #
    #                 for l in range(no):
    #                     ll = l*no + l
    #                     jl = j*no + l
    #                     lj = l*no + j
    #                     kl = k*no + l
    #                     ijll = ij*(no*no) + ll
    #                     ijjl = ij*(no*no) + jl
    #                     ijkk = ij*(no*no) + kk
    #                     ijkl = ij*(no*no) + kl
    #                     ijjk = ij*(no*no) + jk
    #                     ikkl = ik*(no*no) + kl
    #                     ijik = ij*(no*no) + ik
    #                     jlkl = jl*(no*no) + kl
    #                     ijlj = ij*(no*no) + lj
    #
    #
    #                     tmp = contract('ab,db->ad', t2[ij], Y2_A[ij])
    #                     tmp = contract('d,ad->a', Sijmn[ijll] @ X1_C[l], tmp)
    #                     tmp = contract('a,ca->c', tmp, QL[kk].T @ L[k,l,v,v] @ QL[ij])
    #                     Bcon = Bcon - contract('c,c->', tmp, X1_B[k])
    #
    #                     tmp = contract('ab,ba->', Sijmn[ijjl].T @ t2[ij] @ Sijmn[ijjl], Y2_A[jl])
    #                     tmp2 = contract('c,cd->d', X1_B[k], QL[kk].T @ L[k,i,v,v] @ QL[ll])
    #                     tmp2 = contract('d,d->', tmp2, X1_C[l])
    #                     Bcon = Bcon - (tmp2 * tmp)
    #
    #                     tmp = contract('ab,ba->', Sijmn[ijjk].T @ t2[ij] @ Sijmn[ijjk], Y2_A[jk])
    #                     tmp2 = contract('d,dc->c', X1_C[l], QL[ll].T @ L[l,i,v,v] @ QL[kk])
    #                     tmp2 = contract('c,c->', tmp2, X1_B[k])
    #                     Bcon = Bcon - (tmp2 *tmp)
    #
    #                     tmp = contract('ab,cb->ac', t2[ij], Y2_A[ij])
    #                     tmp = contract('c,ac->a', Sijmn[ijkk] @ X1_B[k], tmp)
    #                     tmp2 = contract('d,da->a', X1_C[l], QL[ll].T @ L[l,k,v,v] @ QL[ij])
    #                     Bcon = Bcon - contract('a,a->', tmp2, tmp)
    #
    #                     # <O|L2(A)[[Hbar(0),X2(B)],X2(C)]|0>
    #                     tmp = contract("cd,cd->", Sijmn[ijkl] @ X2_C[kl] @ Sijmn[ijkl].T, Y2_A[ij])
    #                     tmp = tmp * X2_B[ij]
    #                     Bcon = Bcon + 0.5* contract('ab,ab->', tmp, ERIoovv[ij][k,l].copy())
    #
    #                     tmp = contract("ab,bd->ad", X2_C[ij] @ Sijmn[ijik] , Y2_A[ik])
    #                     tmp = contract("ad,cd->ac", tmp, X2_B[kl] @ Sijmn[ikkl].T)
    #                     Bcon = Bcon + contract('ac,ac->', tmp, QL[ij].T @ ERI[j,l,v,v].copy() @ QL[kl])
    #
    #                     #c_kl b_ik, a_ij
    #                     tmp = contract("cd,db->cb", X2_C[kl] @ Sijmn[ikkl].T , Y2_A[ik])
    #                     tmp = contract("cb,ab->ca", tmp, X2_B[ij] @ Sijmn[ijik])
    #                     Bcon = Bcon + contract('ca,ac->', tmp, QL[ij].T @ ERI[l,j,v,v].copy() @ QL[kl])
    #
    #                     tmp = contract("ab,ab->", Sijmn[ijkl].T @ X2_B[ij] @ Sijmn[ijkl], Y2_A[kl])
    #                     tmp = tmp * X2_C[kl]
    #                     Bcon = Bcon + 0.5* contract('cd,cd->', tmp, ERIoovv[kl][i,j].copy())
    #
    #                     tmp = contract("ab,ac->bc", X2_B[ij] @ Sijmn[ijkl], QL[ij].T @ L[i,j,v,v] @ QL[kl])
    #                     tmp = contract("bc,cd->bd", tmp, X2_C[kl])
    #                     Bcon = Bcon - contract("bd,bd->", tmp, Y2_A[kl])
    #
    #                     tmp = contract("ab,ab->", X2_B[ij], Loovv[ij][i,k])
    #                     tmp = tmp * (Sijmn[jlkl] @ X2_C[kl] @ Sijmn[jlkl].T)
    #                     Bcon = Bcon - contract("cd,cd->", tmp, Y2_A[jl])
    #
    #                     tmp = contract("bc,cd->bd", QL[ij].T @ L[i,k,v,v] @ QL[kl], X2_C[kl] @ Sijmn[jlkl].T)
    #                     tmp = contract("bd,ab->ad", tmp, Sijmn[ijjl].T @ X2_B[ij])
    #                     Bcon = Bcon - contract("ad,ad->", tmp, Y2_A[jl])
    #
    #                     tmp = contract("ab,bc->ac", X2_B[ij] @ Sijmn[ijjl], Y2_A[jl])
    #                     tmp = contract("ac,cd->ad", tmp, Sijmn[jlkl] @ X2_C[kl])
    #                     Bcon = Bcon - contract("ad,ad->", tmp, QL[ij].T @ L[i,k,v,v] @ QL[kl])
    #
    #                     tmp = contract("ca,cd->ad", QL[kl].T @ L[k,l,v,v] @ QL[ij], X2_C[kl] @ Sijmn[ijkl].T)
    #                     tmp = contract("ad,db->ab", tmp, Y2_A[ij])
    #                     Bcon = Bcon - contract("ab,ab->", tmp, X2_B[ij])
    #
    #                     tmp = contract("cd,cd->",Loovv[kl][k,i], X2_C[kl])
    #                     tmp = (Sijmn[ijlj].T @ X2_B[ij] @ Sijmn[ijlj]) * tmp
    #                     Bcon = Bcon - contract("ab,ab->", tmp, Y2_A[lj])
    #
    #                     tmp = contract("cd,ac->da", Sijmn[ikkl] @ X2_C[kl], Y2_A[ik])
    #                     tmp = contract("da,bd->ab", tmp, QL[ij].T @ L[j,l,v,v] @ QL[kl])
    #                     Bcon = Bcon + (2.0* contract("ab,ab->", tmp, Sijmn[ijik].T @ X2_B[ij]))
    #
    #                 # ijk loop only
    #                 # <O|L1(A)[[Hbar(0),X1(B)],X2(C)]]|0>
    #                 tmp = contract('bc,c->b', 2.0 * (X2_C[jk] @ Sijmn[jkkk])  - (Sijmn[jkkk].T @ X2_C[jk]).swapaxes(0,1), Y1_A[k])
    #                 tmp = contract('ab,b->a', QL[ii].T @ L[i,j,v,v] @ QL[jk], tmp)
    #                 Bcon = Bcon + contract("a,a->", tmp, X1_B[i])
    #
    #                 tmp = contract("bc,ba->ca", X2_C[jk] @ Sijmn[iijk].T, QL[jk].T @ L[j,k,v,v] @ QL[ii])
    #                 tmp = contract("a,ca->c", X1_B[i], tmp)
    #                 Bcon = Bcon - contract("c,c->", tmp, Y1_A[i])
    #
    #                 tmp = contract("bc,bc->", X2_C[jk], Loovv[jk][j,i])
    #                 tmp = tmp * (Sijmn[iikk].T @ X1_B[i])
    #                 Bcon= Bcon - contract("a,a->", tmp, Y1_A[k])
    #
    #     # <O|L2(A)[[Hbar(0),X1(B)],X2(C)]]|0>
    #     for j in range(no):
    #         jj = j*no + j
    #
    #         for k in range(no):
    #             kk = k*no + k
    #             jk = j*no + k
    #
    #             for l in range(no):
    #                 ll = l*no + l
    #                 kl = k*no + l
    #                 lk = l*no + k
    #                 lj = l*no + j
    #                 jjlk = jj*(no*no) + lk
    #                 kllj = kl*(no*no) + lj
    #                 kllk = kl*(no*no) + lk
    #
    #                 tmp = contract('cd,db -> cb', X2_C[kl], Y2_A[lk])
    #                 tmp = contract('b, cb ->c', Sijmn[jjlk].T @ X1_B[j], tmp)
    #                 Bcon = Bcon - contract('c,c->', tmp, hbar.Hov[kl][j])
    #
    #                 tmp = contract('cd, dc ->', Sijmn[kllj].T @ X2_C[kl] @ Sijmn[kllj], Y2_A[lj])
    #                 tmp = tmp * X1_B[j]
    #                 Bcon = Bcon - contract('b,b ->', tmp, hbar.Hov[jj][k])
    #
    #                 #Hvovv
    #                 Hvovv = contract('acb, aA -> Acb', ERI[v,j,v,v], QL[lk])
    #                 Hvovv_34swap = contract('Abc, bB -> ABc', Hvovv, QL[jj])
    #                 Hvovv_34swap = contract('ABc, cC -> ABC', Hvovv_34swap, QL[kl])
    #                 Hvovv = contract('Acb, cC -> ACb', Hvovv, QL[kl])
    #                 Hvovv = contract('ACb, bB -> ACB', Hvovv, QL[jj])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     lknn = lk*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, cb ->acb', Sijmn[lknn] @ t1[n], QL[kl].T @ ERI[n,j,v,v] @ QL[jj])
    #                     Hvovv_34swap = Hvovv_34swap - contract('a, bc -> abc' , Sijmn[lknn] @ t1[n], QL[jj].T @ ERI[n,j,v,v] @ QL[kl])
    #
    #                 tmp = contract('da, cd -> ac', Y2_A[lk], X2_C[kl] @ Sijmn[kllk])
    #                 tmp2 = contract('b,acb -> ac', X1_B[j], 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2))
    #                 Bcon = Bcon + contract('ac,ac ->', tmp, tmp2)
    #
    #                 #Hvovv
    #                 Hvovv = contract('abc, aA -> Abc', ERI[v,k,v,v], QL[lj])
    #                 Hvovv_34swap = contract('Acb, cC -> ACb', Hvovv, QL[kl])
    #                 Hvovv_34swap = contract('ACb, bB -> ACB', Hvovv_34swap, QL[jj])
    #                 Hvovv = contract('Abc, bB -> ABc', Hvovv, QL[jj])
    #                 Hvovv = contract('ABc, cC -> ABC', Hvovv, QL[kl])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     ljnn = lj*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc ->abc', Sijmn[ljnn] @ t1[n], QL[jj].T @ ERI[n,k,v,v] @ QL[kl])
    #                     Hvovv_34swap = Hvovv_34swap - contract('a, bc -> abc' , Sijmn[ljnn] @ t1[n], QL[kl].T @ ERI[n,k,v,v] @ QL[jj])
    #
    #                 tmp = contract('b, da -> bda', X1_B[j], Y2_A[lj])
    #                 tmp2 = contract('cd, abc -> dab', X2_C[kl] @ Sijmn[kllj], 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2))
    #                 Bcon = Bcon + contract('bda, dab ->', tmp, tmp2)
    #
    #             for i in range(no):
    #                 ii = i*no + i
    #                 ji = j*no + i
    #                 ik = i*no + k
    #                 jkji = jk*(no*no) + ji
    #                 jkik = jk*(no*no) + ik
    #                 jkii = jk*(no*no) + ii
    #
    #                 Hvovv = contract('fba, fF -> Fba', ERI[v,k,v,v], QL[ji])
    #                 Hvovv = contract('Fba, bB -> FBa', Hvovv, QL[jk])
    #                 Hvovv = contract('FBa, aA -> FBA', Hvovv, QL[ii])
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     jinn = ji*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('f, ba -> fba', Sijmn[jinn] @ t1[n], QL[jk].T @ ERI[n,k,v,v] @ QL[ii])
    #
    #                 tmp = contract('a, fba -> fb', X1_B[i], Hvovv)
    #                 tmp = contract('fb, fc -> bc', tmp, Y2_A[ji])
    #                 Bcon = Bcon - contract('bc,bc ->', X2_C[jk] @ Sijmn[jkji], tmp)
    #
    #                 Hvovv = contract('fba, fF -> Fba', ERI[v,j,v,v], QL[ik])
    #                 Hvovv = contract('Fba, bB -> FBa', Hvovv, QL[ii])
    #                 Hvovv = contract('FBa, aA -> FBA', Hvovv, QL[jk])
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     iknn = ik*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('f, ba -> fba', Sijmn[iknn] @ t1[n], QL[ii].T @ ERI[n,j,v,v] @ QL[jk])
    #
    #                 tmp = contract('a, fac -> fc', X1_B[i], Hvovv)
    #                 tmp = contract('fc, fb -> bc', tmp, Y2_A[ik])
    #                 Bcon = Bcon - contract('bc, bc ->', Sijmn[jkik].T @ X2_C[jk], tmp)
    #
    #                 Hvovv = contract('fba, fF -> Fba', ERI[v,i,v,v], QL[jk])
    #                 Hvovv = contract('Fba, bB -> FBa', Hvovv, QL[jk])
    #                 Hvovv = contract('FBa, aA -> FBA', Hvovv, QL[jk])
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     jknn = jk*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('f, ba -> fba', Sijmn[jknn] @ t1[n], QL[jk].T @ ERI[n,i,v,v] @ QL[jk])
    #
    #                 tmp = contract('a, fa -> f', Sijmn[jkii] @ X1_B[i], Y2_A[jk])
    #                 tmp2 = contract('bc, fbc -> f', X2_C[jk], Hvovv)
    #                 Bcon = Bcon - contract('f,f->', tmp2, tmp)
    #
    #                 #Hooov = contract('b, bB -> B', ERI[k,j,i,v], QL[jj])
    #                 #Hooov_12swap = contract('b, bB -> B', ERI[j,k,i,v], QL[jj])
    #                 #Hooov = Hooov + contract('f, bf -> b', t1[i], QL[jj].T @ ERI[j,k,v,v] @ QL[ii])
    #                 #Hooov_12swap = Hooov_12swap + contract('f, bf -> b', t1[i], QL[jj].T @ ERI[k,j,v,v] @ QL[ii])
    #
    #                 #tmp = tmp + contract('a,e -> ae', -2.0 * Hooov + Hooov_12swap, l1[n] @ Sijmn[nnmm])
    #
    #                 #tmp = contract('b,b->', X1_B[j], -2.0 * Hooov + Hooov_12swap)
    #                 for l in range(no):
    #                     kl = k*no + l
    #                     il = i*no + l
    #                     klil = kl*(no*no) + il
    #                     jjil = jj*(no*no) + il
    #
    #                     #this is the same as the above
    #                     Hooov = contract('b, bB -> B', ERI[k,j,i,v], QL[jj])
    #                     Hooov_12swap = contract('b, bB -> B', ERI[j,k,i,v], QL[jj])
    #                     Hooov = Hooov + contract('f, bf -> b', t1[i], QL[jj].T @ ERI[j,k,v,v] @ QL[ii])
    #                     Hooov_12swap = Hooov_12swap + contract('f, bf -> b', t1[i], QL[jj].T @ ERI[k,j,v,v] @ QL[ii])
    #                     tmp = contract('b,b->', X1_B[j], -2.0 * Hooov + Hooov_12swap)
    #                     tmp2 = contract('cd, cd->', Sijmn[klil].T @ X2_C[kl] @ Sijmn[klil], Y2_A[il])
    #                     Bcon = Bcon + (tmp * tmp2)
    #
    #                     Hooov = contract('b, bB -> B', ERI[j,k,i,v], QL[kl])
    #                     Hooov_12swap = contract('b, bB -> B', ERI[k,j,i,v], QL[kl])
    #                     Hooov = Hooov + contract('f, bf -> b', t1[i], QL[kl].T @ ERI[k,j,v,v] @ QL[ii])
    #                     Hooov_12swap = Hooov_12swap + contract('f, bf -> b', t1[i], QL[kl].T @ ERI[j,k,v,v] @ QL[ii])
    #
    #                     tmp1 = contract('c, cd -> d', 2.0 * Hooov - Hooov_12swap, X2_C[kl] @ Sijmn[klil])
    #                     tmp1 = contract('d, b-> bd', tmp1, X1_B[j] @ Sijmn[jjil])
    #                     Bcon = Bcon - contract('bd, bd ->', tmp1, Y2_A[il])
    #
    #                 for n in range(no):
    #                     ni = n*no + i
    #                     nn = n*no + n
    #                     jkni = jk*(no*no) + ni
    #
    #                     Hooov = contract('b, bB -> B', ERI[j,k,n,v], QL[ii])
    #                     Hooov = Hooov + contract('f, bf -> b', t1[n], QL[ii].T @ ERI[k,j,v,v] @ QL[nn])
    #
    #                     tmp = contract('a, a->', X1_B[i], Hooov)
    #                     tmp2 = contract('bc, bc ->', Sijmn[jkni].T @ X2_C[jk] @ Sijmn[jkni], Y2_A[ni])
    #                     Bcon = Bcon +  (tmp * tmp2)
    #
    #     for i in range(no):
    #         ii = i*no + i
    #         for k in range(no):
    #             for n in range(no):
    #                 nn = n*no + n
    #                 nk = n*no + k
    #                 iink = ii*(no*no) + nk
    #
    #                 #tmp = contract('a, ab -> b', Sijmn[iink].T @ X1_B[i], Y2_A[nk])
    #                 for j in range(no):
    #                     jk = j*no + k
    #                     nkjk = nk*(no*no) + jk
    #
    #                     tmp = contract('a, ab -> b', Sijmn[iink].T @ X1_B[i], Y2_A[nk])
    #                     tmp = contract('bc, b -> c', Sijmn[nkjk] @ X2_C[jk], tmp)
    #                     Hooov = contract('c, cC -> C', ERI[i,j,n,v], QL[jk])
    #                     Hooov = Hooov + contract('f, cf -> c', t1[n], QL[jk].T @ ERI[j,i,v,v] @ QL[nn])
    #
    #                     Bcon = Bcon + contract('c, c ->', tmp, Hooov)
    #
    #                     tmp = contract('a, ba -> b', Sijmn[iink].T @ X1_B[i], Y2_A[nk])
    #                     tmp = contract('bc, b -> c', Sijmn[nkjk] @ X2_C[jk], tmp)
    #                     Hooov = contract('c, cC -> C', ERI[j,i,n,v], QL[jk])
    #                     Hooov = Hooov + contract('f, cf -> c', t1[n], QL[jk].T @ ERI[i,j,v,v] @ QL[nn])
    #
    #                     Bcon = Bcon + contract('c, c ->', tmp, Hooov)
    #
    #     for i in range(no):
    #         ii = i*no + i
    #
    #         for j in range(no):
    #             jj = j*no + j
    #             ij = i*no + j
    #
    #             for k in range(no):
    #                 kk = k*no + k
    #                 jk = j*no + k
    #                 jkkk = jk*(no*no) + kk
    #                 iijk = ii*(no*no) + jk
    #                 iikk = ii*(no*no) + kk
    #
    #                 #<O|L1(A)[[Hbar(0),X2(B)],X1(C)]]|0>
    #                 tmp = contract('bc,c->b', 2.0 * (X2_B[jk] @ Sijmn[jkkk])  - (Sijmn[jkkk].T @ X2_B[jk]).swapaxes(0,1), Y1_A[k])
    #                 tmp = contract('ab,b->a', QL[ii].T @ L[i,j,v,v] @ QL[jk], tmp)
    #                 Bcon = Bcon + contract("a,a->", tmp, X1_C[i])
    #
    #                 tmp = contract("bc,ba->ca", X2_B[jk] @ Sijmn[iijk].T, QL[jk].T @ L[j,k,v,v] @ QL[ii])
    #                 tmp = contract("a,ca->c", X1_C[i], tmp)
    #                 Bcon = Bcon - contract("c,c->", tmp, Y1_A[i])
    #
    #                 tmp = contract("bc,bc->", X2_B[jk], Loovv[jk][j,i])
    #                 tmp = tmp * (Sijmn[iikk].T @ X1_C[i])
    #                 Bcon= Bcon - contract("a,a->", tmp, Y1_A[k])
    #
    #     # <O|L2(A)[[Hbar(0),X2(B)],X1(C)]]|0>
    #     for k in range(no):
    #         for l in range(no):
    #                 kl = k*no + l
    #                 lk = l*no + k
    #                 kllk = kl*(no*no) + lk
    #
    #                 #tmp = contract('cd, db -> cb', X2_B[kl], Y2_A[lk])
    #                 for j in range(no):
    #                     jj = j*no + j
    #                     lj = l*no + j
    #                     jjlk = jj*(no*no) + lk
    #                     kllj = kl*(no*no) + lj
    #
    #                     tmp = contract('cd, db -> cb', X2_B[kl] @ Sijmn[kllk], Y2_A[lk])
    #                     tmp = contract('b, cb -> c', Sijmn[jjlk].T @ X1_C[j], tmp)
    #                     Bcon = Bcon - contract('c,c->', tmp, hbar.Hov[kl][j])
    #
    #                     tmp = contract('cd, dc ->', Sijmn[kllj].T @ X2_B[kl] @ Sijmn[kllj], Y2_A[lj])
    #                     tmp = tmp * X1_C[j]
    #                     Bcon = Bcon - contract('b, b ->', tmp, hbar.Hov[jj][k])
    #
    #                     Hvovv = contract('acb, aA -> Acb', ERI[v,j,v,v], QL[lk])
    #                     Hvovv_34swap = contract('Abc, bB -> ABc', Hvovv, QL[jj])
    #                     Hvovv_34swap = contract('ABc, cC -> ABC', Hvovv_34swap, QL[kl])
    #                     Hvovv = contract('Acb, cC -> ACb', Hvovv, QL[kl])
    #                     Hvovv = contract('ACb, bB -> ACB', Hvovv, QL[jj])
    #
    #                     for n in range(no):
    #                         nn = n*no + n
    #                         lknn = lk*(no*no) + nn
    #
    #                         Hvovv = Hvovv - contract('a, cb -> acb', Sijmn[lknn] @ t1[n], QL[kl].T @ ERI[n,j,v,v] @ QL[jj])
    #                         Hvovv_34swap = Hvovv_34swap - contract('a, bc -> abc', Sijmn[lknn] @ t1[n], QL[jj].T @ ERI[n,j,v,v] @ QL[kl])
    #
    #                     tmp = contract('da, cd -> ac', Y2_A[lk], X2_B[kl] @ Sijmn[kllk])
    #                     tmp2 = contract('b,acb -> ac', X1_C[j], 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2))
    #                     Bcon = Bcon + contract('ac, ac->', tmp, tmp2)
    #
    #                     Hvovv = contract('abc, aA -> Abc', ERI[v,k,v,v], QL[lj])
    #                     Hvovv_34swap = contract('Acb, cC -> ACb', Hvovv, QL[kl])
    #                     Hvovv_34swap = contract('ACb, bB -> ACB', Hvovv_34swap, QL[jj])
    #                     Hvovv = contract('Abc, bB -> ABc', Hvovv, QL[jj])
    #                     Hvovv = contract('ABc, cC -> ABC', Hvovv, QL[kl])
    #
    #                     for n in range(no):
    #                         nn = n*no + n
    #                         ljnn = lj*(no*no) + nn
    #
    #                         Hvovv = Hvovv - contract('a, bc -> abc', Sijmn[ljnn] @ t1[n], QL[jj].T @ ERI[n,k,v,v] @ QL[kl])
    #                         Hvovv_34swap = Hvovv_34swap - contract('a, cb -> acb', Sijmn[ljnn] @ t1[n], QL[kl].T @ ERI[n,k,v,v] @ QL[jj])
    #
    #                     tmp = contract('b, da -> bda', X1_C[j], Y2_A[lj])
    #                     tmp2 = contract('cd, abc -> dab', X2_B[kl] @ Sijmn[kllj], 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2))
    #                     Bcon = Bcon + contract('bda, dab ->', tmp, tmp2)
    #
    #     for i in range(no):
    #         ii = i*no + i
    #         for j in range(no):
    #             jj = j*no + j
    #             ji = j*no + i
    #
    #             for k in range(no):
    #                 kk = k*no + k
    #                 jk = j*no + k
    #                 ik = i*no + k
    #                 jijk = ji*(no*no) + jk
    #                 jkik = jk*(no*no) + ik
    #                 iijk = ii*(no*no) + jk
    #
    #                 Hvovv = contract('abc, aA -> Abc', ERI[v,k,v,v], QL[ji])
    #                 Hvovv = contract('Abc, bB -> ABc', Hvovv, QL[jk])
    #                 Hvovv = contract('ABc, cC -> ABC', Hvovv, QL[ii])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     jinn = ji*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc -> abc', Sijmn[jinn] @ t1[n], QL[jk].T @ ERI[n,k,v,v] @ QL[ii])
    #
    #                 tmp = contract('a, fba -> fb', X1_C[i], Hvovv)
    #                 tmp = contract('fb, fc -> bc', tmp, Y2_A[ji])
    #                 Bcon = Bcon - contract('bc, bc ->', X2_B[jk] @ Sijmn[jijk].T, tmp)
    #
    #                 Hvovv = contract('abc, aA -> Abc', ERI[v,j,v,v], QL[ik])
    #                 Hvovv = contract('Abc, bB -> ABc', Hvovv, QL[ii])
    #                 Hvovv = contract('ABc, cC -> ABC', Hvovv, QL[jk])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     iknn = ik*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc -> abc', Sijmn[iknn] @ t1[n], QL[ii].T @ ERI[n,j,v,v] @ QL[jk])
    #
    #                 tmp = contract('a, fac -> fc', X1_C[i], Hvovv)
    #                 tmp = contract('fc, fb -> bc', tmp, Y2_A[ik])
    #                 Bcon = Bcon - contract('bc, bc ->', Sijmn[jkik].T @ X2_B[jk], tmp)
    #
    #                 Hvovv = contract('abc, aA -> Abc', ERI[v,i,v,v], QL[jk])
    #                 Hvovv = contract('Abc, bB -> ABc', Hvovv, QL[jk])
    #                 Hvovv = contract('ABc, cC -> ABC', Hvovv, QL[jk])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     jknn = jk*(no*no) + nn
    #
    #                     Hvovv = Hvovv - contract('a, bc -> abc', Sijmn[jknn] @ t1[n], ERIoovv[jk][n,i])
    #
    #                 tmp = contract('a, fa -> f', X1_C[i] @ Sijmn[iijk], Y2_A[jk])
    #                 tmp2 = contract('bc, fbc -> f', X2_B[jk], Hvovv)
    #                 Bcon = Bcon - contract('f, f ->', tmp2, tmp)
    #
    #                 #Hooov = contract('b, bB -> B', ERI[k,j,i,v], QL[jj])
    #                 #Hooov_12swap = contract('b, bB -> B', ERI[j,k,i,v], QL[jj])
    #                 for l in range(no):
    #                     il = i*no + l
    #                     kl = k*no + l
    #                     klil = kl*(no*no) + il
    #                     jjil = jj*(no*no) + il
    #
    #                     #Hooov = contract('b, bB -> B', ERI[k,j,i,v], QL[jj])
    #                     Hooov = contract('b, bB -> B', ERI[k,j,i,v], QL[jj])
    #                     Hooov_12swap = contract('b, bB -> B', ERI[j,k,i,v], QL[jj])
    #                     Hooov = Hooov + contract('f, bf -> b', t1[i], QL[jj].T @ ERI[j,k,v,v] @ QL[ii])
    #                     Hooov_12swap = Hooov_12swap + contract('f, bf -> b', t1[i], QL[jj].T @ ERI[k,j,v,v] @ QL[ii])
    #
    #                     tmp = contract('b, b->', X1_C[j], -2.0 * Hooov + Hooov_12swap)
    #                     tmp2 = contract('cd, cd->', Sijmn[klil].T @ X2_B[kl] @ Sijmn[klil], Y2_A[il])
    #                     Bcon = Bcon + (tmp * tmp2)
    #
    #                     Hooov = contract('c, cC -> C', ERI[j,k,i,v], QL[kl])
    #                     Hooov_12swap = contract('c, cC -> C', ERI[k,j,i,v], QL[kl])
    #                     Hooov = Hooov + contract('f, cf -> c', t1[i], QL[kl].T @ ERI[k,j,v,v] @ QL[ii])
    #                     Hooov_12swap = Hooov_12swap + contract('f, bf -> b', t1[i], QL[kl].T @ ERI[j,k,v,v] @ QL[ii])
    #
    #                     tmp = contract('c, cd -> d', 2.0 * Hooov - Hooov_12swap, X2_B[kl] @ Sijmn[klil])
    #                     tmp = contract('d, b -> bd', tmp, X1_C[j] @ Sijmn[jjil])
    #                     Bcon = Bcon - contract('bd, bd ->', tmp, Y2_A[il])
    #
    #                 for n in range(no):
    #                     nn = n*no + n
    #                     ni = n*no + i
    #                     nk = n*no + k
    #                     jkni = jk*(no*no) + ni
    #                     iink = ii*(no*no) + nk
    #                     jknk = jk*(no*no) + nk
    #
    #                     Hooov = contract('a, aA -> A', ERI[j,k,n,v], QL[ii])
    #                     Hooov = Hooov + contract('f, af -> a', t1[n], QL[ii].T @ ERI[k,j,v,v] @ QL[nn])
    #                     tmp = contract('a, a ->', X1_C[i], Hooov)
    #                     tmp2 = contract('bc, bc ->', Sijmn[jkni].T @ X2_B[jk] @ Sijmn[jkni], Y2_A[ni])
    #                     Bcon = Bcon + (tmp2 * tmp)
    #
    #                     Hooov = contract('c, cC -> C', ERI[i,j,n,v], QL[jk])
    #                     Hooov = Hooov + contract('f, cf -> c', t1[n], QL[jk].T @ ERI[j,i,v,v] @ QL[nn])
    #                     tmp = contract('a, ab -> b', X1_C[i] @ Sijmn[iink], Y2_A[nk])
    #                     tmp = contract('bc, b -> c', Sijmn[jknk].T @ X2_B[jk], tmp)
    #                     Bcon = Bcon + contract('c, c ->', tmp, Hooov)
    #
    #                     Hooov = contract('c, cC -> C', ERI[j,i,n,v], QL[jk])
    #                     Hooov = Hooov + contract('f, cf -> c', t1[n], QL[jk].T @ ERI[i,j,v,v] @ QL[nn])
    #                     tmp = contract('a, ba -> b', X1_C[i] @ Sijmn[iink], Y2_A[nk])
    #                     tmp = contract('bc, b -> c', Sijmn[jknk].T @ X2_B[jk], tmp)
    #                     Bcon = Bcon + contract('c, c ->', tmp, Hooov)
    #
    #     return Bcon
    #
    # def comp_Bcon(self, Y1_A, X1_B, X1_C, Y2_A, X2_B, X2_C, hbar):
    #     contract = self.contract
    #     #Q = self.ccwfn.Local.Q
    #     #L = self.ccwfn.Local.L
    #     no = self.ccwfn.no
    #     t2 = self.ccwfn.t2
    #     o = self.ccwfn.o
    #     v = self.ccwfn.v
    #     L = self.H.L
    #     ERI = self.H.ERI
    #     Bcon = 0
    #
    #     #skipping hbar components
    #     #first_tmp = 0
    #     # <O|L1(A)[[Hbar(0),X1(B),X1(C)]]|0>
    #     tmp  = -1.0* contract('jc,kb->jkcb', hbar.Hov, Y1_A)
    #     tmp -= contract('jc,kb->jkcb', Y1_A, hbar.Hov)
    #
    #     #swapaxes
    #     tmp -= 2.0* contract('kjib,ic->jkcb', hbar.Hooov , Y1_A)
    #     tmp += contract('jkib,ic->jkcb', hbar.Hooov, Y1_A)
    #
    #     #swapaxes
    #     tmp -= 2.0* contract('jkic,ib->jkcb', hbar.Hooov, Y1_A)
    #     tmp += contract('kjic,ib->jkcb', hbar.Hooov, Y1_A)
    #
    #     # swapaxes
    #     tmp += 2.0* contract('ajcb,ka->jkcb', hbar.Hvovv, Y1_A)
    #     tmp -= contract('ajbc,ka->jkcb', hbar.Hvovv, Y1_A)
    #
    #     # swapaxes
    #     tmp += 2.0* contract('akbc,ja->jkcb', hbar.Hvovv, Y1_A)
    #     tmp -= contract('akcb,ja->jkcb', hbar.Hvovv, Y1_A)
    #
    #     tmp2 = contract('miae,me->ia', tmp, X1_B)
    #     Bcon += contract('ia,ia->', tmp2, X1_C)
    #
    #     # # <O|L2(A)|[[Hbar(0),X1(B)],X1(C)]|0>
    #     tmp   = -1.0* contract('janc,nkba->jckb', hbar.Hovov, Y2_A)
    #     tmp  -= contract('kanb,njca->jckb', hbar.Hovov, Y2_A)
    #     tmp  -= contract('jacn,nkab->jckb', hbar.Hovvo, Y2_A)
    #     tmp  -= contract('kabn,njac->jckb', hbar.Hovvo, Y2_A)
    #     tmp  += 0.5* contract('fabc,jkfa->jckb', hbar.Hvvvv, Y2_A)
    #     tmp  += 0.5* contract('facb,kjfa->jckb', hbar.Hvvvv, Y2_A)
    #     tmp  += 0.5* contract('kjin,nibc->jckb', hbar.Hoooo, Y2_A)
    #     tmp  += 0.5* contract('jkin,nicb->jckb', hbar.Hoooo, Y2_A)
    #     tmp2 = contract('iema,me->ia', tmp, X1_B)
    #     Bcon += contract('ia,ia->', tmp2, X1_C)
    #
    #     #double check work so far
    #     tmp = contract('ijab,ijdb->ad', t2, Y2_A)
    #     tmp = contract('ld,ad->la', X1_C, tmp)
    #     tmp = contract('la,klca->kc', tmp, L[o,o,v,v])
    #     Bcon -= contract('kc,kc->', tmp, X1_B)
    #
    #     tmp = contract('ijab,jlba->il', t2, Y2_A)
    #     tmp2 = contract('kc,kicd->id', X1_B, L[o,o,v,v])
    #     tmp2 = contract('id,ld->il', tmp2, X1_C)
    #     Bcon -= contract('il,il->', tmp2, tmp)
    #
    #     tmp = contract('ijab,jkba->ik', t2, Y2_A)
    #     tmp2 = contract('ld,lidc->ic', X1_C, L[o,o,v,v])
    #     tmp2 = contract('ic,kc->ik', tmp2, X1_B)
    #     Bcon -= contract('ik,ik->', tmp2, tmp)
    #
    #     tmp = contract('ijab,ijcb->ac', t2, Y2_A)
    #     tmp = contract('kc,ac->ka', X1_B, tmp)
    #     tmp2 = contract('ld,lkda->ka', X1_C, L[o,o,v,v])
    #     Bcon -= contract('ka,ka->', tmp2, tmp)
    #
    #     # <O|L2(A)[[Hbar(0),X2(B)],X2(C)]|0>
    #     tmp = contract("klcd,ijcd->ijkl", X2_C, Y2_A)
    #     tmp = contract("ijkl,ijab->klab", tmp, X2_B)
    #     Bcon += 0.5* contract('klab,klab->', tmp, ERI[o,o,v,v])
    #
    #     tmp = contract("ijab,ikbd->jkad", X2_B, Y2_A)
    #     tmp = contract("jkad,klcd->jlac", tmp, X2_C)
    #     Bcon += contract('jlac,jlac->',tmp, ERI[o,o,v,v])
    #
    #     tmp = contract("klcd,ikdb->licb", X2_C, Y2_A)
    #     tmp = contract("licb,ijab->ljca", tmp, X2_B)
    #     Bcon += contract('ljca,ljac->', tmp, ERI[o,o,v,v])
    #
    #     tmp = contract("ijab,klab->ijkl", X2_B, Y2_A)
    #     tmp = contract("ijkl,klcd->ijcd", tmp, X2_C)
    #     Bcon += 0.5* contract('ijcd,ijcd->',tmp, ERI[o,o,v,v])
    #
    #     tmp = contract("ijab,ijac->bc", X2_B, L[o,o,v,v])
    #     tmp = contract("bc,klcd->klbd", tmp, X2_C)
    #     Bcon -= contract("klbd,klbd->", tmp, Y2_A)
    #
    #     tmp = contract("ijab,ikab->jk", X2_B, L[o,o,v,v])
    #     tmp = contract("jk,klcd->jlcd", tmp, X2_C)
    #     Bcon -= contract("jlcd,jlcd->", tmp, Y2_A)
    #
    #     #close???
    #     tmp = contract("ikbc,klcd->ilbd", L[o,o,v,v], X2_C)
    #     tmp = contract("ilbd,ijab->jlad", tmp, X2_B)
    #     Bcon -= contract("jlad,jlad->", tmp, Y2_A)
    #
    #     tmp = contract("ijab,jlbc->ilac", X2_B, Y2_A)
    #     tmp = contract("ilac,klcd->ikad", tmp, X2_C)
    #     Bcon -= contract("ikad,ikad->", tmp, L[o,o,v,v])
    #
    #     tmp = contract("klca,klcd->ad", L[o,o,v,v], X2_C)
    #     tmp = contract("ad,ijdb->ijab", tmp, Y2_A)
    #     Bcon -= contract("ijab,ijab->", tmp, X2_B)
    #
    #     tmp = contract("kicd,klcd->il",L[o,o,v,v], X2_C)
    #     tmp = contract("ijab,il->ljab", X2_B, tmp)
    #     Bcon -= contract("ljab,ljab->", tmp, Y2_A)
    #
    #     tmp = contract("klcd,ikac->lida", X2_C, Y2_A)
    #     tmp = contract("lida,jlbd->ijab", tmp, L[o,o,v,v])
    #     Bcon += 2.0* contract("ijab,ijab->", tmp, X2_B)
    #
    #     # <O|L1(A)[[Hbar(0),X1(B)],X2(C)]]|0>
    #     tmp  = 2.0* contract("jkbc,kc->jb", X2_C, Y1_A)
    #     tmp -= contract("jkcb,kc->jb", X2_C, Y1_A)
    #     tmp = contract('ijab,jb->ia', L[o,o,v,v], tmp)
    #     Bcon += contract("ia,ia->", tmp, X1_B)
    #
    #     tmp = contract("jkbc,jkba->ca", X2_C, L[o,o,v,v])
    #     tmp = contract("ia,ca->ic", X1_B, tmp)
    #     Bcon -= contract("ic,ic->", tmp, Y1_A)
    #
    #     tmp = contract("jkbc,jibc->ki", X2_C, L[o,o,v,v])
    #     tmp = contract("ki,ia->ka", tmp, X1_B)
    #     Bcon -= contract("ka,ka->", tmp, Y1_A)
    #
    #     # <O|L2(A)[[Hbar(0),X1(B)],X2(C)]]|0>
    #     tmp = contract("klcd,lkdb->cb", X2_C, Y2_A)
    #     tmp = contract("jb,cb->jc", X1_B, tmp)
    #     Bcon -= contract("jc,jc->", tmp, hbar.Hov)
    #
    #     tmp = contract("klcd,ljdc->kj", X2_C, Y2_A)
    #     tmp = contract("kj,jb->kb", tmp, X1_B)
    #     Bcon -= contract("kb,kb->", tmp, hbar.Hov)
    #
    #     #Hvovv
    #     tmp = contract('lkda,klcd->ac', Y2_A, X2_C)
    #     tmp2 = contract('jb,ajcb->ac', X1_B, hbar.Hvovv)
    #     Bcon += 2.0* contract('ac,ac->', tmp, tmp2)
    #
    #     #Hvovv
    #     #swapaxes
    #     tmp = contract('lkda,klcd->ac', Y2_A, X2_C)
    #     tmp2 = contract('jb,ajbc->ac', X1_B, hbar.Hvovv)
    #     Bcon -= contract('ac,ac->', tmp, tmp2)
    #
    #     #Hvovv
    #     tmp = contract('jb,ljda->lbda', X1_B, Y2_A)
    #     tmp2 = 2.0* contract('klcd,akbc->ldab', X2_C, hbar.Hvovv)
    #     tmp2 -= contract('klcd,akcb->ldab', X2_C, hbar.Hvovv)
    #     Bcon += contract('lbda,ldab->', tmp, tmp2)
    #
    #     tmp = contract('ia,fkba->fkbi', X1_B, hbar.Hvovv)
    #     tmp = contract('fkbi,jifc->kjbc', tmp, Y2_A)
    #     Bcon -= contract('jkbc,kjbc->', X2_C, tmp)
    #
    #     tmp = contract('ia,fjac->fjic', X1_B, hbar.Hvovv)
    #     tmp = contract('fjic,ikfb->jkbc', tmp, Y2_A)
    #     Bcon -= contract('jkbc,jkbc->', X2_C, tmp)
    #
    #     ###double check
    #     tmp = contract('ia,jkfa->jkfi', X1_B, Y2_A)
    #     tmp2 = contract('jkbc,fibc->jkfi', X2_C, hbar.Hvovv)
    #     Bcon -= contract('jkfi,jkfi->', tmp2, tmp)
    #
    #     tmp = contract('jb,kjib->ki', X1_B, hbar.Hooov)
    #     tmp2 = contract('klcd,ilcd->ki', X2_C, Y2_A)
    #     Bcon -= 2.0*contract('ki,ki->', tmp, tmp2)
    #
    #     tmp = contract('jb,jkib->ki', X1_B, hbar.Hooov)
    #     tmp2 = contract('klcd,ilcd->ki', X2_C, Y2_A)
    #     Bcon += contract('ki,ki->', tmp, tmp2)
    #
    #     tmp  = 2.0* contract('jkic,klcd->jild', hbar.Hooov, X2_C)
    #     tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_C)
    #     tmp  = contract('jild,jb->bild', tmp, X1_B)
    #     Bcon -= contract('bild,ilbd->', tmp, Y2_A)
    #
    #     tmp  = contract('ia,jkna->jkni', X1_B, hbar.Hooov)
    #     tmp2  = contract('jkbc,nibc->jkni', X2_C, Y2_A)
    #     Bcon += contract('jkni,jkni->', tmp2, tmp)
    #
    #     tmp  = contract('ia,nkab->nkib', X1_B, Y2_A)
    #     tmp  = contract('jkbc,nkib->jnic', X2_C, tmp)
    #     Bcon += contract('jnic,ijnc->', tmp, hbar.Hooov)
    #
    #     tmp  = contract('ia,nkba->nkbi', X1_B, Y2_A)
    #     tmp  = contract('jkbc,nkbi->jnci', X2_C, tmp)
    #     Bcon += contract('jnci,jinc->', tmp, hbar.Hooov)
    #
    #     # <O|L1(A)[[Hbar(0),X2(B)],X1(C)]]|0>
    #     #swapaxes
    #     tmp  = 2.0* contract("jkbc,kc->jb", X2_B, Y1_A)
    #     tmp -= contract("jkcb,kc->jb", X2_B, Y1_A)
    #     tmp = contract('ijab,jb->ia', L[o,o,v,v], tmp)
    #     Bcon += contract("ia,ia->", tmp, X1_C)
    #
    #     tmp = contract("jkbc,jkba->ca", X2_B, L[o,o,v,v])
    #     tmp = contract("ia,ca->ic", X1_C, tmp)
    #     Bcon -= contract("ic,ic->", tmp, Y1_A)
    #
    #     tmp = contract("jkbc,jibc->ki", X2_B, L[o,o,v,v])
    #     tmp = contract("ki,ia->ka", tmp, X1_C)
    #     Bcon -= contract("ka,ka->", tmp, Y1_A)
    #
    #     # <O|L2(A)[[Hbar(0),X2(B)],X1(C)]]|0>
    #     tmp = contract("klcd,lkdb->cb", X2_B, Y2_A)
    #     tmp = contract("jb,cb->jc", X1_C, tmp)
    #     Bcon -= contract("jc,jc->", tmp, hbar.Hov)
    #
    #     tmp = contract("klcd,ljdc->kj", X2_B, Y2_A)
    #     tmp = contract("kj,jb->kb", tmp, X1_C)
    #     Bcon -= contract("kb,kb->", tmp, hbar.Hov)
    #
    #     #down
    #     tmp = contract('lkda,klcd->ac', Y2_A, X2_B)
    #     tmp2 = contract('jb,ajcb->ac', X1_C, hbar.Hvovv)
    #     Bcon += 2.0* contract('ac,ac->', tmp, tmp2)
    #
    #     tmp = contract('lkda,klcd->ac', Y2_A, X2_B)
    #     tmp2 = contract('jb,ajbc->ac', X1_C, hbar.Hvovv)
    #     Bcon -= contract('ac,ac->', tmp, tmp2)
    #
    #     tmp = contract('jb,ljda->lbda', X1_C, Y2_A)
    #
    #     #swapaxes
    #     tmp2 = 2.0* contract('klcd,akbc->ldab', X2_B, hbar.Hvovv)
    #     tmp2 -= contract('klcd,akcb->ldab', X2_B, hbar.Hvovv)
    #     Bcon += contract('lbda,ldab->', tmp, tmp2)
    #
    #     tmp = contract('ia,fkba->fkbi', X1_C, hbar.Hvovv)
    #     tmp = contract('fkbi,jifc->kjbc', tmp, Y2_A)
    #     Bcon -= contract('jkbc,kjbc->', X2_B, tmp)
    #
    #     tmp = contract('ia,fjac->fjic', X1_C, hbar.Hvovv)
    #     tmp = contract('fjic,ikfb->jkbc', tmp, Y2_A)
    #     Bcon -= contract('jkbc,jkbc->', X2_B, tmp)
    #
    #     tmp = contract('ia,jkfa->jkfi', X1_C, Y2_A)
    #     tmp2 = contract('jkbc,fibc->jkfi', X2_B, hbar.Hvovv)
    #     Bcon -= contract('jkfi,jkfi->', tmp2, tmp)
    #
    #     tmp = contract('jb,kjib->ki', X1_C, hbar.Hooov)
    #     tmp2 = contract('klcd,ilcd->ki', X2_B, Y2_A)
    #     Bcon -= 2.0* contract('ki,ki->', tmp, tmp2)
    #
    #     tmp = contract('jb,jkib->ki', X1_C, hbar.Hooov)
    #     tmp2 = contract('klcd,ilcd->ki', X2_B, Y2_A)
    #     Bcon += contract('ki,ki->', tmp, tmp2)
    #
    #     tmp  = 2.0* contract('jkic,klcd->jild', hbar.Hooov, X2_B)
    #     tmp -= contract('kjic,klcd->jild', hbar.Hooov, X2_B)
    #     tmp  = contract('jild,jb->bild', tmp, X1_C)
    #     Bcon -= contract('bild,ilbd->', tmp, Y2_A)
    #
    #     tmp  = contract('ia,jkna->jkni', X1_C, hbar.Hooov)
    #     tmp2  = contract('jkbc,nibc->jkni', X2_B, Y2_A)
    #     Bcon += contract('jkni,jkni->', tmp2, tmp)
    #
    #     tmp  = contract('ia,nkab->nkib', X1_C, Y2_A)
    #     tmp  = contract('jkbc,nkib->jnic', X2_B, tmp)
    #     Bcon += contract('jnic,ijnc->', tmp, hbar.Hooov)
    #
    #     tmp  = contract('ia,nkba->nkbi', X1_C, Y2_A)
    #     tmp  = contract('jkbc,nkbi->jnci', X2_B, tmp)
    #     Bcon += contract('jnci,jinc->', tmp, hbar.Hooov)
    #
    #     return Bcon
    #
    # def comp_lLHXYZ(self, X2_A, X1_B, X1_C):
    #     # <L2(0)|[[[H_bar,X2(A)],X1(B)],X1(C)]|0>
    #     # LHX2(A)Y1(B)Z1(C)
    #     v = self.ccwfn.v
    #     no = self.ccwfn.no
    #     ERI = self.H.ERI
    #     L = self.H.L
    #     l2 = self.cclambda.l2
    #     QL = self.Local.QL
    #     Sijmn = self.Local.Sijmn
    #     ERIoovv = self.Local.ERIoovv
    #
    #     G = 0
    #     for i in range(no):
    #         ii = i*no + i
    #
    #         for j in range(no):
    #              jj = j*no + j
    #              ij = i*no + j
    #              ji = j*no + i
    #
    #              for k in range(no):
    #                  kk = k*no + k
    #                  jk = j*no + k
    #                  ik = i*no + k
    #
    #                  for l in range(no):
    #                      ll = l*no + l
    #                      jl = j*no + l
    #                      il = i*no + l
    #                      kl = k*no + l
    #                      jlkk = jl*(no*no) + kk
    #                      ijjl = ij*(no*no) + jl
    #                      jkll = jk*(no*no) + ll
    #                      ijjk = ij*(no*no) + jk
    #                      ijji = ij*(no*no) + ji
    #                      jill = ji*(no*no) + ll
    #                      jikk = ji*(no*no) + kk
    #                      ijll = ij*(no*no) + ll
    #                      ijkk = ij*(no*no) + kk
    #                      ijik = ij*(no*no) + ik
    #                      ikll = ik*(no*no) + ll
    #                      ijil = ij*(no*no) + il
    #                      ilkk = il*(no*no) + kk
    #                      ijil = ij*(no*no) + il
    #                      ijkl = ij*(no*no) + kl
    #
    #                      tmp = contract('c,bc->b', X1_B[k], Sijmn[ijjl] @ l2[jl] @ Sijmn[jlkk])
    #                      tmp2 = contract('d,ad->a', X1_C[l], QL[ij].T @ L[i,k,v,v] @ QL[ll])
    #                      tmp2 = contract('ab,a->b', X2_A[ij], tmp2)
    #                      G -= contract('b,b->', tmp, tmp2)
    #
    #                      tmp = contract('d,bd->b', X1_C[l], Sijmn[ijjk] @ l2[jk] @ Sijmn[jkll])
    #                      tmp2 = contract('c,ac->a', X1_B[k], QL[ij].T @ L[i,l,v,v] @ QL[kk])
    #                      tmp2 = contract('ab,a->b', X2_A[ij], tmp2)
    #                      G -= contract('b,b->',tmp,tmp2)
    #
    #                      tmp = contract('ab,bd->ad', X2_A[ij], Sijmn[ijji] @ l2[ji] @ Sijmn[jill])
    #                      tmp = contract('d,ad->a', X1_C[l], tmp)
    #                      tmp2 = contract('ca,c->a', QL[kk].T @ L[k,l,v,v] @ QL[ij], X1_B[k])
    #                      G -= contract('a,a->', tmp, tmp2)
    #
    #                      tmp = contract('ab,ba->', X2_A[ij], Sijmn[ijjl] @ l2[jl] @ Sijmn[ijjl].T)
    #                      tmp2 = contract('c,cd->d', X1_B[k], QL[kk].T @ L[k,i,v,v] @ QL[ll])
    #                      tmp2 = contract('d,d->', X1_C[l], tmp2)
    #                      G -= tmp * tmp2
    #
    #                      tmp = contract('ab,ba->', X2_A[ij], Sijmn[ijjk] @ l2[jk] @ Sijmn[ijjk].T)
    #                      tmp2 = contract('d,dc->c', X1_C[l], QL[ll].T @ L[l,i,v,v] @ QL[kk])
    #                      tmp2 = contract('c,c->', X1_B[k], tmp2)
    #                      G -= tmp * tmp2
    #
    #                      tmp = contract('ab,bc->ac', X2_A[ij], Sijmn[ijji].T @ l2[ji] @ Sijmn[jikk])
    #                      tmp = contract('ac,c->a', tmp, X1_B[k])
    #                      tmp2 = contract('d,da->a', X1_C[l], QL[ll].T @ L[l,k,v,v] @ QL[ij])
    #                      G -= contract('a,a->', tmp, tmp2)
    #
    #                      tmp = contract('ab,ab->',X2_A[ij], ERIoovv[ij][k,l])
    #                      tmp2 = contract('c,cd->d', X1_B[k], Sijmn[ijkk].T @ l2[ij] @ Sijmn[ijll])
    #                      tmp2 = contract('d,d->', X1_C[l], tmp2)
    #                      G += tmp * tmp2
    #
    #                      tmp = contract('c,ac->a', X1_B[k], QL[ij].T @ ERI[j,l,v,v] @ QL[kk])
    #                      tmp = contract('ab,a->b', X2_A[ij], tmp)
    #                      tmp2 = contract('bd,d->b', Sijmn[ijik] @ l2[ik] @ Sijmn[ikll], X1_C[l])
    #                      G += contract('b,b->', tmp, tmp2)
    #
    #                      tmp = contract('c,ac->a', X1_B[k], QL[ij].T @ ERI[l,j,v,v] @ QL[kk])
    #                      tmp = contract('ab,a->b', X2_A[ij], tmp)
    #                      tmp2 = contract('db,d->b', Sijmn[ikll].T @ l2[ik] @ Sijmn[ijik].T, X1_C[l])
    #                      G += contract('b,b->', tmp, tmp2)
    #
    #                      tmp = contract('d,ad->a', X1_C[l], QL[ij].T @ ERI[j,k,v,v] @ QL[ll])
    #                      tmp = contract('ab,a->b', X2_A[ij], tmp)
    #                      tmp2 = contract('c,bc->b', X1_B[k], Sijmn[ijil] @ l2[il] @ Sijmn[ilkk])
    #                      G += contract('b,b->', tmp, tmp2)
    #
    #                      tmp = contract('d,ad->a', X1_C[l], QL[ij].T @ ERI[k,j,v,v] @ QL[ll])
    #                      tmp = contract('ab,a->b', X2_A[ij], tmp)
    #                      tmp2 = contract('c,cb->b', X1_B[k], Sijmn[ilkk].T @ l2[il] @ Sijmn[ijil].T)
    #                      G += contract('b,b->', tmp, tmp2)
    #
    #                      tmp = contract('c,cd->d', X1_B[k], QL[kk].T @ ERI[i,j,v,v] @ QL[ll])
    #                      tmp = contract('d,d->', X1_C[l], tmp)
    #                      tmp2 = contract('ab,ab->', X2_A[ij], Sijmn[ijkl] @ l2[kl] @ Sijmn[ijkl].T)
    #                      G += tmp * tmp2
    #
    #     return G
    #
    # def comp_LHXYZ(self, X2_A, X1_B, X1_C):
    #     L = self.H.L
    #     ERI = self.H. ERI
    #     l2 = self.cclambda.l2
    #     o = self.ccwfn.o
    #     v = self.ccwfn.v
    #
    #     G = 0
    #
    #     # <L2(0)|[[[H_bar,X2(A)],X1(B)],X1(C)]|0>
    #     tmp = contract('kc,jlbc->jlbk', X1_B, l2)
    #     tmp2 = contract('ld,ikad->ikal', X1_C, L[o,o,v,v])
    #     tmp2 = contract('ijab,ikal->jlbk', X2_A, tmp2)
    #     G -= contract('jlbk,jlbk->', tmp, tmp2)
    #
    #     tmp = contract('ld,jkbd->jkbl', X1_C, l2)
    #     tmp2 = contract('kc,ilac->ilak', X1_B, L[o,o,v,v])
    #     tmp2 = contract('ijab,ilak->jkbl', X2_A, tmp2)
    #     G -= contract('jkbl,jkbl->',tmp,tmp2)
    #
    #     tmp = contract('ijab,jibd->ad', X2_A, l2)
    #     tmp = contract('ld,ad->la', X1_C, tmp)
    #     tmp2 = contract('klca,kc->la', L[o,o,v,v], X1_B)
    #     G -= contract('la,la->', tmp, tmp2)
    #
    #     tmp = contract('ijab,jlba->il', X2_A, l2)
    #     tmp2 = contract('kc,kicd->id', X1_B, L[o,o,v,v])
    #     tmp2 = contract('ld,id->il', X1_C, tmp2)
    #     G -= contract('il,il->', tmp, tmp2)
    #
    #     tmp = contract('ijab,jkba->ik', X2_A, l2)
    #     tmp2 = contract('ld,lidc->ic', X1_C, L[o,o,v,v])
    #     tmp2 = contract('kc,ic->ik', X1_B, tmp2)
    #     G -= contract('ik,ik->', tmp, tmp2)
    #
    #     tmp = contract('ijab,jibc->ac', X2_A, l2)
    #     tmp = contract('ac,kc->ka', tmp, X1_B)
    #     tmp2 = contract('ld,lkda->ka', X1_C, L[o,o,v,v])
    #     G -= contract('ka,ka->', tmp, tmp2)
    #
    #     tmp = contract('ijab,klab->ijkl',X2_A, ERI[o,o,v,v])
    #     tmp2 = contract('kc,ijcd->ijkd', X1_B, l2)
    #     tmp2 = contract('ld,ijkd->ijkl', X1_C, tmp2)
    #     G += contract('ijkl,ijkl->',tmp,tmp2)
    #
    #     tmp = contract('kc,jlac->jlak', X1_B, ERI[o,o,v,v])
    #     tmp = contract('ijab,jlak->ilbk', X2_A, tmp)
    #     tmp2 = contract('ikbd,ld->ilbk', l2, X1_C)
    #     G += contract('ilbk,ilbk->', tmp, tmp2)
    #
    #     tmp = contract('kc,ljac->ljak', X1_B, ERI[o,o,v,v])
    #     tmp = contract('ijab,ljak->ilbk', X2_A, tmp)
    #     tmp2 = contract('ikdb,ld->ilbk', l2, X1_C)
    #     G += contract('ilbk,ilbk->', tmp, tmp2)
    #
    #     tmp = contract('ld,jkad->jkal', X1_C, ERI[o,o,v,v])
    #     tmp = contract('ijab,jkal->ikbl', X2_A, tmp)
    #     tmp2 = contract('kc,ilbc->ilbk', X1_B, l2)
    #     G += contract('ikbl,ilbk->', tmp, tmp2)
    #
    #     tmp = contract('ld,kjad->kjal', X1_C, ERI[o,o,v,v])
    #     tmp = contract('ijab,kjal->iklb', X2_A, tmp)
    #     tmp2 = contract('kc,ilcb->ilkb', X1_B, l2)
    #     G += contract('iklb,ilkb->', tmp, tmp2)
    #
    #     tmp = contract('kc,ijcd->ijkd', X1_B, ERI[o,o,v,v])
    #     tmp = contract('ld,ijkd->ijkl', X1_C, tmp)
    #     tmp2 = contract('ijab,klab->ijkl', X2_A, l2)
    #     G += contract('ijkl,ijkl->', tmp, tmp2)
    #
    #     return G
    #
    # def quadraticresp(self, pertkey_a, pertkey_b, pertkey_c, ccpert_X_A, ccpert_X_B, ccpert_X_C, ccpert_Y_A, ccpert_Y_B, ccpert_Y_C):
    #     """
    #     Calculate the CC quadratic-response function for one-electron perturbations A,B and C at field-frequency omega1(w1) and omega2(w2).
    #
    #     The quadratic response function, <<A;B,C>>w1, generally requires the following perturbed wave functions and frequencies:
    #         A(-w1-w2), A*(w1+w2), B(w1), B*(-w1), C(w2), C*(w2)
    #
    #     Parameters
    #     ----------
    #     pertkey_a: string
    #         String identifying the one-electron perturbation, A, along a cartesian axis
    #     pertkey_b: string
    #         String identifying the one-electron perturbation, B, along a cartesian axis
    #     pertkey_c: string
    #         String identifying the one-electron perturbation, C, along a cartesian axis
    #     ccpert_X_A:
    #         Perturbed right-hand wave functions for A along a cartesian axis
    #     ccpert_X_B:
    #
    #     Return
    #     ------
    #     hyper: float
    #         A value of the chosen quadratic response function corresponding to a specified cartesian direction. For example, Beta_xyz.
    #
    #     Notes
    #     -----
    #     Only the electric dipole is used for computing second harmonic generation (SHG) where w1 and w2 are identical and optical refractivity (OR)
    #     where w1 = -w2
    #
    #     To Do
    #     -----
    #     - Expand to include all avaiable one-electron perturbations
    #     """
    #     contract = self.contract
    #     o = self.ccwfn.o
    #     v = self.ccwfn.v
    #     t1 = self.ccwfn.t1
    #     t2 = self.ccwfn.t2
    #     l1 = self.cclambda.l1
    #     l2 = self.cclambda.l2
    #     # Grab X and Y amplitudes corresponding to perturbation A, omega_sum
    #     X1_A = ccpert_X_A[0]
    #     X2_A = ccpert_X_A[1]
    #     Y1_A = ccpert_Y_A[0]
    #     Y2_A = ccpert_Y_A[1]
    #     # Grab X and Y amplitudes corresponding to perturbation B, omega1
    #     X1_B = ccpert_X_B[0]
    #     X2_B = ccpert_X_B[1]
    #     Y1_B = ccpert_Y_B[0]
    #     Y2_B = ccpert_Y_B[1]
    #     # Grab X and Y amplitudes corresponding to perturbation C, omega2
    #     X1_C = ccpert_X_C[0]
    #     X2_C = ccpert_X_C[1]
    #     Y1_C = ccpert_Y_C[0]
    #     Y2_C = ccpert_Y_C[1]
    #     # Grab pert integrals
    #     pertbar_A = self.pertbar[pertkey_a]
    #     pertbar_B = self.pertbar[pertkey_b]
    #     pertbar_C = self.pertbar[pertkey_c]
    #
    #     #Grab H_bar, L and ERI
    #     hbar = self.hbar
    #     L = self.H.L
    #     ERI = self.H. ERI
    #
    #     self.hyper = 0.0
    #     self.LAX = 0.0
    #     self.LAX2 = 0.0
    #     self.LAX3 = 0.0
    #     self.LAX4 = 0.0
    #     self.LAX5 = 0.0
    #     self.LAX6 = 0.0
    #     self.LHX1Y1 = 0.0
    #     self.LHX1Y2 = 0.0
    #     self.LHX1X2 = 0.0
    #     self.LHX2Y2 = 0.0
    #
    #     #X1_A, X2_A = self.ccwfn.Local.filter_res(X1_A, X2_A)
    #     #X1_B, X2_B = self.ccwfn.Local.filter_res(X1_B, X2_B)
    #     #X1_C, X2_C = self.ccwfn.Local.filter_res(X1_C, X2_C)
    #     #Y1_A, Y2_A = self.ccwfn.Local.filter_res(Y1_A, Y2_A)
    #     #Y1_B, Y2_B = self.ccwfn.Local.filter_res(Y1_B, Y2_B)
    #     #Y1_C, Y2_C = self.ccwfn.Local.filter_res(Y1_C, Y2_C)
    #
    #     #LAX expressions
    #     # BAC
    #     self.LAX = self.comp_LAX(X1_C, X2_C, Y1_B, Y2_B, pertbar_A)
    #     # CAB
    #     self.LAX2 = self.comp_LAX(X1_B, X2_B, Y1_C, Y2_C, pertbar_A)
    #     # ABC
    #     self.LAX3 = self.comp_LAX(X1_C, X2_C, Y1_A, Y2_A, pertbar_B)
    #     # CBA
    #     self.LAX4 = self.comp_LAX(X1_A, X2_A, Y1_C, Y2_C, pertbar_B)
    #     # ACB
    #     self.LAX5 = self.comp_LAX(X1_B, X2_B, Y1_A, Y2_A, pertbar_C)
    #     # BCA
    #     self.LAX6 = self.comp_LAX(X1_A, X2_A, Y1_B, Y2_B, pertbar_C)
    #
    #     self.hyper += self.LAX + self.LAX2 + self.LAX3 + self.LAX4 + self.LAX5 + self.LAX6
    #
    #     self.Fz1 = 0
    #     self.Fz2 = 0
    #     self.Fz3 = 0
    #
    #     #Fz expressions
    #     #L1, X1, X1
    #     self.Fz1 = self.comp_Fz(X1_B, X1_C, X2_B, X2_C, pertbar_A)
    #     #L2, X1, X2
    #     self.Fz2 = self.comp_Fz(X1_A, X1_C, X2_A, X2_C, pertbar_B)
    #     #L2, X2, X1
    #     self.Fz3 = self.comp_Fz(X1_A, X1_B, X2_A, X2_B, pertbar_C)
    #
    #     self.hyper += self.Fz1 + self.Fz2 + self.Fz3
    #
    #     self.Bcon1 = 0
    #     self.Bcon2 = 0
    #     self.Bcon3 = 0
    #
    #     #skipping Bcon for now
    #     #Bcon expressions
    #     #ABC
    #     self.Bcon1 = self.comp_Bcon(Y1_A, X1_B, X1_C, Y2_A, X2_B, X2_C, hbar)
    #     #BAC
    #     self.Bcon2 = self.comp_Bcon(Y1_B, X1_A, X1_C, Y2_B, X2_A, X2_C, hbar)
    #     #CAB
    #     self.Bcon3 = self.comp_Bcon(Y1_C, X1_A, X1_B, Y2_C, X2_A, X2_B, hbar)
    #     self.hyper += self.Bcon1 + self.Bcon2 + self.Bcon3
    #
    #     self.G = 0
    #     # <L1(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>
    #     tmp = contract('ia,ijac->jc', X1_A, L[o,o,v,v])
    #     tmp = contract('kc,jc->jk', X1_C, tmp)
    #     tmp2 = contract('jb,kb->jk', X1_B, l1)
    #     self.G -= contract('jk,jk->', tmp2, tmp)
    #
    #     tmp = contract('ia,ikab->kb', X1_A, L[o,o,v,v])
    #     tmp = contract('jb,kb->jk', X1_B, tmp)
    #     tmp2 = contract('jc,kc->jk', l1, X1_C)
    #     self.G -= contract('jk,jk->', tmp2, tmp)
    #
    #     tmp = contract('jb,jkba->ka', X1_B, L[o,o,v,v])
    #     tmp = contract('ia,ka->ki', X1_A, tmp)
    #     tmp2 = contract('kc,ic->ki', X1_C, l1)
    #     self.G -= contract('ki,ki->', tmp2, tmp)
    #
    #     tmp = contract('jb,jibc->ic', X1_B, L[o,o,v,v])
    #     tmp = contract('kc,ic->ki', X1_C, tmp)
    #     tmp2 = contract('ka,ia->ki', l1, X1_A)
    #     self.G -= contract('ki,ki->', tmp2, tmp)
    #
    #     tmp = contract('kc,kicb->ib', X1_C, L[o,o,v,v])
    #     tmp = contract('jb,ib->ji', X1_B, tmp)
    #     tmp2 = contract('ja,ia->ji', l1, X1_A)
    #     self.G -= contract('ji,ji->', tmp2, tmp)
    #
    #     tmp = contract('kc,kjca->ja', X1_C, L[o,o,v,v])
    #     tmp = contract('ia,ja->ji', X1_A, tmp)
    #     tmp2 = contract('jb,ib->ji', X1_B, l1)
    #     self.G -= contract('ji,ji->', tmp2, tmp)
    #
    #     # <L2(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>
    #     tmp = contract('jb,klib->klij', X1_A, hbar.Hooov)
    #     tmp2  = contract('ld,ijcd->ijcl', X1_C, l2)
    #     tmp2  = contract('kc,ijcl->ijkl', X1_B, tmp2)
    #     self.G += contract('ijkl,klij->', tmp2, tmp)
    #
    #     tmp = contract('jb,lkib->lkij', X1_A, hbar.Hooov)
    #     tmp2 = contract('ld,ijdc->ijlc', X1_C, l2)
    #     tmp2 = contract('kc,ijlc->ijlk', X1_B, tmp2)
    #     self.G += contract('ijlk,lkij->', tmp2, tmp)
    #
    #     tmp = contract('kc,jlic->jlik', X1_B, hbar.Hooov)
    #     tmp2  = contract('jb,ikbd->ikjd', X1_A, l2)
    #     tmp2  = contract('ld,ikjd->ikjl', X1_C, tmp2)
    #     self.G += contract('ikjl,jlik->', tmp2, tmp)
    #
    #     tmp = contract('kc,ljic->ljik', X1_B, hbar.Hooov)
    #     tmp2  = contract('jb,ikdb->ikdj', X1_A, l2)
    #     tmp2  = contract('ld,ikdj->iklj', X1_C, tmp2)
    #     self.G += contract('iklj,ljik->', tmp2, tmp)
    #
    #     tmp = contract('ld,jkid->jkil', X1_C, hbar.Hooov)
    #     tmp2  = contract('jb,ilbc->iljc', X1_A, l2)
    #     tmp2  = contract('kc,iljc->iljk', X1_B, tmp2)
    #     self.G += contract('iljk,jkil->', tmp2, tmp)
    #
    #     tmp = contract('ld,kjid->kjil', X1_C, hbar.Hooov)
    #     tmp2  = contract('jb,ilcb->ilcj', X1_A, l2)
    #     tmp2  = contract('kc,ilcj->ilkj', X1_B, tmp2)
    #     self.G += contract('ilkj,kjil->', tmp2, tmp)
    #
    #     tmp = contract('jb,albc->aljc', X1_A, hbar.Hvovv)
    #     tmp = contract('kc,aljc->aljk', X1_B, tmp)
    #     tmp2  = contract('ld,jkad->jkal', X1_C, l2)
    #     self.G -= contract('jkal,aljk->', tmp2, tmp)
    #
    #     tmp = contract('jb,alcb->alcj', X1_A, hbar.Hvovv)
    #     tmp = contract('kc,alcj->alkj', X1_B, tmp)
    #     tmp2  = contract('ld,jkda->jkla', X1_C, l2)
    #     self.G -= contract('jkla,alkj->', tmp2, tmp)
    #
    #     tmp = contract('jb,akbd->akjd', X1_A, hbar.Hvovv)
    #     tmp = contract('ld,akjd->akjl', X1_C, tmp)
    #     tmp2  = contract('kc,jlac->jlak', X1_B, l2)
    #     self.G -= contract('jlak,akjl->', tmp2, tmp)
    #
    #     tmp = contract('jb,akdb->akdj', X1_A, hbar.Hvovv)
    #     tmp = contract('ld,akdj->aklj', X1_C, tmp)
    #     tmp2  = contract('kc,jlca->jlka', X1_B, l2)
    #     self.G -= contract('jlka,aklj->', tmp2, tmp)
    #
    #     tmp = contract('kc,ajcd->ajkd', X1_B, hbar.Hvovv)
    #     tmp = contract('ld,ajkd->ajkl', X1_C, tmp)
    #     tmp2  = contract('jb,klab->klaj', X1_A, l2)
    #     self.G -= contract('klaj,ajkl->', tmp2, tmp)
    #
    #     tmp = contract('kc,ajdc->ajdk', X1_B, hbar.Hvovv)
    #     tmp = contract('ld,ajdk->ajlk', X1_C, tmp)
    #     tmp2  = contract('jb,klba->klja', X1_A, l2)
    #     self.G -= contract('klja,ajlk->', tmp2, tmp)
    #
    #     #LHX2Y1Z1
    #     self.G += self.comp_LHXYZ(X2_A, X1_B, X1_C)
    #     #LHX1Y2Z1
    #     self.G += self.comp_LHXYZ(X2_B, X1_A, X1_C)
    #     #LHX1Y1Z2
    #     self.G += self.comp_LHXYZ(X2_C, X1_A, X1_B)
    #
    #     self.hyper += self.G
    #
    #     return self.hyper
    #
    # def hyperpolar(self):
    #     """
    #     Return
    #     ------
    #     Beta_avg: float
    #         Hyperpolarizability average
    #     """
    #     solver_start = time.time()
    #
    #     ccpert_om1_X = self.ccpert_om1_X
    #     ccpert_om2_X = self.ccpert_om2_X
    #     ccpert_om_sum_X = self.ccpert_om_sum_X
    #
    #     ccpert_om1_2nd_X = self.ccpert_om1_2nd_X
    #     ccpert_om2_2nd_X = self.ccpert_om2_2nd_X
    #     ccpert_om_sum_2nd_X = self.ccpert_om_sum_2nd_X
    #
    #     ccpert_om1_Y = self.ccpert_om1_Y
    #     ccpert_om2_Y = self.ccpert_om2_Y
    #     ccpert_om_sum_Y = self.ccpert_om_sum_Y
    #
    #     ccpert_om1_2nd_Y = self.ccpert_om1_2nd_Y
    #     ccpert_om2_2nd_Y = self.ccpert_om2_2nd_Y
    #     ccpert_om_sum_2nd_Y = self.ccpert_om_sum_2nd_Y
    #
    #     hyper_AB_1st = np.zeros((3,3,3))
    #     hyper_AB_2nd = np.zeros((3,3,3))
    #     hyper_AB = np.zeros((3,3,3))
    #
    #     for a in range(0, 3):
    #         pertkey_a = "MU_" + self.cart[a]
    #         for b in range(0, 3):
    #             pertkey_b = "MU_" + self.cart[b]
    #             for c in range(0, 3):
    #                 pertkey_c = "MU_" + self.cart[c]
    #
    #                 hyper_AB_1st[a,b,c] = self.quadraticresp(pertkey_a, pertkey_b, pertkey_c, ccpert_om_sum_X[pertkey_a], ccpert_om1_X[pertkey_b], ccpert_om2_X[pertkey_c],  ccpert_om_sum_Y[pertkey_a], ccpert_om1_Y[pertkey_b], ccpert_om2_Y[pertkey_c] )
    #                 hyper_AB_2nd[a,b,c] = self.quadraticresp(pertkey_a, pertkey_b, pertkey_c, ccpert_om_sum_2nd_X[pertkey_a], ccpert_om1_2nd_X[pertkey_b], ccpert_om2_2nd_X[pertkey_c],  ccpert_om_sum_2nd_Y[pertkey_a], ccpert_om1_2nd_Y[pertkey_b], ccpert_om2_2nd_Y[pertkey_c])
    #                 hyper_AB[a,b,c] = (hyper_AB_1st[a,b,c] + hyper_AB_2nd[a,b,c] )/2
    #
    #     Beta_avg = 0
    #     for i in range(0,3):
    #         Beta_avg += (hyper_AB[2,i,i] + hyper_AB[i,2,i] + hyper_AB[i,i,2])/5
    #
    #     print("\Beta_zxx = %10.12lf" %(hyper_AB[2,0,0]))
    #     print("\Beta_xzx = %10.12lf" %(hyper_AB[0,2,0]))
    #     print("\Beta_xxz = %10.12lf" %(hyper_AB[0,0,2]))
    #     print("\Beta_zyy = %10.12lf" %(hyper_AB[2,1,1]))
    #     print("\Beta_yzy = %10.12lf" %(hyper_AB[1,2,1]))
    #     print("\Beta_yyz = %10.12lf" %(hyper_AB[1,1,2]))
    #     print("\Beta_zzz = %10.12lf" %(hyper_AB[2,2,2]))
    #
    #     print("Beta_avg = %10.12lf" %(Beta_avg))
    #     print("\n First Dipole Hyperpolarizability computed in %.3f seconds.\n" % (time.time() - solver_start))
    #
    #     print(self.psuedoresponse)
    #     return Beta_avg
    #
    # def lhyperpolar(self):
    #     """
    #     Return
    #     ------
    #     Beta_avg: float
    #         Hyperpolarizability average
    #     """
    #     solver_start = time.time()
    #
    #     lccpert_om1_X = self.lccpert_om1_X
    #     lccpert_om2_X = self.lccpert_om2_X
    #     lccpert_om_sum_X = self.lccpert_om_sum_X
    #
    #     lccpert_om1_2nd_X = self.lccpert_om1_2nd_X
    #     lccpert_om2_2nd_X = self.lccpert_om2_2nd_X
    #     lccpert_om_sum_2nd_X = self.lccpert_om_sum_2nd_X
    #
    #     lccpert_om1_Y = self.lccpert_om1_Y
    #     lccpert_om2_Y = self.lccpert_om2_Y
    #     lccpert_om_sum_Y = self.lccpert_om_sum_Y
    #
    #     lccpert_om1_2nd_Y = self.lccpert_om1_2nd_Y
    #     lccpert_om2_2nd_Y = self.lccpert_om2_2nd_Y
    #     lccpert_om_sum_2nd_Y = self.lccpert_om_sum_2nd_Y
    #
    #     lhyper_AB_1st = np.zeros((3,3,3))
    #     lhyper_AB_2nd = np.zeros((3,3,3))
    #     lhyper_AB = np.zeros((3,3,3))
    #
    #     for a in range(0, 3):
    #         pertkey_a = "MU_" + self.cart[a]
    #         for b in range(0, 3):
    #             pertkey_b = "MU_" + self.cart[b]
    #             for c in range(0, 3):
    #                 pertkey_c = "MU_" + self.cart[c]
    #
    #                 lhyper_AB_1st[a,b,c] = self.lquadraticresp(pertkey_a, pertkey_b, pertkey_c, lccpert_om_sum_X[pertkey_a], lccpert_om1_X[pertkey_b], lccpert_om2_X[pertkey_c],  lccpert_om_sum_Y[pertkey_a], lccpert_om1_Y[pertkey_b], lccpert_om2_Y[pertkey_c] )
    #                 lhyper_AB_2nd[a,b,c] = self.lquadraticresp(pertkey_a, pertkey_b, pertkey_c, lccpert_om_sum_2nd_X[pertkey_a], lccpert_om1_2nd_X[pertkey_b], lccpert_om2_2nd_X[pertkey_c],  lccpert_om_sum_2nd_Y[pertkey_a], lccpert_om1_2nd_Y[pertkey_b], lccpert_om2_2nd_Y[pertkey_c])
    #                 lhyper_AB[a,b,c] = (lhyper_AB_1st[a,b,c] + lhyper_AB_2nd[a,b,c] )/2
    #
    #     Beta_avg = 0
    #     for i in range(0,3):
    #         Beta_avg += (lhyper_AB[2,i,i] + lhyper_AB[i,2,i] + lhyper_AB[i,i,2])/5
    #
    #     print("\Beta_zxx = %10.12lf" %(lhyper_AB[2,0,0]))
    #     print("\Beta_xzx = %10.12lf" %(lhyper_AB[0,2,0]))
    #     print("\Beta_xxz = %10.12lf" %(lhyper_AB[0,0,2]))
    #     print("\Beta_zyy = %10.12lf" %(lhyper_AB[2,1,1]))
    #     print("\Beta_yzy = %10.12lf" %(lhyper_AB[1,2,1]))
    #     print("\Beta_yyz = %10.12lf" %(lhyper_AB[1,1,2]))
    #     print("\Beta_zzz = %10.12lf" %(lhyper_AB[2,2,2]))
    #
    #     print("Beta_avg = %10.12lf" %(Beta_avg))
    #     print("\n First Dipole Hyperpolarizability computed in %.3f seconds.\n" % (time.time() - solver_start))
    #
    #     print(self.psuedoresponse)
    #     return Beta_avg

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

        pseudo = pseudo = self.pseudoresponse(pertbar, X1, X2)
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

        return r_X1

    def lr_X1(self, lpertbar, omega):
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

        return r_X2

    def lr_X2(self, lpertbar, conv_hbar, omega):
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
        
        return lr2    

    def in_Y1(self, pertbar, X1, X2):
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

        return r_Y1

    def in_lY1(self, lpertbar, X1, X2):
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
        mu = lpertbar.pert #z axis only for now       
        ERIoovv = self.Local.ERIoovv

        in_Y1 = []
        for i in range(no): 
            ii = i * no + i 

            # <O|A_bar|phi^a_i> good
            r_Y1 = 2.0 * lpertbar.Aov[ii][i].copy()

            #collecting Gvv terms here
            for m in range(no): 
                for n in range(no):
                    nn = n*no + n
                    mn = m*no + n 
                    iimn = ii*(no*no) + mn
       
                    #read the resulting index more carefully in Gvv its ae but need ea 
                    Gvv = -1.0 * contract('ab,eb -> ea', QL[ii].T @ L[m,n,v,v] @ QL[mn], X2[mn])
            
                    r_Y1 = r_Y1 + contract('e, ea ->a', Sijmn[iimn].T @ l1[i], Gvv) 


            for n in range(no): 
                nn = n*no + n
                for m in range(no): 
                    for _o in range(no):
                        mo = m*no + _o
                        nnmo = nn*(no*no) + mo
  
                        Gvv = -1.0 * contract('bc,fc -> fb', Sijmn[nnmo] @ l2[mo], t2[mo]) 
                        tmp = contract('b, fb ->f', X1[n], Gvv)  
                        r_Y1 = r_Y1 + contract('af, f -> a', QL[ii].T @ L[i,n,v,v] @ QL[mo], tmp) 

            for m in range(no): 
                mm = m*no + m 
                for n in range(no):
                    for _o in range(no):
                        _no = n*no + _o 
                        iino = ii*(no*no) + _no                      

                        Gvv = -1.0 * contract('ac,fc -> fa', Sijmn[iino] @ l2[_no], t2[_no]) 
                        tmp = contract('e, fa -> efa', X1[m], Gvv) 
                        r_Y1 = r_Y1 + contract('ef, efa -> a', QL[mm].T @ L[m,i,v,v] @ QL[_no], tmp)             

            for m in range(no):
                for n in range(no):
                    mn = m*no + n
                    iimn = ii*(no*no) + mn

                    Gvv = -1.0 * contract('ab, eb -> ea', Sijmn[iimn] @ l2[mn], X2[mn])
                    r_Y1 = r_Y1 + contract('e, ea -> a', hbar.Hov[mn][i], Gvv)

            for m in range(no):
                for n in range(no):
                    mn = m*no + n
                    iimn = ii*(no*no) + mn

                    Hvovv = contract('gea, gG -> Gea', ERI[v,i,v,v], QL[mn]) 
                    Hvovv_34swap = contract('Gae, aA -> GAe', Hvovv, QL[ii]) 
                    Hvovv_34swap = contract('GAe, eE -> GAE', Hvovv_34swap, QL[mn])
                    Hvovv = contract('Gea, eE -> GEa', Hvovv, QL[mn]) 
                    Hvovv = contract('GEa, aA -> GEA', Hvovv, QL[ii])                    

                    for _o in range(no): 
                        oo = _o*no +_o 
                        mnoo = mn*(no*no) + oo

                        Hvovv = Hvovv - contract('g, ea ->gea', Sijmn[mnoo] @ t1[_o], QL[mn].T @ ERI[_o,i,v,v] @ QL[ii]) 
                        Hvovv_34swap = Hvovv_34swap - contract('g, ea ->gea', Sijmn[mnoo] @ t1[_o], QL[ii].T @ ERI[_o,i,v,v] @ QL[mn]) 
                     
                    Gvv = -1.0 * contract('eb, gb -> ge', X2[mn], l2[mn])
                    r_Y1 = r_Y1 + contract('gea, ge -> a', -2.0 * Hvovv + Hvovv_34swap.swapaxes(1,2), Gvv)            

            #Goo terms 
            for n in range(no):
                for _o in range(no):
                    _no = n*no + _o
                    io = i*no + _o 
                    iono = io*(no*no) + _no 
                    Goo = contract('ab, ab->', Sijmn[iono] @ t2[_no] @ Sijmn[iono].T, l2[io])

                    for m in range(no):
                        mm = m*no + m
                        tmp_X = X1[m] * Goo
                        r_Y1 = r_Y1 - contract('e, ea -> a', tmp_X, QL[mm].T @ L[m,n,v,v] @ QL[ii])
   
            for n in range(no):
                for m in range(no): 
                    X_tmp = contract('e,ea ->a', X1[m], QL[mm].T @ L[m,n,v,v] @ QL[ii]) 
                    for _o in range(no):
                        _no = n*no + _o
                        io = i*no + _o
                        iono = io*(no*no) + _no
                        Goo = contract('ab, ab ->', Sijmn[iono] @ t2[_no] @ Sijmn[iono].T, l2[io]) 
                        #r_Y1 = r_Y1 - (Goo * X_tmp)  

            for j in range(no):
                jj = j*no +j 
                for n in range(no):
                    for m in range(no):
                        nm = n*no + m 
                        jm = j*no + m
                        nmjm = nm*(no*no) + jm

                        Goo = contract('ab, ab->', Sijmn[nmjm].T @ t2[nm] @ Sijmn[nmjm], l2[jm]) 
                        tmp = X1[j] * Goo
                        r_Y1 = r_Y1 - contract('f, af ->a', tmp, QL[ii].T @ L[i,n,v,v] @ QL[jj])

            for m in range(no):
                for n in range(no): 
                    mn = m*no +n 
                    _in = i*no + n 
                    mnin = mn*(no*no) + _in

                    Goo = contract('ab, ab ->', X2[mn], Sijmn[mnin] @ l2[_in] @ Sijmn[mnin].T)
                    r_Y1 = r_Y1 - (Goo * hbar.Hov[ii][m]) 

            for m in range(no):
                for _o in range(no):
                    oo = _o*no + _o
                    for n in range(no):
                        mn = m*no + n
                        on = _o*no + n
                        mnon = mn*(no*no) + on 

                        Goo = contract('ab, ab ->', X2[mn], Sijmn[mnon] @ l2[on] @ Sijmn[mnon].T ) 
                        Hooov = ERI[m,i,_o,v] @ QL[ii]
                        Hooov = Hooov + contract('f, af -> a', t1[_o], QL[ii].T @ ERI[i,m,v,v] @ QL[oo])
                        Hooov_12swap = ERI[i,m,_o,v] @ QL[ii]
                        Hooov_12swap = Hooov_12swap + contract('f, af -> a', t1[_o], QL[ii].T @ ERI[m,i,v,v] @ QL[oo])
                        r_Y1 = r_Y1 + ((-2.0 * Hooov + Hooov_12swap) * Goo) 

            # <O|L1(0)|A_bar|phi^a_i> good
            for m in range(no):
                mm = m*no + m 
                iimm = ii*(no*no) + mm

                r_Y1 = r_Y1 - (lpertbar.Aoo[i,m] * l1[m] @ Sijmn[iimm])
           
            r_Y1 = r_Y1 + contract('e, ea -> a', l1[i], lpertbar.Avv[ii]) 
     
            #<O|L2(0)|A_bar|phi^a_i> 
            for m in range(no):
                im = i*no + m
                mi = m*no + i
                mm = m*no + m 
                iimm = ii*(no*no) + mm
                miim = mi*(no*no) + im
                immm = im*(no*no) + mm

                Avvvo = 0
                #for m sum in Avvvo becomes n since m is being used for the og terms
                for n in range(no):
                    nm = n*no + m 
                    nmim = nm*(no*no) + im 
                    Avvvo = Avvvo - contract('fe,a -> fea', Sijmn[nmim].T @ t2[nm] @ Sijmn[nmim], mu[n,v].copy() @ QL[ii]) 
                r_Y1 = r_Y1 + contract('fe, fea -> a', l2[im], Avvvo) 
                
                for n in range(no):
                    nm = n*no + m 
                    mn = m*no + n 
                    mnii = nm*(no*no) + ii 
                    nmmn = nm*(no*no) + mn

                    Aovoo = contract('fe, f->e', t2[nm] @ Sijmn[nmmn], mu[i,v] @ QL[nm]) 
                    r_Y1 = r_Y1 -0.5 * contract('e, ea -> a', Aovoo, l2[mn] @ Sijmn[mnii])

                    Aovoo = contract('fe, f->e', t2[mn], mu[i,v] @ QL[mn]) 
                    r_Y1 = r_Y1 -0.5 * contract('e, ae -> a', Aovoo, Sijmn[mnii].T @ l2[mn]) 

                # <O|[Hbar(0), X1]|phi^a_i>
                Loovv = QL[ii].T @ L[i,m,v,v] @ QL[mm]
                r_Y1 = r_Y1 + 2.0 * contract('ae, e ->a', Loovv, X1[m]) 

                # <O|L1(0)|[Hbar(0), X1]|phi^a_i>          
                tmp = -1.0 * contract('a, e -> ae', hbar.Hov[ii][m], l1[i] @ Sijmn[iimm]) 
                tmp = tmp - contract('a, e -> ae', Sijmn[iimm] @ l1[m], hbar.Hov[mm][i]) 
               
                for n in range(no):
                    nn = n* no + n 
                    nnmm = nn*(no*no) + mm
                    nnii = nn*(no*no) + ii

                    Hooov = ERI[m,i,n,v] @ QL[ii]
                    Hooov_12swap = ERI[i,m,n,v] @ QL[ii]
                    Hooov = Hooov + contract('f, af -> a', t1[n], QL[ii].T @ ERI[i,m,v,v] @ QL[nn])                
                    Hooov_12swap = Hooov_12swap + contract('f, af -> a', t1[n], QL[ii].T @ ERI[m,i,v,v] @ QL[nn])
  
                    tmp = tmp + contract('a,e -> ae', -2.0 * Hooov + Hooov_12swap, l1[n] @ Sijmn[nnmm]) 
                    
                    Hooov = ERI[i,m,n,v] @ QL[mm]
                    Hooov_12swap = ERI[m,i,n,v] @ QL[mm]
                    Hooov = Hooov + contract('f, ef -> e', t1[n], QL[mm].T @ ERI[m,i,v,v] @ QL[nn])
                    Hooov_12swap = Hooov_12swap + contract('f, af -> a', t1[n], QL[mm].T @ ERI[i,m,v,v] @ QL[nn])

                    tmp = tmp + contract('e,a -> ae', -2.0 * Hooov + Hooov_12swap, l1[n] @ Sijmn[nnii])   
                    
                Hvovv = contract('fae, fF -> Fae', ERI[v,m,v,v], QL[ii]) 
                Hvovv_34swap = contract('Fea, eE -> FEa', Hvovv, QL[mm])
                Hvovv_34swap = contract('FEa, aA -> FEA', Hvovv_34swap, QL[ii])
                Hvovv = contract('Fae, aA -> FAe', Hvovv, QL[ii]) 
                Hvovv = contract('FAe, eE -> FAE', Hvovv, QL[mm]) 
    
                for n in range(no):
                    nn = n*no + n 
                    iinn = ii*(no*no) + nn

                    Hvovv = Hvovv - contract('f, ae ->fae', Sijmn[iinn] @ t1[n], QL[ii].T @ ERI[n,m,v,v] @ QL[mm]) 
                    Hvovv_34swap = Hvovv_34swap - contract('f, ea -> fea' , Sijmn[iinn] @ t1[n], QL[mm].T @ ERI[n,m,v,v] @ QL[ii])                 

                tmp = tmp + contract('fae, f -> ae', 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2), l1[i]) 
           
                Hvovv = contract('fea, fF -> Fea', ERI[v,i,v,v], QL[mm]) 
                Hvovv_34swap = contract('Fae, aA -> FAe', Hvovv, QL[ii])
                Hvovv_34swap = contract('FAe, eE -> FAE', Hvovv_34swap, QL[mm])
                Hvovv = contract('Fea, eE -> FEa', Hvovv, QL[mm])        
                Hvovv = contract('FEa, aA -> FEA', Hvovv, QL[ii]) 

                for n in range(no):
                    nn = n*no + n
                    iinn = ii*(no*no) + nn
                    mmnn = mm*(no*no) + nn

                    Hvovv = Hvovv - contract('f, ea ->fea', Sijmn[mmnn] @ t1[n], QL[mm].T @ ERI[n,i,v,v] @ QL[ii])
                    Hvovv_34swap = Hvovv_34swap - contract('f, ae -> fae' , Sijmn[mmnn] @ t1[n], QL[ii].T @ ERI[n,i,v,v] @ QL[mm])
                 
                tmp = tmp + contract('fea, f -> ae', 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2), l1[m]) 
                r_Y1 = r_Y1 + contract('ae, e -> a', tmp, X1[m]) 

                # <O|L1(0)|[Hbar(0), X2]|phi^a_i>
                for n in range(no): 
                    nn = n*no + n 
                    mn = m*no + n 
                    nm = n*no + m 
                    ni = n*no + i 
                    _in = i*no + n
                    im = i*no + m 
                    nimm = ni*(no*no) + mm
                    nnmm = nn*(no*no) + mm
                    nmii = nm*(no*no) + ii
                    inmm = _in*(no*no) + mm
                    nmmm = nm*(no*no) + mm
                    nnmn = nn*(no*no) + mn
                    iini = ii*(no*no)+ ni
                    iinn = ii*(no*no) + nn
                    iimn = ii*(no*no) + mn

                    tmp = 2.0 * contract('ef, f -> e', X2[mn], l1[n] @ Sijmn[nnmn]) 
                    tmp =  tmp - contract('fe, f -> e', X2[mn], l1[n] @ Sijmn[nnmn])
                    Loovv = QL[ii].T @ L[i,m,v,v] @ QL[mn] 
                    r_Y1 = r_Y1 + contract('ae, e -> a', Loovv, tmp) 
                
                    Goo = contract('ab, ab ->', X2[nm], self.Local.Loovv[nm][i,m]) 
                    r_Y1 = r_Y1 - (Goo * l1[n] @ Sijmn[iinn]) 
                    
                    # <O|L2(0)|[Hbar(0), X1]|phi^a_i>
                    #e_mm a_ii
                    Hovov = QL[ni].T @ ERI[m,v,n,v] @ QL[ii]
                    ERIvovv = contract('fae, fF -> Fae', ERI[v,m,v,v], QL[ni]) 
                    ERIvovv = contract('Fae, aA -> FAe', ERIvovv, QL[ii]) 
                    ERIvovv = contract('FAe, eE -> FAE', ERIvovv, QL[nn]) 
                    Hovov = Hovov + contract('e, fae -> fa', t1[n], ERIvovv)

                    for _o in range(no):
                        oo = _o*no + _o 
                        _no = n*no + _o
                        nioo = ni*(no*no) + oo 
                        nino = ni*(no*no) + _no
                        Hovov = Hovov - contract('f, a-> fa', Sijmn[nioo] @ t1[_o], ERI[m,_o,n,v] @ QL[ii]) 
                        Hovov = Hovov - contract('ef, ae -> fa', t2[_no] @Sijmn[nino].T, QL[ii].T @ ERI[_o,m,v,v] @ QL[_no])
                        tmp = contract('f, ae -> fae', Sijmn[nioo] @ t1[_o], QL[ii].T @ ERI[_o,m,v,v] @ QL[nn])
                        Hovov = Hovov - contract('e, fae -> fa', t1[n], tmp)   

                    tmp1 = -1.0 * contract('ef, fa -> ea', Sijmn[nimm].T @ l2[ni] , Hovov) 
                
                    Hovov = QL[nm].T @ ERI[i,v,n,v] @ QL[mm]
                    ERIvovv = contract('fec, fF -> Fec', ERI[v,i,v,v], QL[nm])
                    ERIvovv = contract('Fec, eE-> FEc', ERIvovv, QL[mm])
                    ERIvovv = contract('FEc, cC -> FEC', ERIvovv, QL[nn])
                    Hovov = Hovov + contract('c, fec -> fe', t1[n], ERIvovv)

                    for _o in range(no):
                        oo = _o*no + _o 
                        _no = n*no + _o
                        nmoo = nm*(no*no) + oo
                        nmno = nm*(no*no) + _no
                        Hovov = Hovov - contract('f, e-> fe', Sijmn[nmoo] @ t1[_o], ERI[i,_o,n,v] @ QL[mm]) 
                        Hovov = Hovov - contract('cf, ec -> fe', t2[_no] @ Sijmn[nmno].T, QL[mm].T @ ERI[_o,i,v,v] @ QL[_no])
                        tmp = contract('f, ec -> fec', Sijmn[nmoo] @ t1[_o], QL[mm].T @ ERI[_o,i,v,v] @ QL[nn])
                        Hovov = Hovov - contract('c, fec -> fe', t1[n], tmp) 

                    tmp1 = tmp1 -  contract('fe, af -> ea', Hovov , Sijmn[nmii].T @ l2[nm]) 
                                            
                    Hovvo = QL[_in].T @ ERI[m,v,v,n] @ QL[ii]
                    ERIovvv = contract('fae, fF -> Fae', ERI[m,v,v,v], QL[_in])
                    ERIovvv = contract('Fae, aA -> FAe', ERIovvv, QL[ii])
                    ERIovvv = contract('FAe, eE -> FAE', ERIovvv, QL[nn])
                    Hovvo = Hovvo + contract('e, fae -> fa', t1[n], ERIovvv)

                    for _o in range(no):
                        oo = _o*no + _o
                        _no = n*no + _o 
                        on = _o*no + n
                        inoo = _in*(no*no) + oo 
                        inno = _in*(no*no) + _no
                        inon = _in*(no*no) + on

                        Hovvo = Hovvo - contract('f, a -> fa', Sijmn[inoo] @ t1[_o], ERI[m,_o, v, n] @ QL[ii])
                        Hovvo = Hovvo - contract('ef, ae -> fa', t2[_no] @ Sijmn[inno].T, QL[ii].T @ ERI[m,_o,v,v] @ QL[_no]) 
                        tmp = contract('f, ae -> fae', Sijmn[inoo] @ t1[_o], QL[ii].T @ ERI[m,_o,v,v] @ QL[nn])
                        Hovvo = Hovvo - contract('e, fae -> fa', t1[n], tmp) 
                        Hovvo = Hovvo + contract('ef, ae -> fa', t2[on] @ Sijmn[inon].T , QL[ii].T @ L[m,_o,v,v] @ QL[on])

                    tmp1 = tmp1 - contract('ef,fa -> ea', Sijmn[inmm].T @ l2[_in], Hovvo) 

                    Hovvo = QL[nm].T @ ERI[i,v,v,n] @ QL[mm]
                    ERIovvv = contract('fec, fF -> Fec', ERI[i,v,v,v], QL[nm])
                    ERIovvv = contract('Fec, eE-> FEc', ERIovvv, QL[mm])
                    ERIovvv = contract('FEc, cC -> FEC', ERIovvv, QL[nn])
                    Hovvo = Hovvo + contract('c, fec -> fe', t1[n], ERIovvv)

                    for _o in range(no):
                        oo = _o*no + _o
                        _no = n*no + _o
                        on = _o*no + n
                        nmoo = nm*(no*no) + oo
                        nmno = nm*(no*no) + _no
                        nmon = nm*(no*no) + on
                        Hovvo = Hovvo - contract('f, e-> fe', Sijmn[nmoo] @ t1[_o], ERI[i,_o,v,n] @ QL[mm])
                        Hovvo = Hovvo - contract('cf, ec -> fe', t2[_no] @ Sijmn[nmno].T, QL[mm].T @ ERI[i,_o,v,v] @ QL[_no])
                        tmp = contract('f, ec -> fec', Sijmn[nmoo] @ t1[_o], QL[mm].T @ ERI[i,_o,v,v] @ QL[nn])
                        Hovvo = Hovvo - contract('c, fec -> fe', t1[n], tmp)
                        Hovvo = Hovvo + contract('cf, ec -> fe', t2[on] @ Sijmn[nmon].T, QL[mm].T @ L[i,_o,v,v] @ QL[on])
                    tmp1 = tmp1 - contract('fe,fa -> ea', Hovvo, l2[nm] @ Sijmn[nmii]) 
                    
                    for _o in range(no):
                        oo = _o*no + _o
                        on = _o*no + n
                        _no = n*no + _o 
                        nomm = _no*(no*no) + mm
                        noii = _no*(no*no) + ii 
                        onmm = on*(no*no) + mm
                        onii = on*(no*no) + ii 
                        
                        Hoooo = ERI[i,m,n,_o].copy()
                        tmp_oooo = contract('e,e ->', t1[_o], ERI[i,m,n,v] @ QL[oo]) 
                        tmp_oooo = tmp_oooo + contract('e,e ->', t1[n], ERI[m,i,_o,v] @ QL[nn]) 
                        Hoooo = Hoooo + tmp_oooo
                        Hoooo = Hoooo + contract('ef, ef->', t2[_no], ERIoovv[_no][i,m])
                        tmp =  contract('f, ef -> e', t1[_o], QL[nn].T @ ERI[i,m,v,v] @ QL[oo])
                        Hoooo = Hoooo + contract('e, e->', t1[n], tmp) 

                        tmp1 = tmp1 + 0.5 * Hoooo * (Sijmn[onmm].T @ l2[on] @ Sijmn[onii])

                        Hoooo = ERI[m,i,n,_o].copy()
                        tmp_oooo = contract('e,e ->', t1[_o], ERI[m,i,n,v] @ QL[oo])
                        tmp_oooo = tmp_oooo + contract('e,e ->', t1[n], ERI[i,m,_o,v] @ QL[nn])
                        Hoooo = Hoooo + tmp_oooo
                        Hoooo = Hoooo + contract('ef, ef->', t2[_no], ERIoovv[_no][m,i])
                        tmp =  contract('f, ef -> e', t1[_o], QL[nn].T @ ERI[m,i,v,v] @ QL[oo])
                        Hoooo = Hoooo + contract('e, e->', t1[n], tmp)

                        tmp1 = tmp1 + 0.5 * Hoooo * (Sijmn[nomm].T @ l2[_no] @ Sijmn[noii]) 

                    r_Y1 = r_Y1 + contract('ea,e ->a', tmp1, X1[m])       
                
                Hvvvv = contract('fgae, fF -> Fgae', ERI[v,v,v,v], QL[im]) 
                Hvvvv = contract('Fgae, gG -> FGae', Hvvvv, QL[im]) 
                Hvvvv = contract('FGae, aA -> FGAe', Hvvvv, QL[ii]) 
                Hvvvv = contract('FGAe, eE -> FGAE', Hvvvv, QL[mm]) 
                
                for n in range(no):
                    nn = n*no + n 
                    imnn = im*(no*no) + nn
                    ERIvovv = contract('fae, fF -> Fae', ERI[v,n,v,v], QL[im]) 
                    ERIvovv = contract('Fae, aA -> FAe', ERIvovv, QL[ii]) 
                    ERIvovv = contract('FAe, eE -> FAE', ERIvovv, QL[mm]) 
                
                    Hvvvv = Hvvvv - contract('g, fae -> fgae', Sijmn[imnn] @ t1[n], ERIvovv) 

                    ERIvovv = contract('gea, gG -> Gea', ERI[v,n,v,v], QL[im])
                    ERIvovv = contract('Gea, eE -> GEa', ERIvovv, QL[mm])
                    ERIvovv = contract('GEa, aA -> GEA', ERIvovv, QL[ii])

                    tmp = contract('f, gea -> gfea', Sijmn[imnn] @ t1[n], ERIvovv)
                    Hvvvv = Hvvvv - tmp.swapaxes(0,1).swapaxes(2,3)
             
                    for _o in range(no): 
                        _no = n*no + _o 
                        oo = _o*no + _o
                        imno = im*(no*no) + _no
                        imoo = im*(no*no) + oo
                    
                        Hvvvv = Hvvvv + contract('fg, ae -> fgae', Sijmn[imno] @ t2[_no] @ Sijmn[imno].T, QL[ii].T @ ERI[n,_o,v,v] @ QL[mm]) 
                        tmp = contract('g, ae-> gae', Sijmn[imoo] @ t1[_o], QL[ii].T @ ERI[n,_o,v,v] @ QL[mm])  
                        Hvvvv = Hvvvv + contract('f, gae -> fgae', Sijmn[imnn] @ t1[n], tmp) 
                           
                tmp1 = 0.5 * contract('fg, fgae -> ea', l2[im], Hvvvv) 
 
                Hvvvv = contract('fgea, fF -> Fgea', ERI[v,v,v,v], QL[im])
                Hvvvv = contract('Fgea, gG -> FGea', Hvvvv, QL[im])
                Hvvvv = contract('FGea, eE -> FGEa', Hvvvv, QL[mm])
                Hvvvv = contract('FGEa, aA -> FGEA', Hvvvv, QL[ii])

                for n in range(no):
                    nn = n*no + n  
                    imnn = im*(no*no) + nn 
                    ERIvovv = contract('fea, fF -> Fea', ERI[v,n,v,v], QL[im]) 
                    ERIvovv = contract('Fea, eE -> FEa', ERIvovv, QL[mm]) 
                    ERIvovv = contract('FEa, aA -> FEA', ERIvovv, QL[ii]) 
     
                    Hvvvv = Hvvvv - contract('g, fea -> fgea', Sijmn[imnn] @ t1[n], ERIvovv) 

                    ERIvovv = contract('gae, gG -> Gae', ERI[v,n,v,v], QL[im])
                    ERIvovv = contract('Gae, aA -> GAe', ERIvovv, QL[ii])
                    ERIvovv = contract('GAe, eE -> GAE', ERIvovv, QL[mm])

                    tmp = contract('f, gae -> gfae', Sijmn[imnn] @ t1[n], ERIvovv)
                    Hvvvv = Hvvvv - tmp.swapaxes(0,1).swapaxes(2,3)
     
                    for _o in range(no): 
                        _no = n*no + _o 
                        oo = _o*no + _o 
                        imno = im*(no*no) + _no
                        imoo = im*(no*no) + oo 
     
                        Hvvvv = Hvvvv + contract('fg, ea -> fgea', Sijmn[imno] @ t2[_no] @ Sijmn[imno].T, QL[mm].T @ ERI[n,_o,v,v] @ QL[ii]) 
                        tmp = contract('g, ea-> gea', Sijmn[imoo] @ t1[_o], QL[mm].T @ ERI[n,_o,v,v] @ QL[ii])  
                        Hvvvv = Hvvvv + contract('f, gea -> fgea', Sijmn[imnn] @ t1[n], tmp) 
                tmp1 = tmp1 + 0.5 * contract('gf, fgea -> ea', l2[im], Hvvvv) 
            
                r_Y1 = r_Y1 + contract('ea,e ->a', tmp1, X1[m])

                for n in range(no): 
                    for _o in range(no): 
                        _no = n*no + _o
                        io = i*no + _o
                        iono = io*(no*no) + _no

                        #Goo = contract('ab, ab ->', Sijmn[iono] @ t2[_no] @ Sijmn[iono].T, l2[io])
                        #tmp = X1[m] * Goo 
                        #r_Y1 = r_Y1 - contract('e, ea ->', tmp, QL[mm].T @ L[m,n,v,v] @ QL[ii]) 
        
#tmp  =  contract('me,ni->meni', X1, cclambda.build_Goo(t2, l2))
        #r_Y1 -= contract('meni,mnea->ia', tmp, L[o,o,v,v])
        #tmp  =  contract('jf,nj->fn', X1, cclambda.build_Goo(t2, l2))
        #r_Y1 -= contract('inaf,fn->ia', L[o,o,v,v], tmp)

        #i__
            #m__
            ## <O|L2(0)|[Hbar(0), X2]|phi^a_i>
            #r_Y1 -= contract('mi,ma->ia', cclambda.build_Goo(X2, l2), hbar.Hov)
            #r_Y1 += contract('ie,ea->ia', hbar.Hov, cclambda.build_Gvv(l2, X2))
                for n in range(no):
                    mn = m*no + n 
                    ni = n*no + i 
                    immn = im*(no*no) + mn
                    mimn = mi*(no*no) + mn
                    iimn = ii*(no*no) + mn 
                    mnni = mn*(no*no) + ni 

                    #g_im e_mn
                    tmp = contract('fg,ef->ge', Sijmn[immn].T @ l2[im], X2[mn]) 
                    Hvovv = contract('gea, gG -> Gea', ERI[v,n,v,v], QL[im]) 
                    Hvovv = contract('Gea, eE -> GEa', Hvovv, QL[mn])
                    Hvovv = contract('GEa, aA -> GEA', Hvovv, QL[ii]) 

                    for _o in range(no):
                        oo = _o*no + _o
                        imoo = im*(no*no) + oo

                        Hvovv = Hvovv - contract('g, ea -> gea', Sijmn[imoo] @ t1[_o], QL[mn].T @ ERI[_o,n,v,v] @ QL[ii]) 

                    r_Y1 = r_Y1 - contract('ge, gea -> a', tmp, Hvovv) 
                     
                    #g_mi e_mn
                    tmp = contract('fg,ef->ge', Sijmn[mimn].T @ l2[mi], X2[mn])       
                    Hvovv = contract('gae, gG -> Gae', ERI[v,n,v,v], QL[mi]) 
                    Hvovv = contract('Gae, aA -> GAe', Hvovv, QL[ii])
                    Hvovv = contract('GAe, eE -> GAE', Hvovv, QL[mn]) 

                    for _o in range(no):
                        oo = _o*no + _o 
                        mioo = mi*(no*no) + oo 

                        Hvovv = Hvovv - contract('g, ae -> gae', Sijmn[mioo] @ t1[_o], QL[ii].T @ ERI[_o,n,v,v] @ QL[mn]) 
 
                    r_Y1 = r_Y1 - contract('ge, gae -> a', tmp, Hvovv) 
            
                    #g_mn a_ii e_mn f_mn
                    #v^4
                    tmp = contract('ga,ef->gaef', l2[mn] @ Sijmn[iimn].T , X2[mn])
                    Hvovv = contract('gef, gG -> Gef', ERI[v,i,v,v], QL[mn])
                    Hvovv = contract('Gef, eE -> GEf', Hvovv, QL[mn])
                    Hvovv = contract('GEf, fF -> GEF', Hvovv, QL[mn]) 
 
                    for _o in range(no):
                        oo = _o*no + _o
                        mnoo = mn*(no*no) + oo

                        Hvovv = Hvovv - contract('g, ef -> gef', Sijmn[mnoo] @ t1[_o], ERIoovv[mn][_o,i]) # QL[mn].T @ ERI[_o,n,v,v] @ QL[mn]) 

                    r_Y1 = r_Y1 - contract('gef, gaef -> a', Hvovv, tmp)

                    #g_ni e_mn f_mn a_ii
                    Hvovv = contract('gae, gG -> Gae', ERI[v,m,v,v], QL[ni])
                    Hvovv_34swap = contract('Gea, eE -> GEa', Hvovv, QL[mn])
                    Hvovv_34swap = contract('GEa, aA -> GEA', Hvovv_34swap, QL[ii])
                    Hvovv = contract('Gae, aA -> GAe', Hvovv, QL[ii])
                    Hvovv = contract('GAe, eE -> GAE', Hvovv, QL[mn])
 
                    for _o in range(no):
                        oo = _o*no + _o
                        nioo = ni*(no*no) + oo
                        Hvovv = Hvovv - contract('g, ae -> gae', Sijmn[nioo] @ t1[_o], QL[ii].T @ ERI[_o,m,v,v] @ QL[mn]) 
                        Hvovv_34swap = Hvovv_34swap - contract('g,ea -> gea', Sijmn[nioo] @ t1[_o], QL[mn].T @ ERI[_o,m,v,v] @ QL[ii]) 
                    tmp = contract('gae, ef -> gaf', 2.0 * Hvovv - Hvovv_34swap.swapaxes(1,2), X2[mn]) 
                    r_Y1 = r_Y1 + contract('fg, gaf -> a', Sijmn[mnni] @ l2[ni], tmp) 

        ##can combine the next two to swapaxes type contraction
        #r_Y1 -= 2.0 * contract('giea,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))
        #r_Y1 += contract('giae,ge->ia', hbar.Hvovv, cclambda.build_Gvv(X2, l2))

                    for _o in range(no): 
                        oi = _o*no + i 
                        oo = _o*no + _o
                        mo = m*no + _o 
                        on = _o*no + n
                        _no = n*no + _o
                        oimn = oi*(no*no) + mn
                        momn = mo*(no*no) + mn                        
                        moii = mo*(no*no) + ii 
                        onmn = on*(no*no) + mn
                        noii = _no*(no*no) + ii
                        onii = on*(no*no) + ii
                        nomn = _no*(no*no) + mn
 
                        tmp = contract('ef, ef ->', Sijmn[oimn].T @ l2[oi] @ Sijmn[oimn], X2[mn])
                        Hooov = ERI[m,n,_o,v] @ QL[ii] 
                        Hooov = Hooov + contract('f, af -> a', t1[_o], QL[ii].T @ ERI[n,m,v,v] @ QL[oo])
                        r_Y1 = r_Y1 + (tmp * Hooov) 
    
                        tmp = contract('fa, ef -> ae', Sijmn[momn].T @ l2[mo] @ Sijmn[moii], X2[mn])
                        Hooov = ERI[i,n,_o,v] @ QL[mn]
                        Hooov = Hooov + contract('f, ef -> e', t1[_o], QL[mn].T @ ERI[n,i,v,v] @ QL[oo])
                        r_Y1 = r_Y1 + contract('e, ae -> a', Hooov, tmp) 

                        #a_ii f_mn
                        tmp = contract('ea, ef -> af', Sijmn[onmn].T @ l2[on] @ Sijmn[onii], X2[mn]) 
                        Hooov = ERI[m,i,_o,v] @ QL[mn] 
                        Hooov = Hooov + contract('e,fe  -> f', t1[_o], QL[mn].T @ ERI[i,m,v,v] @ QL[oo])
                        r_Y1 = r_Y1 + contract('f, af -> a', Hooov, tmp) 

        ##can combine the next two to swapaxes type contraction
        #r_Y1 -= 2.0 * contract('mioa,mo->ia', hbar.Hooov, cclambda.build_Goo(X2, l2))
        #r_Y1 += contract('imoa,mo->ia', hbar.Hooov, cclambda.build_Goo(X2, l2))

                        Hooov = ERI[i,m,_o,v] @ QL[mn] 
                        Hooov = Hooov + contract('f, ef -> e', t1[_o], QL[mn].T @ ERI[m,i,v,v] @ QL[oo])
                        Hooov_12swap = ERI[m,i,_o,v] @ QL[mn]
                        Hooov_12swap = Hooov_12swap + contract('f, ef -> e', t1[_o], QL[mn].T @ ERI[i,m,v,v] @ QL[oo])
                        tmp = contract('e, ef -> f', -2.0 * Hooov + Hooov_12swap, X2[mn]) 
                        r_Y1 = r_Y1 + contract('f, fa -> a', tmp, Sijmn[nomn].T @ l2[_no] @ Sijmn[noii]) 
            #r_Y1 = r_Y1 + contract('e, ea->a', l1[i], Gvv)
            in_Y1.append(r_Y1)
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
        return r_Y1 

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

        return r_Y1
   
    def in_Y2(self, pertbar, X1, X2):
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

        return r_Y2

    def in_lY2(self, lpertbar, X1, X2):
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
        
        #G_in = np.zeros((no,no))
        #for i in range(self.no):
            #for j in range(self.no):
                #ij = i*self.no + j

                #for n in range(self.no):
                    #nj = n*self.no + j
                    #ijn = ij*self.no + n

                    #tmp = self.Local.Loovv[nj][i,j]
                    #G_in[i,n] += contract('ab,ab->',tmp,l2[nj])

        #Goo_LX = np.zeros((self.no,self.no))
        #for i in range(self.no):
            #for j in range(self.no):
                #ij = i*self.no + j

                #for m in range(self.no):
                    #mj = m*self.no + j
                    #ijm = ij*self.no + m

                    #tmp = Sijmj[ijm] @ X2[mj]
                    #tmp = tmp @ Sijmj[ijm].T
                    #Goo_LX[i,m] += contract('ab,ab->',self.Local.Loovv[ij][i,j], tmp)

        #Gvv terms needed for Expression 5, Term 7
        #for i in range(no):
            #for j in range(no):
                #ij = i*no + j 
                #Gvv term needed for Expression 5, Term 7
                #self.Gaf.append(-1.0 * contract('fb, ab-> af', X2[ij], self.Local.Loovv[ij][i,j])) 
                #print("Gaf", self.Gaf[ij].shape)

                #Gvv term needed for Expression 5, Term 8 
                #self.Gae.append(-1.0 * contract('eb, ab->ae', X2[ij], l2[ij]))

        for i in range(no):
            ii = i*no + i 
            for j in range(no):
                ij = i*no + j
                jj = j*no + j
                
                #Gvv term needed for Expression 5, Term 7
                #Gaf = contract('fb, ab->af', X2[ij], self.Local.Loovv[ij][i,j]) 
 
                #Gvv term needed for Expression 5, Term 8 
                #Gae = contract('eb, ab->ae', X2[ij], l2[ij])

                # <O|L1(0)|A_bar|phi^ab_ij>, Eqn 162
                r_Y2  = 2.0 * contract('a,b->ab', l1[i] @ Sijii[ij].T, lpertbar.Aov[ij][j].copy())
                r_Y2 = r_Y2 - contract('a,b->ab', l1[j] @ Sijjj[ij].T, lpertbar.Aov[ij][i].copy())

                # <O|L2(0)|A_bar|phi^ab_ij>, Eqn 163
                r_Y2 += contract('eb,ea->ab', l2[ij], lpertbar.Avv[ij])
               
                #collecting Gvv terms here
                for m in range(no):
                    for n in range(no):
                        mn = m*no + n
                        ijmn = ij*(no*no) + mn

                        Gvv = -1.0 * contract('fe, ae ->af', X2[mn], QL[ij].T @ L[m,n,v,v] @ QL[mn])
                        r_Y2 = r_Y2 + contract('fb, af -> ab', Sijmn[ijmn].T @ l2[ij], Gvv)                            

                for m in range(no):
                    for n in range(no):
                        mn = m*no + n
                        ijmn = ij*(no*no) + mn 

                        Gvv = -1.0 * contract('ef, bf -> be', X2[mn], Sijmn[ijmn] @ l2[mn])
                        r_Y2 = r_Y2 + contract('ae, be -> ab', QL[ij].T @ L[i,j,v,v] @ QL[mn], Gvv)             
       
                #last Goo here
                for m in range(no): 
                    for n in range(no): 
                        mn = m*no + n 
                        jn = j*no + n 
                        mnjn = mn*(no*no) + jn

                        Goo = contract('ab, ab ->', Sijmn[mnjn] @ l2[jn] @ Sijmn[mnjn].T, X2[mn])
                        r_Y2 = r_Y2 - (self.Local.Loovv[ij][i,m] * Goo)             

                for m in range(no):
                    mj = m*no + j
                    ijm = ij*no + m

                    tmp = Sijmj[ijm] @ l2[mj] @ Sijmj[ijm].T 
                    r_Y2 = r_Y2 - lpertbar.Aoo[i,m] * tmp

                # <O|L1(0)|[Hbar(0), X1]|phi^ab_ij>, Eqn 164
                for m in range(no):
                    ijm = ij*no + m
                    mm = m*no + m
                    iim = ii*no + m

                    tmp = contract('e,a->ea', X1[m], (l1[j] @ Sijjj[ij].T))
                    tmp1 = contract('eb, eE, bB ->EB', L[m,i,v,v], QL[mm], QL[ij])  
                    r_Y2 = r_Y2 - contract('eb, ea-> ab', tmp1, tmp) 
        
                    tmp = contract('e,b->eb', X1[m], (l1[m] @ Sijmm[ijm].T))  
                    tmp1 = contract('ae, aA, eE ->AE', L[i,j,v,v], QL[ij], QL[mm]) 
                    r_Y2 = r_Y2 - contract('ae, eb-> ab', tmp1, tmp) 

                    tmp = contract('e,e->', X1[m], (l1[i] @ Sijmm[iim].T))   
                    r_Y2 = r_Y2 - tmp * self.Local.Loovv[ij][j,m].swapaxes(0,1) 

                    tmp = 2.0 * contract('e,b ->eb', X1[m], (l1[j] @ Sijjj[ij].T))
                    tmp1 = contract('ae, aA, eE ->AE', L[i,m,v,v], QL[ij], QL[mm])
                    r_Y2 = r_Y2 + contract('ae, eb-> ab', tmp1, tmp)
           
                # <O|L2(0)|[Hbar(0), X1]|phi^ab_ij>, Eqn 165
                for m in range(no):
                    mm = m*no + m
                    ijm = ij*no + m
                    jm = j*no + m
                    im = i*no + m 
 
                    tmp = contract('e,a-> ea',  X1[m], hbar.Hov[ij][m]) 
                    r_Y2 = r_Y2 - contract('eb,ea->ab', Sijmm[ijm].T @ l2[ij], tmp)
                    
                    tmp = contract('e,e->', X1[m], hbar.Hov[mm][i]) 
                    r_Y2 = r_Y2 - tmp * (Sijmj[ijm] @ l2[jm] @ Sijmj[ijm].T).swapaxes(0,1) 
 
                    #may need to double-check this one
                    tmp = contract('e,ef->f', X1[m], Sijmm[ijm].T @ l2[ij]) 
                    r_Y2 = r_Y2 - contract('f, fba -> ab', tmp, hbar.Hvovv_ij[ij][:,m,:,:])

                    tmp = contract('e,bf->ebf', X1[m], Sijim[ijm] @ l2[im]) 
                    r_Y2 = r_Y2 - contract('ebf, fea->ab', tmp, hbar.Hfjea[ijm]) 
 
                    tmp = contract('e,fa->efa', X1[m], l2[jm] @ Sijmj[ijm].T)
                    r_Y2 = r_Y2 - contract('fbe, efa->ab', hbar.Hfibe[ijm], tmp) 

                    tmp = contract('e, fae -> fa', X1[m], 2.0 * hbar.Hfmae[ijm] - hbar.Hfmea[ijm].swapaxes(1,2))
                    r_Y2 = r_Y2 + contract('fb,fa->ab', l2[ij], tmp)

                    tmp = contract('e, fea -> fa', X1[m], 2.0 * hbar.Hfieb[ijm] - hbar.Hfibe[ijm].swapaxes(1,2))
                    r_Y2  = r_Y2 + contract('fa,bf->ab', tmp, Sijmj[ijm] @ l2[jm]) 
 
                    for n in range(no):
                        ijmn = ijm*no + n 
                        _in = i*no + n
                        inm = _in * no + m
                        ijn = ij*no + n
                        ni = n*no + i
                        nim = ni*no + m
                        nm = n*no + m
                        ijnm = ij*(no*no) + nm
                        nj = n*no + j
                        njm = nj*no + m 
                        jn = j*no + n

                        imn = im*no + n

                        tmp = contract('e,a -> ea', X1[m], hbar.Hjmna[ijmn])
                        tmp1 = Sijmm[inm].T @ l2[_in] @ Sijim[ijn].T  
                        r_Y2 = r_Y2 + contract('eb,ea->ab', tmp1, tmp) 

                        tmp = contract('e,a -> ea', X1[m], hbar.Hmjna[ijmn])
                        tmp1 = Sijmm[nim].T @ l2[ni] @ Sijim[ijn].T  
                        r_Y2 = r_Y2 + contract('eb,ea->ab', tmp1, tmp) 

                        tmp = Sijmn[ijnm] @ l2[nm] @ Sijmn[ijnm].T
                        tmp = contract('e,ba->eba', X1[m], tmp) 
                        r_Y2 = r_Y2 + contract('eba,e->ab', tmp, hbar.Hjine[ijmn])

                        tmp = contract('e,a->ea', X1[m], 2.0*hbar.Hmine[ijmn] - hbar.Himne[ijmn])
                        tmp1 = Sijmm[njm].T @ l2[nj] @ Sijmj[ijn].T  
                        r_Y2 = r_Y2 - contract('ea, eb->ab', tmp, tmp1)     
 
                        tmp = contract('e,e->', X1[m], 2.0*hbar.Himne_mm[imn] - hbar.Hmine_mm[imn])
                        tmp1 = Sijmj[ijn] @ l2[jn] @ Sijmj[ijn].T
                        r_Y2 = r_Y2 - tmp * tmp1.swapaxes(0,1)

                #<O|L2(0)|[Hbar(0), X2]|phi^ab_ij>, Eqn 174
                Gin = np.zeros((no, no))
                for m in range(no):
                    ijm = ij*no + m
                    mi = m*no + i
                    im = i*no + m
                    mj = m*no + j 
                    for n in range(no):
                        mn = m*no +n
                        ijmn = ijm*(no) + n
                        imn = i*(no*no) + mn
                        min = mi*no + n 
                        mni = mn*no + i
                        nm = n*no + m
                        inm = i*(no*no) + nm
                        mjn = mj*no + n
                        jn = j*no + j
                        ijn = i*(no*no) + jn
                        ni = n*no + i 
                        nim = ni*no + m 
                        nj = n*no + j 
                        njm = nj*no + m
                        mnni = mn*(no*no) + ni
                        ijni = ij*(no*no) + ni
                        ijnj = ij*(no*no) + nj
                        mnnj = mn*(no*no) + nj

                        tmp = Sijmn[ijmn].T  @ l2[ij] @ Sijmn[ijmn]
                        tmp = 0.5 * contract('ef,ef->', tmp, X2[mn]) 
                        r_Y2 = r_Y2 + tmp * self.Local.ERIoovv[ij][m,n]
        
                        tmp = Sijmn[ijmn] @ l2[mn] @ Sijmn[ijmn].T
                        tmp1 = 0.5 * contract('fe,ef->', self.Local.ERIoovv[mn][i,j], X2[mn])
                        r_Y2 = r_Y2 + tmp1 * tmp.swapaxes(0,1)

                        tmp = Sijim[min].T @ l2[mi] @ Sijim[ijm].T    
                        tmp = contract('fb, ef-> be', tmp, X2[mn]) 
                        r_Y2 = r_Y2 + contract('be, ae->ab', tmp, QL[ij].T @ ERI[j,n,v,v] @ QL[mn]) 
                        
                        tmp = Sijim[min].T @ l2[im] @ Sijim[ijm].T
                        tmp = contract('fb, ef-> be', tmp, X2[mn])
                        r_Y2 = r_Y2 + contract('be, ae->ab', tmp, QL[ij].T @ ERI[n,j,v,v] @ QL[mn])                        
                         
                        tmp = Sijim[mjn].T @ l2[mj] @ Sijmj[ijm].T
                        tmp = contract('fb, ef-> be', tmp, X2[mn])
                        r_Y2 = r_Y2 - contract('be, ae->ab', tmp, QL[ij].T @ L[i,n,v,v] @ QL[mn])
 
                        # Expression 5, Term 10 
                        tmp = Sijmn[mnni] @ l2[ni] @ Sijmn[ijni].T 
                        tmp = contract('fb, ef-> be', tmp, X2[mn]) 
                        r_Y2 = r_Y2 - contract('be,ea->ab', tmp, QL[mn].T @ L[m,j,v,v] @ QL[ij]) 

                        # Expression 5, Term 11
                        tmp = Sijmn[mnnj] @ l2[nj] @ Sijmn[ijnj].T
                        tmp = 2.0 * contract('fb, ef-> be', tmp, X2[mn]) 
                        r_Y2 = r_Y2 + contract('ae, be-> ab', QL[ij].T @ L[i,m,v,v] @ QL[mn], tmp)  
 
                        #Goo term for Term 6   
                        Gin[i,n] += contract('ef,ef->', QL[nm].T @ L[i,m,v,v] @ QL[nm], X2[nm]) 
 
                for n in range(no):
                    ijn = ij*no + n
                    jn = j*no + n  
 
                    #Term 6
                    tmp = Sijmj[ijn] @ l2[jn] @ Sijmj[ijn].T
                    r_Y2 = r_Y2 - Gin[i,n] * tmp.swapaxes(0,1)
  
                in_Y2.append(r_Y2) 

        return in_Y2
        
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

    def lr_Y2(self, lpertbar, omega):
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
        return lr_Y2

    def pseudoresponse(self, pertbar, X1, X2):
        contract = self.ccwfn.contract
        polar1 = 2.0 * contract('ai,ia->', np.conj(pertbar.Avo), X1)
        polar2 = 2.0 * contract('ijab,ijab->', np.conj(pertbar.Avvoo), (2.0*X2 - X2.swapaxes(2,3)))
        return -2.0*(polar1 + polar2)

    def local_pseudoresponse(self, lpertbar, X1, X2):
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
