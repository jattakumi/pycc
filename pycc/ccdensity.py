"""
ccdensity.py: Builds the CC density.
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")

import time
import numpy as np
import torch
from .cctriples import t3c_ijk, t3c_abc, l3_ijk, l3_abc, t3c_bc, l3_bc 

class ccdensity(object):
    """
    An RHF-CC Density object.

    Attributes
    ----------
    Dov : NumPy array
        The occupied-virtual block of the one-body density.
    Dvo : NumPy array
        The virtual-occupied block of the one-body density.
    Dvv : NumPy array
        The virtual-virtual block of the one-body density.
    Doo : NumPy array
        The occupied-occupied block of the one-body density.
    Doooo : NumPy array
        The occ,occ,occ,occ block of the two-body density.
    Dvvvv : NumPy array
        The vir,vir,vir,vir block of the two-body density.
    Dooov : NumPy array
        The occ,occ,occ,vir block of the two-body density.
    Dvvvo : NumPy array
        The vir,vir,vir,occ block of the two-body density.
    Dovov : NumPy array
        The occ,vir,occ,vir block of the two-body density.
    Doovv : NumPy array
        The occ,occ,vir,vir block of the two-body density.
        The occ,vir,occ,occ block of the two-body density.

    Methods
    -------
    compute_energy() :
        Compute the CC energy from the density.  If only onepdm is available, just compute the one-electron energy.
    compute_onepdm() :
        Compute the one-electron density for a given set of amplitudes (useful for RTCC)
    """
    def __init__(self, ccwfn, cclambda, onlyone=False):
        """
        Parameters
        ----------
        ccwfn : PyCC ccwfn object
            contains the necessary T-amplitudes (either instantiated to defaults or converged)
        cclambda : PyCC cclambda object
            Contains the necessary Lambda-amplitudes (instantiated to defaults or converged)
        onlyone : Boolean
            only compute the onepdm if True

        Returns
        -------
        None
        """

        time_init = time.time()
        self.ccwfn = ccwfn
        self.cclambda = cclambda
        self.contract = self.ccwfn.contract

        o = ccwfn.o
        v = ccwfn.v
        no = ccwfn.no
        #comment it out 
        self.no = ccwfn.no
        nv = ccwfn.nv

        if ccwfn.local is not None and ccwfn.filter is not True:
            self.lccwfn = ccwfn.lccwfn
            self.local = ccwfn.local
            self.model = ccwfn.model
            self.Local = ccwfn.Local
            self.cclambda = cclambda 

            # lDvv needed
            t1 = self.lccwfn.t1
            l1 = self.cclambda.l1
            t2 = self.lccwfn.t2 
            l2 = self.cclambda.l2
            self.Doo = self.build_lDoo(t1, t2, l1, l2)
            self.Dvv, self.vv_energy = self.build_lDvv(t1, t2, l1, l2)
            self.Dov = self.build_lDov(t1, t2, l1, l2)
            #print("=== lDoo ===\n", self.Doo)
            #print("=== lDvv ===\n",self.Dvv)
            #print("=== lDov ===\n", self.Dov)
            #print("Local energy:\n", self.compute_local_energy(lDvv, t1, t2, l1, l2))
        else: 
            F = ccwfn.H.F
            ERI = ccwfn.H.ERI
            L = ccwfn.H.L
            t1 = ccwfn.t1
            t2 = ccwfn.t2
            l1 = cclambda.l1
            l2 = cclambda.l2

            self.Dov = self.build_Dov(t1, t2, l1, l2)
            self.Dvo = self.build_Dvo(l1)
            self.Dvv = self.build_Dvv(t1, t2, l1, l2)
            self.Doo = self.build_Doo(t1, t2, l1, l2)

          #  print(self.Doo)
            if self.ccwfn.filter is True:
                Q = self.ccwfn.Local.Q
                L = self.ccwfn.Local.L
                for ij in range(self.no**2):
                    print("Dvv_ij", ij, (Q[ij] @ L[ij]).T @ self.Dvv @ (Q[ij] @ L[ij]))
            #print(self.Dvv)
            self.onlyone = onlyone

            #if onlyone is False:
                #self.Doooo = self.build_Doooo(t1, t2, l2)
                #self.Dvvvv = self.build_Dvvvv(t1, t2, l2)
                #self.Dooov = self.build_Dooov(t1, t2, l1, l2)
                #self.Dvvvo = self.build_Dvvvo(t1, t2, l1, l2)
                #self.Dovov = self.build_Dovov(t1, t2, l1, l2)
                #self.Doovv = self.build_Doovv(t1, t2, l1, l2)

            print("\nCCDENSITY constructed in %.3f seconds.\n" % (time.time() - time_init))

    def compute_energy(self):
        """
        Compute the CC energy from the density.  If only onepdm is available, just compute the one-electron energy.

        Parameters
        ----------
        None

        Returns
        -------
        ecc | float
            CC correlation energy computed using the one- and two-electron densities
        """

        o = self.ccwfn.o
        v = self.ccwfn.v
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI

        contract = self.contract 

        # We assume here that the Brillouin condition holds
        oo_energy = contract('ij,ij->', F[o,o], self.Doo)
        vv_energy = contract('ab,ab->', F[v,v], self.Dvv)
        eone = vv_energy #+ vv_energy
        print("One-electron CC energy = %20.15f" % eone)

        if self.onlyone is True:
            print("Only one-electron density available.")
            ecc = eone
        else:
            oooo_energy = 0.5 * contract('ijkl,ijkl->', ERI[o,o,o,o], self.Doooo)
            vvvv_energy = 0.5 * contract('abcd,abcd->', ERI[v,v,v,v], self.Dvvvv)
            ooov_energy = contract('ijka,ijka->', ERI[o,o,o,v], self.Dooov)
            vvvo_energy = contract('abci,abci->', ERI[v,v,v,o], self.Dvvvo)
            ovov_energy = contract('iajb,iajb->', ERI[o,v,o,v], self.Dovov)
            oovv_energy = 0.5 * contract('ijab,ijab->', ERI[o,o,v,v], self.Doovv)
            etwo = oooo_energy + vvvv_energy + ooov_energy + vvvo_energy + ovov_energy + oovv_energy

            print("OOOO Energy = %20.15f" % oooo_energy)
            print("VVVV Energy = %20.15f" % vvvv_energy)
            print("OOOV Energy = %20.15f" % ooov_energy)
            print("VVVO Energy = %20.15f" % vvvo_energy)
            print("OVOV Energy = %20.15f" % ovov_energy)
            print("OOVV Energy = %20.15f" % oovv_energy)
            print("Two-electron CC energy = %20.15f" % etwo)
            ecc = eone + etwo

        print("CC Correlation Energy  = %20.15f" % ecc)

        self.ecc = ecc
        self.eone = eone
        self.etwo = etwo

        return ecc

    def compute_onepdm(self, t1, t2, l1, l2, withref=False):
        """
        Parameters
        ----------
        t1, t2, l1, l2 : NumPy arrays
            current cluster amplitudes
        withref : Boolean (default: False)
            include the reference contribution if True

        Returns
        -------
        onepdm : NumPy array
            the CC one-electron density as a single, full matrix
        """
        o = self.ccwfn.o
        v = self.ccwfn.v
        no = self.ccwfn.no
        nv = self.ccwfn.nv
        nt = no + nv
        F = self.ccwfn.H.F
        ERI = self.ccwfn.H.ERI
        L = self.ccwfn.H.L
 
        if isinstance(t1, torch.Tensor):
            if self.ccwfn.precision == 'DP':
                opdm = torch.zeros((nt, nt), dtype=torch.complex128, device=self.ccwfn.device1)
            elif self.ccwfn.precision == 'SP':
                opdm = torch.zeros((nt, nt), dtype=torch.complex64, device=self.ccwfn.device1)
        else:
            if self.ccwfn.precision == 'DP':
                opdm = np.zeros((nt, nt), dtype='complex128')
            elif self.ccwfn.precision == 'SP':
                opdm = np.zeros((nt, nt), dtype='complex64')
        opdm[o,o] = self.build_Doo(t1, t2, l1, l2)
        if withref is True:
            if isinstance(t1, torch.Tensor):
                if self.ccwfn.precision == 'DP':
                    opdm[o,o] += 2.0 * torch.eye(no, dtype=torch.complex128, device=self.ccwfn.device1)  # Reference contribution
                elif self.ccwfn.precision == 'SP':
                    opdm[o,o] += 2.0 * torch.eye(no, dtype=torch.complex64, device=self.ccwfn.device1)
            else:
                if self.ccwfn.precision == 'DP':
                    opdm[o,o] += 2.0 * np.eye(no)  # Reference contribution
                elif self.ccwfn.precision == 'SP':
                    opdm[o,o] += 2.0 * np.eye(no, dtype=np.complex64)

        opdm[v,v] = self.build_Dvv(t1, t2, l1, l2)
        opdm[o,v] = self.build_Dov(t1, t2, l1, l2)
        opdm[v,o] = self.build_Dvo(l1)

        if self.ccwfn.model == 'CC3':
            Fov = self.ccwfn.build_Fme(o, v, F, L, t1)
            Wvvvo = self.ccwfn.build_cc3_Wabei(o, v, ERI, t1)
            Woooo = self.ccwfn.build_cc3_Wmnij(o, v, ERI, t1)
            Wovoo = self.ccwfn.build_cc3_Wmbij(o, v, ERI, t1, Woooo)
            Wvovv = self.ccwfn.build_cc3_Wamef(o, v, ERI, t1)
            Wooov = self.ccwfn.build_cc3_Wmnie(o, v, ERI, t1)

            opdm[o,v] += self.build_cc3_Dov(o, v, no, nv, F, L, t1, t2, l1, l2, Wvvvo, Wovoo, Fov, Wvovv, Wooov)

            # Density matrix blocks in contractions with T1-transformed dipole integrals
            if isinstance(t1, torch.Tensor):
                opdm_cc3 = torch.zeros_like(opdm)
            else:
                opdm_cc3 = np.zeros_like(opdm)
            opdm_cc3[o,o] += self.build_cc3_Doo(o, v, no, nv, F, L, t2, l1, l2, Fov, Wvvvo, Wovoo, Wvovv, Wooov)
            opdm_cc3[v,v] += self.build_cc3_Dvv(o, v, no, nv, F, L, t2, l1, l2, Fov, Wvvvo, Wovoo, Wvovv, Wooov)

            return (opdm, opdm_cc3)

        else:
            return opdm

    def build_Doo(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        o = self.ccwfn.o 
        if self.ccwfn.model == 'CCD':
            Doo = -contract('imef,jmef->ij', t2, l2)
            self.oo_energy = contract('ij,ij->', self.ccwfn.H.F[o,o], Doo)
            print("sim oo energy", self.oo_energy)
        else:
            Doo = -1.0 * contract('ie,je->ij', t1, l1)
            Doo -= contract('imef,jmef->ij', t2, l2)
            # (T) contributions computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Doo += self.ccwfn.Doo

        return Doo

    def build_lDoo(self, t1, t2, l1, l2):  # beginning - implementing the ccd-density
        if self.ccwfn.model == 'CCD': 
            lDoo = np.zeros((self.ccwfn.no, self.ccwfn.no))
            contract = self.contract
            oo_energy = 0
            for ij in range (self.ccwfn.no * self.ccwfn.no):
                i = ij // self.ccwfn.no 
                j = ij % self.ccwfn.no
                for m in range(self.ccwfn.no):
                    im = i * self.ccwfn.no + m
                    jm = j * self.ccwfn.no + m
                    Q = self.ccwfn.Local.Q
                    L = self.ccwfn.Local.L
                    Q_im = Q[im] @ L[im]
                    Q_jm = Q[jm] @ L[jm]
                    S_jmim = (Q_jm).T @ (Q_im)
                    temp = S_jmim @ t2[im] @ S_jmim.T  
                    lDoo[i,j] -= contract('ef, ef-> ', temp, l2[jm])
                oo_energy += self.ccwfn.H.F[i,j] * lDoo[i,j] #contract(',ij->', self.ccwfn.H.F[i,j], lDoo)
        else:
            contract = self.contract
            lDoo_1 = np.zeros((self.ccwfn.no, self.ccwfn.no))
            lDoo_2 = np.zeros((self.ccwfn.no, self.ccwfn.no))
            for ij in range (self.ccwfn.no * self.ccwfn.no):
                i = ij // self.ccwfn.no
                j = ij % self.ccwfn.no
                ii = i * self.no + i
                jj = j * self.no + j
                Q = self.ccwfn.Local.Q
                L = self.ccwfn.Local.L
                Q_i = Q[ii] @ L[ii]
                Q_j = Q[jj] @ L[jj]
                S_jjii = (Q_j).T @ (Q_i)
                temp = S_jjii @ t1[i]
                lDoo_1[i,j] -= contract('e,e->', temp, l1[j])
                for m in range(self.ccwfn.no):
                    im = i * self.ccwfn.no + m
                    jm = j * self.ccwfn.no + m
                    Q = self.ccwfn.Local.Q
                    L = self.ccwfn.Local.L
                    Q_im = Q[im] @ L[im]
                    Q_jm = Q[jm] @ L[jm]
                    S_jmim = (Q_jm).T @ (Q_im)
                    temp = S_jmim @ t2[im] @ S_jmim.T
                    lDoo_2[i,j] -= contract('ef,ef->', temp, l2[jm])
            lDoo = lDoo_1 + lDoo_2
            #(T) contributions computed in ccwfn.t3_density()
            #if self.ccwfn.model == 'CCSD(T)':
                #Doo += self.ccwfn.Doo

        return lDoo#, oo_energy



    def build_Dvv(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        v = self.ccwfn.v
        if self.ccwfn.model == 'CCD':
            Dvv = contract('mnbe,mnae->ab', t2, l2)
            self.vv_energy = contract('ab,ab->', self.ccwfn.H.F[v, v], Dvv)
            print("sim vv energy", self.vv_energy)
            # collect variables needed to transform Dvv to Dvv^ij
            if self.ccwfn.filter is True:
                Q = self.ccwfn.Local.Q
                L = self.ccwfn.Local.L
                for ij in range(self.no**2):
                    print("Dvv_ij", ij, (Q[ij] @ L[ij]).T @ Dvv @ (Q[ij] @ L[ij]))
                
        else:
            Dvv = contract('mb,ma->ab', t1, l1)
            Dvv += contract('mnbe,mnae->ab', t2, l2)
            # (T) contributions computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Dvv += self.ccwfn.Dvv

        return Dvv
    
    def build_lDvv(self, t1, t2, l1, l2):  # begining - implementing the ccsd & ccd-density
        lDvv = []
        if self.ccwfn.model == 'CCD':
            contract = self.contract
            vv_energy = 0
            for ij in range(self.ccwfn.no * self.ccwfn.no):
                Dvv = np.zeros((self.Local.dim[ij], self.Local.dim[ij]))
                #for mn in range(self.ccwfn.no * self.ccwfn.no):
                    #S_ijmn = self.Local.QL[ij].T @ self.Local.QL[mn]
                    #emp_t2 = S_ijmn @ t2[mn]
                    #temp_l2 = S_ijmn @ l2[mn]
                Dvv = contract('be,ae->ab', t2[ij], l2[ij])
                #print("Dvv_ij", Dvv)
                lDvv.append(Dvv)
                vv_energy += contract('ab, ab -> ', self.Local.Fvv[ij], Dvv)
        else:
            contract = self.contract
            vv_energy = 0
            for i in range(self.ccwfn.no):
                ii = i*self.ccwfn.no + i
                lDvv_1 = np.zeros((self.Local.dim[ii], self.Local.dim[ii]))
                lDvv_1 = contract('b,a->ab', t1[i], l1[i])
                vv_energy += contract('ab, ab -> ', self.Local.Fvv[ii], lDvv_1)
                for j in range(self.ccwfn.no):
                    ij = i*self.ccwfn.no + j 
                    lDvv_2 = np.zeros((self.Local.dim[ij], self.Local.dim[ij]))
                    lDvv_2 = contract('be,ae->ab', t2[ij], l2[ij] )
                    vv_energy += contract('ab, ab -> ', self.Local.Fvv[ij], lDvv_2)
                #Q = self.ccwfn.Local.Q
                #L = self.ccwfn.Local.L
                #QL_ii = Q[ii] @ L[ii]
                #QL_ij = Q[ij] @ L[ij]
                #S_ijii = QL_ij.T @ QL_ii
                #lDvv_1 = contract('b,a->ab', S_ijii@ t1[i], S_ijii@ l1[i])
                #lDvv_2 = contract('be,ae->ab', t2[ij], l2[ij] )
                #for m in range(self.ccwfn.no):
                #    mm = m*self.ccwfn.no + m
                #    Q = self.ccwfn.Local.Q
                #    L = self.ccwfn.Local.L
                #    QL_mm = Q[mm] @ L[mm]   
                #    QL_ij = Q[ij] @ L[ij]
                #    S_ijmm = (QL_ij).T @ (QL_mm)
                #    lDvv_1 += contract('b,a->ab', S_jmim@t1[m], S_jmim@l1[m])
                #    for n in range(self.ccwfn.no):
                #        mn = m * self.ccwfn.no + n
                #        Q_im = Q[mn] @ L[mn]
                #        Q_jm = Q[ij] @ L[ij]
                #        S_jmim = (Q_jm).T @ (Q_im)
                #        lDvv_2 += contract('be,ae->ab', S_jmim @ t2[mn], S_jmim @ l2[mn] )
                    if (ii == ij):
                        tDvv = lDvv_1 + lDvv_2
                    else:
                        tDvv = lDvv_2
                    lDvv.append(tDvv)
            print("Dvv energy", vv_energy)  
                #vv_energy += contract('ab, ab -> ', self.Local.Fvv[ij], tDvv)
                #print("Dvv_ij", ij, tDvv)
                    #lDvv.append(tDvv)
        return lDvv, vv_energy
                 

    def build_Dvo(self, l1):  # complete
        if isinstance(l1, torch.Tensor):
            return l1.T.clone()

        else:
            return l1.T.copy()

    def build_lDvo(self, t2, l2):  # begining - implementing the ccd-density
        pass 

    def build_Dov(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            if isinstance(t1, torch.Tensor):
                Dov = torch.zeros_like(t1)
            else:
                Dov = np.zeros_like(t1)
        else:
            if isinstance(t1, torch.Tensor):
                Dov = 2.0 * t1.clone()
            else:
                Dov = 2.0 * t1.copy()

            Dov += 2.0 * contract('me,imae->ia', l1, t2)
            Dov -= contract('me,miae->ia', l1, self.ccwfn.build_tau(t1, t2))
            tmp = contract('mnef,inef->mi', l2, t2)
            Dov -= contract('mi,ma->ia', tmp, t1)
            tmp = contract('mnef,mnaf->ea', l2, t2)
            Dov -= contract('ea,ie->ia', tmp, t1)
            for i in range(self.ccwfn.no):
                for j in range(self.ccwfn.no):
                    ij = i*self.ccwfn.no + j
                    QL = self.ccwfn.Local.Q[ij] @ self.ccwfn.Local.L[ij]
                    print("Dov", i, ij, Dov[i] @ QL)
            if self.ccwfn.model == 'CCSD(T)':
                Dov += self.ccwfn.Dov

            if isinstance(tmp, torch.Tensor):
                del tmp

        return Dov

    def build_lDov(self, t1, t2, l1, l2):  # beginning - implementing the ccd-density
        lDov = []
        if self.ccwfn.model == 'CCD':
            pass
        else:
            contract = self.contract
            QL = self.ccwfn.Local.QL
            for i in range(self.ccwfn.no):
                ii = i*self.ccwfn.no + i

                lDov_1 = 2 * t1[i].copy()

                for m in range(self.ccwfn.no):
                    im = i*self.ccwfn.no + m
                    mi = m*self.ccwfn.no + i 
                    mm = m*self.ccwfn.no + m
                    S_iiim = QL[ii].T @ QL[im]
                    S_immm = QL[im].T @ QL[mm]
                    S_mimm = QL[mi].T @ QL[mm]
                    S_iimi = QL[ii].T @ QL[mi]
                    S_iimm = QL[ii].T @ QL[mm]

                    tmp = S_iiim @ t2[im] @ S_immm
                    lDov_1 += 2 *contract('e,ae->a', l1[m], tmp)
                    
                    tmp = S_iimi @ t2[mi] @ S_mimm
                    lDov_1 -= contract('e, ae-> a', l1[m], tmp)
 
                    lDov_1 -= contract('e,a,e->a', l1[m] @ S_iimm.T, t1[m] @ S_iimm.T, t1[i]) 
   
                    Xmi = np.zeros((self.ccwfn.no, self.ccwfn.no))
                    Xea = np.zeros((self.Local.dim[ii], self.Local.dim[ii]))
                    for n in range(self.ccwfn.no):
                        _in = i*self.ccwfn.no + n
                        mn = m*self.ccwfn.no + n

                        S_mnin = QL[mn].T @ QL[_in]
                        S_iimn = QL[ii].T @ QL[mn]
                        tmp = S_mnin @ t2[_in] @ S_mnin.T 
                        Xmi[m,i] += contract('ef,ef->', l2[mn], tmp)

                        tmp = S_iimn @ l2[mn] 
                        tmp1 = S_iimn @ t2[mn]
                        Xea += contract('ef,af->ea', tmp, tmp1)

                    tmp = S_iimm @ t1[m] 
                    lDov_1 -= Xmi[m,i] * tmp 

                    lDov_1 -= contract('ea,e->a', Xea, t1[i]) 
                lDov.append(lDov_1)  
 
        return lDov

    def compute_local_energy(self, t1, t2, l1, l2):
        #lDoo = self.build_lDoo(t1,t2,l1, l2)
        #lDvv, vv_energy = self.build_lDvv(lDvv, t1,t2, l1, l2)
        #total_energy = vv_energy
        return total_energy



    # CC3 contributions to the one electron densities
    def build_cc3_Dov(self, o, v, no, nv, F, L, t1, t2, l1, l2, Wvvvo, Wovoo, Fov, Wvovv, Wooov):
        contract = self.contract  
        if isinstance(t1, torch.Tensor):
            Dov = torch.zeros_like(t1)   
            Zlmdi = torch.zeros_like(t2[:,:,:,:no])
        else:
            Dov = np.zeros_like(t1)    
            Zlmdi = np.zeros_like(t2[:,:,:,:no])    
        for i in range(no):
            for j in range(no):
                for k in range(no):                    
                    l3 = l3_ijk(i, j, k, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract)  
                    # Intermediate for Dov_2
                    Zlmdi[i,j] += contract('def,ife->di', l3, t2[k])
                    # Dov_1
                    t3 = t3c_ijk(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                    Dov[i] +=  contract('abc,bc->a', t3 - t3.swapaxes(0,1), l2[j,k])
        # Dov_2
        Dov -= contract('lmdi, lmda->ia', Zlmdi, t2)

        return Dov
                                    
    def build_cc3_Doo(self, o, v, no, nv, F, L, t2, l1, l2, Fov, Wvvvo, Wovoo, Wvovv, Wooov):
        contract = self.contract
        if isinstance(l1, torch.Tensor):
            Doo = torch.zeros_like(l1[:,:no])
        else:
            Doo = np.zeros_like(l1[:,:no])        
        for b in range(nv): 
            for c in range(nv):
                t3 = t3c_bc(o, v, b, c, t2, Wvvvo, Wovoo, F, contract)
                l3 = l3_bc(b, c, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract)
                Doo -= 0.5 * contract('lmia,lmja->ij', t3, l3)        

        return Doo        

    def build_cc3_Dvv(self, o, v, no, nv, F, L, t2, l1, l2, Fov, Wvvvo, Wovoo, Wvovv, Wooov):
        contract = self.contract
        if isinstance(l1, torch.Tensor):
            Dvv = torch.zeros_like(l1)
            Dvv = torch.nn.functional.pad(Dvv, (0,0,0,nv-no))
        else:
            Dvv = np.zeros_like(l1)
            Dvv = np.pad(Dvv, ((0,nv-no), (0,0)))
        for i in range(no):
            for j in range(no):
                for k in range(no):
                    t3 = t3c_ijk(o, v, i, j, k, t2, Wvvvo, Wovoo, F, contract)
                    l3 = l3_ijk(i, j, k, o, v, L, l1, l2, Fov, Wvovv, Wooov, F, contract)
                    Dvv += 0.5 * contract('bdc,adc->ab', t3, l3)

        return Dvv

    def build_Doooo(self, t1, t2, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            return contract('ijef,klef->ijkl', t2, l2)
        elif self.ccwfn.model == 'CC2':
            return contract('jf, klif->ijkl', t1, contract('ie, klef->klif', t1, l2))
        else:
            return contract('ijef,klef->ijkl', self.ccwfn.build_tau(t1, t2), l2)

    def build_Dvvvv(self, t1, t2, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            return contract('mnab,mncd->abcd', t2, l2)
        elif self.ccwfn.model == 'CC2':
            return contract('nb,ancd->abcd', t1, contract('ma,mncd->ancd', t1, l2))
        else:
            return contract('mnab,mncd->abcd', self.ccwfn.build_tau(t1, t2), l2)

    def build_Dooov(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            no = self.ccwfn.no
            nv = self.ccwfn.nv
            if isinstance(t1, torch.Tensor):
                if self.ccwfn.precision == 'DP':
                    Dooov = torch.zeros((no,no,no,nv), dtype=torch.complex128, device=self.ccwfn.device1)
                elif self.ccwfn.precision == 'SP':                    
                    Dooov = torch.zeros((no,no,no,nv), dtype=torch.complex64, device=self.ccwfn.device1)
            else:
                if self.ccwfn.precision == 'DP':
                    Dooov = np.zeros((no,no,no,nv))
                elif self.ccwfn.precision == 'SP':
                    Dooov = np.zeros((no,no,no.nv), dtype=np.complex64)
        else:
            tmp = 2.0 * self.ccwfn.build_tau(t1, t2) - self.ccwfn.build_tau(t1, t2).swapaxes(2, 3)
            Dooov = -1.0 * contract('ke,ijea->ijka', l1, tmp)
            Dooov -= contract('ie,jkae->ijka', t1, l2)

            if self.ccwfn.model != 'CC2':

                Goo = self.cclambda.build_Goo(t2, l2)
                Dooov -= 2.0 * contract('ik,ja->ijka', Goo, t1)
                Dooov += contract('jk,ia->ijka', Goo, t1)
                tmp = contract('jmaf,kmef->jake', t2, l2)
                Dooov -= 2.0 * contract('jake,ie->ijka', tmp, t1)
                Dooov += contract('iake,je->ijka', tmp, t1)

                tmp = contract('ijef,kmef->ijkm', t2, l2)
                Dooov += contract('ijkm,ma->ijka', tmp, t1)
                tmp = contract('mjaf,kmef->jake', t2, l2)
                Dooov += contract('jake,ie->ijka', tmp, t1)
                tmp = contract('imea,kmef->iakf', t2, l2)
                Dooov += contract('iakf,jf->ijka', tmp, t1)

            tmp = contract('kmef,jf->kmej', l2, t1)
            tmp = contract('kmej,ie->kmij', tmp, t1)
            Dooov += contract('kmij,ma->ijka', tmp, t1)

            # (T) contributions to twopdm computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Dooov += self.ccwfn.Gooov
            
            if isinstance(tmp, torch.Tensor):
                del tmp
                del Goo

        return Dooov


    def build_Dvvvo(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            no = self.ccwfn.no
            nv = self.ccwfn.nv
            if isinstance(t1, torch.Tensor):
                if self.ccwfn.precision == 'DP':
                    Dvvvo = torch.zeros((nv,nv,nv,no), dtype=torch.complex128, device=self.ccwfn.device1)
                if self.ccwfn.precision == 'SP':
                    Dvvvo = torch.zeros((nv,nv,nv,no), dtype=torch.complex64, device=self.ccwfn.device1)
            else:
                if self.ccwfn.precision == 'DP':
                    Dvvvo = np.zeros((nv,nv,nv,no))
                if self.ccwfn.precision == 'SP':
                    Dvvvo = np.zeros((nv,nv,nv,no), dtype=np.complex64)
        else: 
            tmp = 2.0 * self.ccwfn.build_tau(t1, t2) - self.ccwfn.build_tau(t1, t2).swapaxes(2, 3)
            Dvvvo = contract('mc,miab->abci', l1, tmp)
            Dvvvo += contract('ma,imbc->abci', t1, l2)

            if self.ccwfn.model != 'CC2':
                
                Gvv = self.cclambda.build_Gvv(t2, l2)
                Dvvvo -= 2.0 * contract('ca,ib->abci', Gvv, t1)
                Dvvvo += contract('cb,ia->abci', Gvv, t1)
                tmp = contract('imbe,nmce->ibnc', t2, l2)
                Dvvvo += 2.0 * contract('ibnc,na->abci', tmp, t1)
                Dvvvo -= contract('ianc,nb->abci', tmp, t1)

                tmp = contract('nmab,nmce->abce', t2, l2)
                Dvvvo -= contract('abce,ie->abci', tmp, t1)
                tmp = contract('niae,nmce->iamc', t2, l2)
                Dvvvo -= contract('iamc,mb->abci', tmp, t1)
                tmp = contract('mibe,nmce->ibnc', t2, l2)
                Dvvvo -= contract('ibnc,na->abci', tmp, t1)

            tmp = contract('nmce,ie->nmci', l2, t1)
            tmp = contract('nmci,na->amci', tmp, t1)
            Dvvvo -= contract('amci,mb->abci', tmp, t1)
 
            # (T) contributions to twopdm computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Dvvvo += self.ccwfn.Gvvvo

            if isinstance(tmp, torch.Tensor):
                del tmp
                del Gvv

        return Dvvvo


    def build_Dovov(self, t1, t2, l1, l2):  # complete
        contract = self.contract
        if self.ccwfn.model == 'CCD':
            Dovov = -contract('mibe,jmea->iajb', t2, l2)
            Dovov -= contract('imbe,mjea->iajb', t2, l2)
        else:
            Dovov = -1.0 * contract('ia,jb->iajb', t1, l1)
            if self.ccwfn.model == 'CC2':
                Dovov -= contract('mb,jmia->iajb', t1, contract('ie,jmea->jmia', t1, l2))
            else:
                Dovov -= contract('mibe,jmea->iajb', self.ccwfn.build_tau(t1, t2), l2)
                Dovov -= contract('imbe,mjea->iajb', t2, l2)
        return Dovov


    def build_Doovv(self, t1, t2, l1, l2):
        contract = self.contract
        tau = self.ccwfn.build_tau(t1, t2)
        tau_spinad = 2.0 * tau - tau.swapaxes(2,3)

        if self.ccwfn.model == 'CCD':
            Doovv = 2.0 * tau_spinad + l2

            Doovv += 4.0 * contract('imae,mjeb->ijab', t2, l2)
            Doovv -= 2.0 * contract('mjbe,imae->ijab', tau, l2)

            tmp_oooo = contract('ijef,mnef->ijmn', t2, l2)
            Doovv += contract('ijmn,mnab->ijab', tmp_oooo, t2)
            tmp1 = contract('njbf,mnef->jbme', t2, l2)
            Doovv += contract('jbme,miae->ijab', tmp1, t2)
            tmp1 = contract('imfb,mnef->ibne', t2, l2)
            Doovv += contract('ibne,njae->ijab', tmp1, t2)
            Gvv = self.cclambda.build_Gvv(t2, l2)
            Doovv += 4.0 * contract('eb,ijae->ijab', Gvv, tau)
            Doovv -= 2.0 * contract('ea,ijbe->ijab', Gvv, tau)
            Goo = self.cclambda.build_Goo(t2, l2)
            Doovv -= 4.0 * contract('jm,imab->ijab', Goo, tau)  # use tau_spinad?
            Doovv += 2.0 * contract('jm,imba->ijab', Goo, tau)
            tmp1 = contract('inaf,mnef->iame', t2, l2)
            Doovv -= 4.0 * contract('iame,mjbe->ijab', tmp1, tau)
            Doovv += 2.0 * contract('ibme,mjae->ijab', tmp1, tau)
            Doovv += 4.0 * contract('jbme,imae->ijab', tmp1, t2)
            Doovv -= 2.0 * contract('jame,imbe->ijab', tmp1, t2)
            
            if isinstance(tmp1, torch.Tensor):
                del tmp_oooo, tmp1, Gvv, Goo

        else:
            Doovv = 4.0 * contract('ia,jb->ijab', t1, l1)
            Doovv += 2.0 * tau_spinad
            Doovv += l2

            tmp1 = 2.0 * t2 - t2.swapaxes(2,3)
            tmp2 = 2.0 * contract('me,jmbe->jb', l1, tmp1)
            Doovv += 2.0 * contract('jb,ia->ijab', tmp2, t1)
            Doovv -= contract('ja,ib->ijab', tmp2, t1)
            tmp2 = 2.0 * contract('ijeb,me->ijmb', tmp1, l1)
            Doovv -= contract('ijmb,ma->ijab', tmp2, t1)
            tmp2 = 2.0 * contract('jmba,me->jeba', tau_spinad, l1)
            Doovv -= contract('jeba,ie->ijab', tmp2, t1)

            if self.ccwfn.model == 'CC2':
                Doovv -= 2.0 * contract('mb,imaj->ijab', t1, contract('je,imae->imaj', t1, l2))
            else:
                Doovv += 4.0 * contract('imae,mjeb->ijab', t2, l2)
                Doovv -= 2.0 * contract('mjbe,imae->ijab', tau, l2)

                tmp_oooo = contract('ijef,mnef->ijmn', t2, l2)
                Doovv += contract('ijmn,mnab->ijab', tmp_oooo, t2)
                tmp1 = contract('njbf,mnef->jbme', t2, l2)
                Doovv += contract('jbme,miae->ijab', tmp1, t2)
                tmp1 = contract('imfb,mnef->ibne', t2, l2)
                Doovv += contract('ibne,njae->ijab', tmp1, t2)
                Gvv = self.cclambda.build_Gvv(t2, l2)
                Doovv += 4.0 * contract('eb,ijae->ijab', Gvv, tau)
                Doovv -= 2.0 * contract('ea,ijbe->ijab', Gvv, tau)
                Goo = self.cclambda.build_Goo(t2, l2)
                Doovv -= 4.0 * contract('jm,imab->ijab', Goo, tau)  # use tau_spinad?
                Doovv += 2.0 * contract('jm,imba->ijab', Goo, tau)
                tmp1 = contract('inaf,mnef->iame', t2, l2)
                Doovv -= 4.0 * contract('iame,mjbe->ijab', tmp1, tau)
                Doovv += 2.0 * contract('ibme,mjae->ijab', tmp1, tau)
                Doovv += 4.0 * contract('jbme,imae->ijab', tmp1, t2)
                Doovv -= 2.0 * contract('jame,imbe->ijab', tmp1, t2)

                # this can definitely be optimized better
                tmp = contract('nb,ijmn->ijmb', t1, tmp_oooo)
                Doovv += contract('ma,ijmb->ijab', t1, tmp)
                tmp = contract('ie,mnef->mnif', t1, l2)
                tmp = contract('jf,mnif->mnij', t1, tmp)
                Doovv += contract('mnij,mnab->ijab', tmp, t2)
                tmp = contract('ie,mnef->mnif', t1, l2)
                tmp = contract('mnif,njbf->mijb', tmp, t2)
                Doovv += contract('ma,mijb->ijab', t1, tmp)
                tmp = contract('jf,mnef->mnej', t1, l2)
                tmp = contract('mnej,miae->njia', tmp, t2)
                Doovv += contract('nb,njia->ijab', t1, tmp)
                tmp = contract('je,mnef->mnjf', t1, l2)
                tmp = contract('mnjf,imfb->njib', tmp, t2)
                Doovv += contract('na,njib->ijab', t1, tmp)
                tmp = contract('if,mnef->mnei', t1, l2)
                tmp = contract('mnei,njae->mija', tmp, t2)
                Doovv += contract('mb,mija->ijab', t1, tmp)

            tmp = contract('jf,mnef->mnej', t1, l2)
            tmp = contract('ie,mnej->mnij', t1, tmp)
            tmp = contract('nb,mnij->mbij', t1, tmp)
            Doovv += contract('ma,mbij->ijab', t1, tmp)
 
            # (T) contributions to twopdm computed in ccwfn.t3_density()
            if self.ccwfn.model == 'CCSD(T)':
                Doovv += self.ccwfn.Goovv

            if isinstance(tmp, torch.Tensor):
                del tmp, tmp1, tmp2, Goo, Gvv

        return Doovv

    # T1-transformed dipole integrals needed in CC3
    def build_Moo(self, no, nv, ints, t1):
        contract = self.contract
        if isinstance(t1, torch.Tensor):
            Moo = ints[:no,:no].clone()
        else:
            Moo = ints[:no,:no].copy()
        Moo = Moo + contract('ma,ia->mi', ints[:no,-nv:], t1)

        return Moo

    def build_Mvv(self, no, nv, ints, t1):
        contract = self.contract
        if isinstance(t1, torch.Tensor):
            Mvv = ints[-nv:,-nv:].clone()
        else:
            Mvv = ints[-nv:,-nv:].copy()
        Mvv = Mvv - contract('ie,ia->ae', ints[:no,-nv:], t1)

        return Mvv
