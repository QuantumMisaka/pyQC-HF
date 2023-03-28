import numpy as np
from atom import Atom
from scipy.special import erf
from numpy.linalg import norm
from basis_db import STO3G_Table

class STO_3G():
    def __init__(self):
        # only support H and He now
        pass

    def norm_factor(self, a: np.float64):
        '''
        normalizer ((2/np.pi) *a)**(0.75) for GTO
        
        Inputs:
            a: np.float64, alpha in GTO
        '''
        # normolizer implemented in GTO at the first time
        return ((2/np.pi) *a)**(0.75)
    
    def F0(self,t:np.float64) -> np.float64:
        '''special correlation function described by erf(z)
        
        Input:
            t: np.float64: variables
        '''
        if t < 1.0E-5:
            # zero-point treatment: use expansion
            return 1.0
        else:
            return 0.5 * np.sqrt(np.pi/t) * erf(np.sqrt(t))


    def ao_s(self, atom:Atom, r:np.ndarray):
        '''sto3g_basis_set, only support s
        
        Inputs:
            ele: str
                elements type
            atom: Atom
                attritube of nuclei
            r: np.ndarray
                spacial vector of elec
        '''
        a_list, d_list, R = self.ao_s_coef(atom)
        def g_s(a,r,R):
            dist = norm(R, r)
            cn = self.norm_factor(a)
            return cn * np.exp(-a * np.power(dist, 2))
        return np.sum([d * g_s(a,r,R) for d,a in zip(d_list, a_list)])
    
    def gto_product(self, a1:np.float64, a2:np.float64, R1:np.ndarray, R2:np.ndarray, r:np.ndarray):
        '''GTO product solution
        '''
        Rp = (a1*R1 + a2*R2) / (a1 + a2)
        K = np.exp(-((a1*a2)/(a1+a2)) * norm(R1 - R2)**2)
        p = a1 + a2
        cn1, cn2 = self.norm_factor(a1), self.norm_factor(a2)
        # r maybe a set of array, treated as row vector
        if r.ndim == 2:
            # for vector matrix
            r_ep = norm(r - Rp, axis=1)
        else:
            r_ep = norm(r - Rp)
        return cn1 * cn2 * K * np.exp(-p * r_ep**2)


    def ao_s_product(self, atom1:Atom, atom2:Atom, r:np.ndarray):
        '''sto3g_basis_set product

        Inputs:
            atom1, atom2: Atom
                AO basis reference
            r: np.ndarray
                electron position
        '''
        a1_list, d1_list, R1 = self.ao_s_coef(atom1)
        a2_list, d2_list, R2 = self.ao_s_coef(atom2)
        basis_prod = 0
        for d1,a1 in zip(d1_list, a1_list):
            for d2,a2 in zip(d2_list, a2_list):
                gto_prod = self.gto_product(a1, a2, R1, R2, r)
                basis_prod += d1*d2 * gto_prod
        return basis_prod
    

    def ao_s_coef(self, atom: Atom) -> list:
        '''getting coefficient of sto-3g atomic orbital by atom infomation
        
        Inputs:
            atom: Atom object for AO reference
        '''
        a_list = STO3G_Table[atom.ele][0]
        d_list = STO3G_Table[atom.ele][1]
        R = atom.pos
        return a_list,d_list,R


    def ovlp_1g(self,a1:np.float64,a2:np.float64,R1:np.ndarray,R2:np.ndarray) -> np.float64:
        '''calculated <A|B> overlap matrix element in GTO'''
        cn_1 = self.norm_factor(a1)
        cn_2 = self.norm_factor(a2)
        return cn_1 * cn_2 * np.power(np.pi / (a1 + a2), 3/2) * np.exp(-np.divide(a1*a2, a1+a2) * norm(R1 - R2)**2)


    def kine_1g(self,a1:np.float64,a2:np.float64,R1:np.ndarray,R2:np.ndarray) -> np.float64:
        '''calculated <A|-0.5\nabla|B> kinetic matrix element in GTO'''
        # normalizer embedded in ovlp_1g
        return np.divide(a1*a2, a1+a2) * (3 - np.divide(2*a1*a2, a1+a2) 
        * norm(R1 - R2)**2) * self.ovlp_1g(a1,a2,R1,R2)


    def intg_ne_1g(self, a1:np.float64, a2:np.float64, R1:np.ndarray, R2:np.ndarray, Rc: np.ndarray, Zc:np.float64) -> np.float64:
        '''calculated <A|-Z_n/r_ne|B> nuclei-elec matrix element in GTO'''
        Rp = (a1*R1 + a2*R2) / (a1 + a2)
        r_ab = norm(R1 - R2)
        r_pc = norm(Rp - Rc)
        cn_1 = self.norm_factor(a1)
        cn_2 = self.norm_factor(a2)
        return cn_1 * cn_2 * 2*( - np.pi/(a1+a2)) * Zc * np.exp(
            -(a1*a2/(a1+a2)) * r_ab**2) * self.F0((a1+a2) * r_pc**2)

    
    def intg_ee_1g(self, a1:np.float64, a2:np.float64, a3:np.float64, a4:np.float64, R1:np.ndarray, R2:np.ndarray, R3:np.ndarray, R4:np.ndarray) -> np.float64:
        '''calculated <AB|-1/r_ee|CD> electronic matrix element in GTO'''
        Rp = (a1*R1 + a2*R2) / (a1 + a2)
        Rq = (a3*R3 + a4*R4) / (a3 + a4)
        cn_1 = self.norm_factor(a1)
        cn_2 = self.norm_factor(a2)
        cn_3 = self.norm_factor(a3)
        cn_4 = self.norm_factor(a4)
        r_ab = norm(R1 - R2)
        r_cd = norm(R3 - R4)
        r_pq = norm(Rp - Rq)
        item1 = np.divide(2*np.pi**2.5, (a1+a2)*(a3+a4)*(a1+a2+a3+a4)**0.5)
        item2 = np.exp(-(a1*a2/(a1+a2) * r_ab**2) -(a3*a4/(a3+a4) * r_cd**2))
        item3 = self.F0( ((a1+a2)*(a3+a4)/(a1+a2+a3+a4)) * r_pq**2)
        return cn_1 * cn_2 * cn_3 * cn_4 * item1 * item2 * item3
    

    # def hcore_ele(self, atom1: Atom, atom2: Atom, r) -> np.float64:
    #     '''
    #     hcore integrals (matrix element) calculation, 
    #     including kinetic energy integrals
    #     and nuclear attraction integrals.
    #     '''
    #     hcore_12 = self.kinetic_ele(atom1, atom2)
    #     hcore_12 += self.overlap_ele(atom1, atom2)
    #     hcore_12 += self.intg_ne_ele(atom1, atom2, atom1)
    #     hcore_12 += self.intg_ne_ele(atom1, atom2, atom2)
    #     return hcore_12
    

    def kinetic_ele(self, atom1: Atom, atom2: Atom) -> np.float64:
        '''
        kinetic ingegral matrix element calculation

        Input:
            atom, atom2: Atom object
                atom for basis
        '''
        a1_list, d1_list, R1 = self.ao_s_coef(atom1)
        a2_list, d2_list, R2 = self.ao_s_coef(atom2)
        t12 = 0
        for d1, a1 in zip(d1_list, a1_list):
            for d2, a2 in zip(d2_list, a2_list):
                # print(a1, a2, d1, d2, R1, R2)
                t12 += d1*d2 * self.kine_1g(a1, a2, R1, R2)
        return t12
    

    def overlap_ele(self, atom1: Atom, atom2: Atom) -> np.float64:
        '''
        overlap integral matrix element calculation
        
        Input:
            atom, atom2: Atom object
                atom for AO basis
        '''
        a1_list, d1_list, R1 = self.ao_s_coef(atom1)
        a2_list, d2_list, R2 = self.ao_s_coef(atom2)

        # 3*3 = 9 items, not 6 items
        # s12 = np.sum([d1*d2 * self.ovlp_1g(a1,a2,R1,R2) 
        #            for d1,a1,d2,a2 in zip(d1_list, a1_list, d2_list, a2_list)])
        s12 = 0
        for d1,a1 in zip(d1_list, a1_list):
            for d2,a2 in zip(d2_list, a2_list):
                ovlp = self.ovlp_1g(a1, a2, R1, R2)
                s12 += d1*d2 * ovlp
        return s12



    def intg_ne_ele(self, atom1: Atom, atom2: Atom, atom_z: Atom) -> np.float64:
        '''intg_ne matrix element for nuclei-elec interact
        
        Input:
            atom1, atom2: Atom object
                atom for AO basis
            atomz: Atom object
                atom for nuclei
        '''
        a1_list, d1_list, R1 = self.ao_s_coef(atom1)
        a2_list, d2_list, R2 = self.ao_s_coef(atom2)
        Zc, Rc = atom_z.charge, atom_z.pos
        intg_ne_12 = 0
        for d1, a1 in zip(d1_list, a1_list):
            for d2, a2 in zip(d2_list, a2_list):
                # print(a1, a2, d1, d2, R1, R2)
                intg_ne_12 += d1*d2 * self.intg_ne_1g(a1, a2, R1, R2, Rc, Zc)
        return intg_ne_12
    

    def intg_ee_ele(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom) -> np.float64:
        '''intg_ee matrix element for ERI used
        
        Input:
            atom1, atom2, atom3, atom4: Atom object
                atom for AO basis
        '''
        a1_list, d1_list, R1 = self.ao_s_coef(atom1)
        a2_list, d2_list, R2 = self.ao_s_coef(atom2)
        a3_list, d3_list, R3 = self.ao_s_coef(atom3)
        a4_list, d4_list, R4 = self.ao_s_coef(atom4)
        intg_ee_1234 = 0
        for d1, a1 in zip(d1_list, a1_list):
            for d2, a2 in zip(d2_list, a2_list):
                for d3, a3 in zip(d3_list, a3_list):
                    for d4, a4 in zip(d4_list, a4_list):
                        intg_ee_1234 += d1*d2*d3*d4 * self.intg_ee_1g(a1, a2, a3, a4, R1, R2, R3, R4)
        return intg_ee_1234
    
    # def S_matrix(self) -> np.ndarray:
    #     return
    
    # def T_matrix(self) -> np.ndarray:
    #     return

    # def intg_ne_matrix(self) -> np.ndarray:
    #     return
    
    # def intg_ee_matrix(self) -> np.ndarray:
    #     return
    
    # def H_matrix(self) -> np.ndarray:
    #     return

    # def Fock_matrix(self) -> np.ndarray:
    #     return

    

    
    

    





    
