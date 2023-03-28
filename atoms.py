from typing import Tuple,List
import numpy as np
#import scipy
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import time


# Use eigh to diagonalize matrices
from scipy.linalg import eigh

#import PeriodicTable as PT
from basis_set import STO_3G
from atom import Atom

class Atoms():
    '''
    Atoms object contains all function to describe and solve this system.
    (support solving by RHF: H, He)

    Inputs:
        atoms: str
            string of atoms constructing structure
        charge: int
            net charge of structure, default 0
    Attirtubes:
        atoms: List[Atom]
            list of atom objects
        nelec: int
            number of electrons in whole structure
        Others: Waiting for notes
    '''
    def __init__(self, atoms: str, charge=0 ):
        # Introduction
        intro_msg = " ---- Isolated Atoms System Treating Package  ---- \n"
        intro_msg += "---- by Hartree-Fock Method and STO-3G Basis ---- \n"
        intro_msg += "---- Authors: JamesMisaka & i.i.d ---- \n"
        intro_msg += "---- Version 0.1, Last Update on 2023-03-23 ---- \n"
        print(intro_msg)

        self.atoms = self.read_from_str(atoms)
        self.nelec = - charge # +1 charge means lose an elec
        self.num_ao = 0
        self.charge = charge
        for atom in self.atoms:
            atom: Atom
            self.nelec += atom.charge
            self.num_ao += atom.num_ao
        # only support rhf now
        self.nelec_alpha = int(np.ceil(self.nelec / 2))
        self.nelec_beta = int(np.floor(self.nelec / 2))
        if self.nelec_alpha != self.nelec_beta:
            print("Notices: This code only support solving closed-shell systems. ")
            self.RHF = False
        else:
            self.RHF = True
        self.energy = 0
        #self.wf = None
        self.basis = None
        self.V_nuc = 0
        self.dm = []
        self.mo_info = None
        self.spin_multi = (self.nelec_alpha - self.nelec_beta)*2+1
        self.atom_msg()

    def atom_msg(self):
        '''print atom infomation message'''
        atoms_msg = f"charge: {self.charge}, spin_multiplicity: {self.spin_multi}\n"
        for atom in self.atoms:
            atoms_msg += f"{atom.ele:^4} {atom.pos[0]:^8.6f} {atom.pos[1]:^8.6f} {atom.pos[2]:^8.6f}\n"
        print(atoms_msg)

    def read_from_str(self, atoms:str) -> List[Atom]:
        '''
        Read atoms infomation from xyz infomation by string

        Inputs: 
            atoms: str
                string of atoms constructing structure
        Returns:
            atoms_list: list of atom objects
        '''
        print(" ----- Reading Atoms System -----")
        atoms_list = []
        atoms = atoms.split('\n')
        for atom in atoms:
            atom = atom.strip()
            if atom:
                atom = atom.split()
                ele = atom[0]
                coord = np.array(atom[1:4], dtype=float)
                atom_obj = Atom(ele, coord)
                atoms_list.append(atom_obj)
        return atoms_list

    
    def do_HF(self, max_iter=120, tol=1E-6, is_nuc=True):
        '''
        do HF calculation, only support H and He atomtype and closed-shell system

        Inputs: 
            max_iter: int = 120
                max iteration number
            tol: float = 1E-6
                tolerance of electronic energy and density matrix L2 norm
            is_nuc: bool = True
                set True to include nuclei-nuclei repelsion energy in total energy
        Returns
            energy: np.float64
                total energy of atom system
            And will set energy, mo_info, dm (for density matrix) attributes for Atoms objects
        '''
        start_time = time.perf_counter()
        energy = 0
        mo_info = None
        if self.RHF == True:
            energy, mo_info, dens_matrix = self.solve_rhf(
                max_iter=max_iter, tol=tol, is_nuc=is_nuc)
            self.energy = energy
            self.mo_info = mo_info
            self.dm = dens_matrix
        else:
            raise NotImplementedError("Solving only support closed-shell system")
        end_time = time.perf_counter()
        time_used = end_time - start_time
        print(f"Time Consumed of Hartree-Fock Calculation: {time_used:6.2f} sec")
        return energy


    def set_sto3g_elec_matrices(self, override=False):
        '''setting matrices by sto-3g basis expansion'''
        if self.basis and not override:
            print("Basis is Set! Specified override option to set again")
            return 
        else:
            self.basis = "sto-3g"

        sto3g = STO_3G()
        nao = self.num_ao
        S = np.zeros((nao, nao)) # overlap matrix
        T = np.zeros((nao, nao)) # kinetic matrix
        V_ne = np.zeros((nao, nao)) # nuclei_elec matrix
        eri_m = np.zeros((nao, nao, nao, nao)) # ERI matrix contain all two-elec integral

        for i,atom_i in enumerate(self.atoms):
            for j,atom_j in enumerate(self.atoms):
                # overlap
                if i <= j:
                    S[i][j] = sto3g.overlap_ele(atom_i, atom_j)
                    T[i][j] = sto3g.kinetic_ele(atom_i, atom_j)
                    for atom_z in self.atoms:
                        V_ne[i][j] += sto3g.intg_ne_ele(
                            atom_i, atom_j, atom_z)
                else:
                    S[i][j] = S[j][i]
                    T[i][j] = T[j][i]
                    V_ne[i][j] = V_ne[j][i]
                for k, atom_k in enumerate(self.atoms):
                    for l, atom_l in enumerate(self.atoms):
                        # using symmetrices to simplify
                        if i>j and k==l:
                            eri_m[i][j][k][l] = eri_m[j][i][k][l]
                        elif i>k and j==l:
                            eri_m[i][j][k][l] = eri_m[k][i][j][l]
                        elif i>l and j==k:
                            eri_m[i][j][k][l] = eri_m[l][j][k][i]
                        elif j>k and i==l:
                            eri_m[i][j][k][l] = eri_m[i][k][j][l]
                        elif j>l and i==k:
                            eri_m[i][j][k][l] = eri_m[i][l][k][j]
                        elif k>l and i==j:
                            eri_m[i][j][k][l] = eri_m[i][j][l][k]
                        else:
                            eri_m[i][j][k][l] = sto3g.intg_ee_ele(atom_i, atom_j, atom_k, atom_l)
        self.S = S
        self.T = T
        self.V_ne = V_ne
        self.eri_m = eri_m
        self.Hcore = T + V_ne
        return
    
    
    def V_nn(self) -> float:
        '''give nuclei-nuclei interaction energy'''
        energy_nuc = 0
        for i,atom_i in enumerate(self.atoms):
            for j,atom_j in enumerate(self.atoms):
                if i >= j:
                    continue
                else:
                    atom_i: Atom; atom_j: Atom
                    zi, zj = atom_i.charge, atom_j.charge
                    Ri, Rj = atom_i.pos, atom_j.pos
                    energy_nuc += np.divide(zi*zj, np.linalg.norm(Ri-Rj))
        self.V_nuc = energy_nuc
        return energy_nuc 
    

    def solve_rhf(self, max_iter=120, tol=1E-6, is_nuc=False) -> Tuple[np.float64, list, np.ndarray]:
        '''
        Solve the Hartree-Fock with SCF iterations.
        Reference: Szabo and Ostlund, 3.4.6. (p. 146, start from step 2)

        The SCF procedure is:
            - Obtain a guess at the density matrix.
            - Calculate the exchange and coulomb matrices from the density matrix
            and the two-electron repulsion integrals.
            - Add exchange and coulomb matrices to the core-Hamiltonian to obtain the
            Fock matrix.
            - Diagonalize the Fock matrix and Overlap matrix to solve generalized eigenvalue problem.
            - Select the occupied orbitals and calculate the new density matrix.
            - Compute the energy
            - Compute the errors and check convergence
                - If converged, return the energy
                - If not converged, return to second step with new density matrix

        Inputs:
            max_iter : int = 120
                The maximum number of SCF iterations.
            tol : float = 1e-6
                The convergence tolerance.
            Other Inputs are already treated in Atoms objects

        Returns:
            energy_all : np.float64
                Total energy of Atoms system
            mo_inf: list
                Molecular orbital infomation list, 
                format is [energy : np.floa64, nelec_occ: int , cofficients : np.ndarray]
            dm_cur: np.ndarray
                Density matrix of Atoms system
        '''
        self.set_sto3g_elec_matrices()
        nelec_a, nelec_b = int(self.nelec_alpha), int(self.nelec_beta)
        assert nelec_a == nelec_b, "rhf method only supports closed-shell systems."

        hcore = self.Hcore
        S = self.S
        eri = self.eri_m 

        iter_scf = 0
        is_converged = False
        is_max_iter = False

        energy_err = 1.0
        dm_err = 1.0

        energy_rhf = None
        energy_old = None
        energy_cur = None
        energy_nuc = 0.0
        energy_all = 0.0


        nmo = self.num_ao
        nocc = int((nelec_a + nelec_b) // 2)
        mo_occ = np.zeros(nmo, dtype=int)
        mo_occ[:nocc] = 2
        occ_list = np.where(mo_occ > 0)[0]
        # print(mo_occ)
        # print(occ_list)

        # Diagonalize the core Hamiltonian, 
        # Get the initial guess for density matrix
        energy_mo, coeff_mo = eigh(hcore, S)
        coeff_occ = coeff_mo[:, occ_list]
        dm_old = np.dot(coeff_occ, coeff_occ.T) * 2.0
        dm_cur = None
        fock = None

        # for print
        print(" ----- Using Hartree-Fock Method to Solve Atoms System -----")
        # for print
        print(" ----- Hartree-Fock SCF Start Running !! -----")
        headline = f"--- SCF iteration Num,  {'Energy_e (Ha)':^12},  {'Err_E (Ha)':^6}, {'Err_dm':^6}"
        print(headline)

        # iteration
        while not is_converged and not is_max_iter:
            # Compute the Fock matrix using Einstein sum implement in numpy
            iter_scf += 1

            coul = np.einsum("pqrs,rs->pq", eri , dm_old)
            exch = - np.einsum("prqs,rs->pq", eri, dm_old) / 2.0
            fock = hcore + coul + exch

            # Diagonalize the Fock matrix
            # return eigenvalue and eigenvector, which sorted
            energy_mo, coeff_mo = eigh(fock, S)

            # Compute the new density matrix
            coeff_occ = coeff_mo[:, occ_list]
            dm_cur = np.dot(coeff_occ, coeff_occ.T) * 2.0

            # Compute the energy
            energy_cur = 0.5 * np.einsum("pq,pq->", hcore + fock, dm_cur)
            energy_rhf = energy_cur

            # Compute the errors
            if energy_old is not None:
                dm_err = np.linalg.norm(dm_cur - dm_old)
                energy_err = abs(energy_cur - energy_old)
                print(
                    f"--- SCF iteration {iter_scf:3d}, {energy_rhf: 12.8f}, {energy_err: 6.4e}, {dm_err: 6.4e}")
            else:
                print(
                    f"--- SCF iteration {iter_scf:3d}, {energy_rhf: 12.8f},")

            # Updating
            dm_old = dm_cur # update density matrix
            energy_old = energy_cur # update rhf energy

            # Check convergence
            is_max_iter = iter_scf >= max_iter
            is_converged = energy_err < tol and dm_err < tol

        if is_converged:
            print(f"SCF converged in {iter_scf} iterations., tolerance={tol}")
            if is_nuc == True:
                print("is_nuc = True, Nuclei-nuclei Coulumb reposion energy will be included in total energy")
                energy_nuc = self.V_nn()
                print(f"Energy for N-N interaction: {energy_nuc:12.8f} (Ha)")
                energy_all = energy_nuc + energy_rhf
            else:
                energy_all = energy_rhf
        else:
            if energy_rhf is not None:
                print(f"SCF did not converge in {max_iter} iterations.")
                return 
            else:
                energy_rhf = 0.0
                print("SCF is not running.")
                return

        print(f"Total Energy: {energy_all:12.8} Ha")
        mo_info = []
        for energy_mo, coeff_mo, nelec_occ in zip(energy_mo, coeff_mo, mo_occ):
            mo_info.append((energy_mo, coeff_mo, nelec_occ))
        return energy_all, mo_info, dm_cur


    def MO_infomation(self):
        '''
            print moleculer orbital infomation
        '''
        if not bool(self.mo_info):
            raise ValueError("MO infomation NOT Found! SCF do not calculate or not converged!")
        else:
            print(" --- MO infomation --- ")
            for i,mo_info in enumerate(self.mo_info):
                energy_mo, c_mo, nelec_occ = mo_info
                mo_msg = f"MO Level {i}: Energy: {energy_mo:12.8f} (Ha), "
                mo_msg += f"Elec Occ Num: {nelec_occ} \n, "
                mo_msg += f"Coefficients: {c_mo}, "
                print(mo_msg)


    def f_elec_density(self, r:np.ndarray):
        '''
        get electron density from atom.dm and r

        Inputs:
            r: position of electron, by np.array([x,y,z])

        '''
        elec_dens = 0
        if self.energy == 0:
            raise ValueError("SCF not calculated !")
        else:
            if "sto" and "3g" in self.basis:
                basis = STO_3G()
            else:
                raise NotImplementedError("basis not supported")
            for i,atom_i in enumerate(self.atoms):
                for j,atom_j in enumerate(self.atoms):
                    elec_dens += self.dm[i][j] * basis.ao_s_product(atom_i, atom_j, r)
            return elec_dens



    def elec_density_mesh(self, X,Y,Z) -> np.ndarray:
        '''
        generate elec density by density matrix
        '''
        print("calcualting electronic distribution")
        start_time = time.perf_counter()
        l_x, l_y, l_z = np.shape(X)
        r_flat = np.array(list(zip(X.flatten(),Y.flatten(),Z.flatten()))) # merge xx,yy,zz to [(x,y,z)]
        elec_dens_mesh = self.f_elec_density(r_flat)
        elec_dens_mesh = elec_dens_mesh.reshape((l_x, l_y, l_z))
        end_time = time.perf_counter()
        time_used = end_time - start_time
        print(f"electronic distribution calculation done in {time_used:6.4f} sec !")
        return elec_dens_mesh


    def plot_elec_distribution(self, box=(10,10,10), isosur=(0.1,0.1,1), grid=0.1):
        '''
        Plot electronic distribution in real space. (NOT complete)

        Inputs:
            box: 3D-tuple, default is (10,10,10)
                giving the range for elec density and plot.
            isosur: 3D-tuple, default is (0.1,0.1,1)
                giving the isosurface setting.
                namely: (isomin, isomax, surface_count)
            grid: float, default is 0.1
                giving the grid for plot.
        '''
        x = np.arange(-box[0]/2, box[0]/2+grid, grid, )
        y = np.arange(-box[1]/2, box[1]/2+grid, grid, )
        z = np.arange(-box[2]/2, box[2]/2+grid, grid, )
        X,Y,Z = np.meshgrid(x,y,z, indexing='ij') # dim1=x,dim2=y,dim3=z
        elec_density = self.elec_density_mesh(X,Y,Z)

        # # along X-Y, cut by Z=0
        # elec_dens_xy = elec_density[:,:,0]
        # xx_xy,yy_xy = X[:,:,0], Y[:,:,0]
        # # others
        # elec_dens_xz = elec_density[:,0,:]
        # elec_dens_yz = elec_density[0,:,:]
        # xx_xz, zz_xz = np.meshgrid(x,z)
        # yy_yz, zz_yz = np.meshgrid(y,z)

        # plot 2D X-Y electron distrubition
        # plt.figure(figsize=(12,8))
        # plt.contourf(xx_xy, yy_xy, elec_dens_xy, 50, alpha=0.75, cmap='viridis')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # cb = plt.colorbar()
        # cb.set_label("charge density")
        # plt.show()

        # plot 3D electron distribution
        fig = go.Figure(
            go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=elec_density.flatten(),
                isomin=isosur[0],
                isomax=isosur[1],
                surface_count=isosur[2],
                colorscale="Viridis",
                colorbar=dict(title="charge density"),
                opacity=0.4,
            )
        )
                # caps=dict(x_show=False, y_show=False),
        fig.show()


def main():
    dia_len = 1.4632  # a.u.
    atoms = f'''
    He  0.0  0.0  0.0
    H   {dia_len}  0.0  0.0
    '''
    charge = +1
    atoms = Atoms(atoms, charge)
    energy = atoms.do_HF(max_iter=120, tol=1E-8, is_nuc=True)
    print(" --- MO infomation --- ")
    atoms.MO_infomation()
    # print()
    # print("kinetic matrix:")
    # print(atoms.T)
    # print("overlap matrix:")
    # print(atoms.S)
    # print("nuclei-electron interaction matrix:")
    # print(atoms.V_ne)
    # print("ERI (two-electron repulsion integrals) matrix:")
    # print(atoms.eri_m)

    # atoms.plot_elec_distribution(box=(12,10,8))
    
    
if __name__ == "__main__":
    main()


        