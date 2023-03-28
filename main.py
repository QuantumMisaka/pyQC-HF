from atoms import Atoms

def main():
    dia_len = 1.4632  # a.u.
    # dia_len = 1.2 # a.u. for H2
    atoms = f'''
    He  0.0  0.0  0.0
    H  {dia_len}  0.0  0.0
    '''
    charge = +1
    atoms = Atoms(atoms, charge)
    atoms.do_HF(max_iter=120, tol=1E-8, is_nuc=True)
    atoms.MO_infomation()
        
    print()
    print("kinetic matrix:")
    print(atoms.T)
    print("overlap matrix:")
    print(atoms.S)
    print("nuclei-electron interaction matrix:")
    print(atoms.V_ne)
    print("ERI (two-electron repulsion integrals) matrix:")
    print(atoms.eri_m)

    atoms.plot_elec_distribution(box=(8,6,6), isosur=(0.001,0.2,20), grid=0.1)
    
if __name__ == "__main__":
    main()