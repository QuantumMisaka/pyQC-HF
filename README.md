## pyQC: python code for Quantum Chemistry Calculation

### Introduction

pyQC-HF: Hartree-Fock method in python.

Using STO-3G basis set to solve the Hartree-Fock with SCF iterations by using object-oriented python code.

It's now only part of homework in the *Quantum Theory in Many-body System* class

### Authors

JamesMisaka & i.i.d , PKU., CCME.

### Implement of one electron and two electron integrals matrix calculation

Using analytical from of STO-3G and it's integral in Szabo and Ostlund, appendix A and B. 
and using parameter from BSE database http://www.basissetexchange.org/

### Implement the SCF procedure
We will be basically following the algorithm described in Szabo and Ostlund, 3.4.6, p. 146 to implement the Hartree-Fock method for $\mathrm{HeH}^+$ system. 
The steps are as follows:

1. Obtain a guess at the density matrix.
2. Calculate the exchange and coulomb matrices from the density matrix and the two-electron repulsion integrals.
3. Add exchange and coulomb matrices to the core-Hamiltonian to obtain the Fock matrix.
4. Diagonalize the Fock matrix.
5. Select the occupied orbitals and calculate the new density matrix.
6. Compute the energy.
7. Compute the errors and check for convergence.
  - If converged, return the energy.
  - If not converged, return to second step with the new density matrix.

You can also try other molecules, such as $\mathrm{H2}^+$ and $\mathrm{He2}$.

Code of this part is mainly learned from yangdatou's Hartree-Fock tutorial

MO_analysis and electron density distribution function are also implemented

### Easy Usage 
python main.py

One can also modify the context of main.py to run other atoms system,
Unfortunately, now this code only support H and He atom.

### Dependencies
- `numpy`
- `scipy`
- If you wish to draw electron density distribution, `matplotlib` and `plotly` are also required

### Reference
- Szabo and Ostlund, _Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory_,
  Dover Publications, New York, 1996
- https://github.com/yangdatou/hf-tutorial
