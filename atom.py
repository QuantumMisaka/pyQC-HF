import numpy as np
import PeriodicTable as PT

class Atom():
    '''
    Atom object.
    
    Inputs:
        ele: str
            element type of atom
    Attritubes:
        ele: str
            element type of atom
        ele_ind: int
            element index of atom
        charge: int
            charge of nuclei, or number of elec
        pos: np.ndarray
            position of atom, by xyz format
    '''
    def __init__(self, ele: str, coord: np.ndarray) -> None:
        self.ele = ele
        self.ele_ind = PT.Eledict[ele]
        self.charge = self.ele_ind
        self.pos = coord
        self.mass = PT.Elemass[self.ele_ind - 1]
        self.num_ao = 1 # only support sto-3g-s now