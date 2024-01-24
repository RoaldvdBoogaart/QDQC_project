# encoding: utf-8

import numpy as np
from qutip import Qobj, sigmax, sigmay, sigmaz

# S=1/2 matrices
S_x = sigmax() / 2
S_y = sigmay() / 2
S_z = sigmaz() / 2

# S=1 matrices
S1_x = Qobj(np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
])) / np.sqrt(2)

S1_y = Qobj(np.array([
    [0, 1, 0],
    [-1, 0, 1],
    [0, -1, 0],
])) / (np.sqrt(2) * 1j)

S1_z = Qobj(np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, -1],
]))




