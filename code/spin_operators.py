# encoding: utf-8

import numpy as np
from qutip import Qobj

# S=1 matrices
spin_1_x = Qobj(np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
])) / np.sqrt(2)

spin_1_y = Qobj(np.array([
    [0, 1, 0],
    [-1, 0, 1],
    [0, -1, 0],
])) / (np.sqrt(2) * 1j)

spin_1_z = Qobj(np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, -1],
]))
