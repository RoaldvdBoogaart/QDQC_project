# encoding: utf-8

import numpy as np
from qutip import Qobj

from scipy.constants import elementary_charge, Planck
from scipy.constants import physical_constants

BOHR_MAGNETON_EV = physical_constants['Bohr magneton in eV/T'][0]
G_NV = 2  # electron g-factor
G_N = 2  # electron g-factor

def nitrogen_vacancy_hamiltonian(b_field: float, spin_ops: tuple[Qobj]) -> Qobj:
    """Hamiltonian of the NV center electron spin. Hamiltonian based on: PhysRevLett.97.087601 (https://arxiv.org/abs/quant-ph/0605179). 
        Returns:
            H_NV (Qobj): Hamiltonian of the NV center electron spin
    """
    # unpack operators
    NV_Sx, NV_Sy, NV_Sz = spin_ops

    # define prefactors
    D = 2.88e9  # zero-field splitting. unit: Hz
    B0 = G_NV * BOHR_MAGNETON_EV * b_field * elementary_charge / (Planck)  # Zeeman splitting energy of NV electron. unit: Hz

    # define Hamiltonian contributions
    H_NV = D * (NV_Sz * NV_Sz - (1 / 3) * (NV_Sx * NV_Sx + NV_Sy * NV_Sy + NV_Sz * NV_Sz)) + B0 * NV_Sz

    return H_NV

def nitrogen_atom_hamiltonian(b_field: float, hf_coupling: float, nuclues_s_ops: tuple[Qobj], electron_s_ops, rwa: bool = False) -> Qobj:
    """Hamiltonian of the N electron spin. Hamiltonian based on: PhysRevLett.97.087601 (https://arxiv.org/abs/quant-ph/0605179). 
        Returns:
            H_N (Qobj): Hamiltonian of the N electron spin
    """
    # unpack operators
    I_Sx, I_Sy, I_Sz = nuclues_s_ops
    N_Sx, N_Sy, N_Sz = electron_s_ops

    # define prefactors
    B1 = G_N * BOHR_MAGNETON_EV * b_field * elementary_charge / (Planck)  # Zeeman splitting energy of N electron. unit: Hz

    # define Hamiltonian contributions
    H_N = B1 * N_Sz + hf_coupling * (N_Sx * I_Sx + N_Sy * I_Sy + N_Sz * I_Sz)

    if rwa:
        H_N = B1 * N_Sz + hf_coupling * (N_Sz * I_Sz)

    return H_N

def hyperfine_coupling(hf_coupling: float, nuclues_s_ops: tuple[Qobj], electron_s_ops: tuple[Qobj], rwa: bool = True) -> Qobj:
    """Hamiltonian of the hyperfine coupling between NV center electron and N electron. Hamiltonian based on: PhysRevLett.97.087601 (https://arxiv.org/abs/quant-ph/0605179).


        Returns:
            H_HF (Qobj): Hamiltonian of the hyperfine coupling between NV center electron and N electron
    """
    # unpack operators
    I_Sx, I_Sy, I_Sz = nuclues_s_ops
    N_Sx, N_Sy, N_Sz = electron_s_ops

    H_HF = hf_coupling * (N_Sx * I_Sx + N_Sy * I_Sy + N_Sz * I_Sz)

    if rwa:
        H_HF = hf_coupling * (N_Sz * I_Sz)

    return H_HF

def dipolar_coupling(theta, distance, nv_s_ops: tuple[Qobj], n_s_ops: tuple[Qobj], rwa: bool = False) -> Qobj:
    """Hamiltonian of the dipolar coupling between NV center electron and N electron. Hamiltonian based on: PhysRevLett.97.087601 (https://arxiv.org/abs/quant-ph/0605179). 
        Returns:
            H_dip (Qobj): Hamiltonian of the dipolar coupling between NV center electron and N electron
    """
    # unpack operators
    NV_Sx, NV_Sy, NV_Sz = nv_s_ops
    N_Sx, N_Sy, N_Sz = n_s_ops

    # define dipolar coupling coefficients between electrons of NV and N (fine-structure tensor)
    x, y, z = distance * np.sin(-np.pi * theta / 180), 0, distance * np.cos(np.pi * theta / 180)
    Mrr = np.kron(np.array([x, y, z]), np.array([[x], [y], [z]])) / (distance**2)
    J = (np.eye(3) - 3 * Mrr) * 1e-7 * G_NV * G_N * (BOHR_MAGNETON_EV**2 * elementary_charge**2) / (distance**3 * Planck)

    # define Hamiltonian contributions
    H_dip = NV_Sx*N_Sx*J[0,0] + NV_Sx*N_Sy*J[0,1] + NV_Sx*N_Sz*J[0,2] \
            + NV_Sy*N_Sx*J[1,0] + NV_Sy*N_Sy*J[1,1] + NV_Sy*N_Sz*J[1,2] \
            + NV_Sz*N_Sx*J[2,0] + NV_Sz*N_Sy*J[2,1] + NV_Sz*N_Sz*J[2,2]

    if rwa:
        H_dip = (1 - 3 * np.cos(theta) ** 2) * (3 * NV_Sz * N_Sz - (NV_Sx * N_Sx + NV_Sy * N_Sy + NV_Sz * N_Sz))

    return H_dip