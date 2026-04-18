"""
utils.py — Utilities for the Potts-Lattice-Gas long-range RG.

power-law initial couplings: J_r = J0 / r^a, and analogously K_r = K0 / r^a, Delta_r = Delta0 / r^a.

Convention: index 0 is unused (or zero), couplings live at r = 1..D.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def logsumexp(values):
    m = np.max(values)
    return m + np.log(np.sum(np.exp(values - m)))


def build_J(J0, a, D):
    """Power-law coupling vector J_r = J0 / r^a, for r = 1..D. J[0] = 0."""
    J = np.zeros(D + 1)
    r = np.arange(1, D + 1)
    J[1:] = J0 / (r ** a)
    return J


def build_K(K0, a, D):
    """Power-law biquadratic (vacancy-vacancy pair) coupling K_r = K0 / r^a."""
    K = np.zeros(D + 1)
    r = np.arange(1, D + 1)
    K[1:] = K0 / (r ** a)
    return K


def build_Delta(Delta0, a, D, long_range=True):
    """Chemical-potential vector.

    If long_range=True, Delta is a per-distance coupling Delta_r = Delta0 / r^a,
    matching the handwritten Hamiltonian where (t_i + t_j) appears inside the
    sum multiplied by |r_i - r_j|^{-a}.

    If long_range=False, Delta is treated as a single site chemical potential
    (broadcast to all r). Default matches the handwritten notes: long_range=True.
    """
    D_arr = np.zeros(D + 1)
    r = np.arange(1, D + 1)
    if long_range:
        D_arr[1:] = Delta0 / (r ** a)
    else:
        D_arr[1:] = Delta0
    return D_arr


def build_initial_couplings(J0, K0, Delta0, a, D, delta_long_range=True):
    """Convenience: build (J, K, Delta) vectors all at once."""
    J = build_J(J0, a, D)
    K = build_K(K0, a, D)
    Delta = build_Delta(Delta0, a, D, long_range=delta_long_range)
    return J, K, Delta
