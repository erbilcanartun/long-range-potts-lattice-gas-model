"""
renormalization.py
----------------------
Main RG driver for the d=1 long-range Potts-Lattice-Gas model.

    - rg_step(J, K, Delta, a=None)  -> (J', K', Delta')
    - required_initial_max_distance(max_dist_final, n_steps)
    - construct_transfer_matrix(J, K, Delta, r)
    - find_Jc(...)  [bisection on the J0 axis at fixed K0, Delta0]

Choose the cell geometry by commenting one of the two imports below.
"""

import numpy as np
from numba import njit

#from decimation_contiguous import *
from decimation_staggered import *


@njit(cache=True)
def determine_r_max(D):
    return r_max(D)


# =============================================================================
# Single RG step
# =============================================================================

@njit(cache=True)
def rg_step(J, K, Delta, a=None, aK=None, aD=None, r_max_cap=-1):
    """One RG iteration.

    Parameters
    ----------
    J, K, Delta : np.ndarray, shape (D+1,)
        Current coupling vectors. Index 0 unused (set to 0). Index r in 1..D.
    a : float or None
        If given, use a power-law tail continuation J'_r = J'_{r_max} * (r_max/r)**a
        for r > r_max. Same treatment for K and Delta (reuse a unless aK/aD given).
    aK, aD : float or None
        Optional independent exponents for K and Delta tails. Default to `a`.
    r_max_cap : int
        If positive, cap the number of distances at which we compute the exact
        renormalization (head) and use the power-law (or flat, if a=None)
        continuation for r > r_max_cap. Defaults to -1 (no cap), in which case
        head is computed up to the geometric r_max (= (D-2)//3 for contiguous,
        a different lookup for staggered). This cap is the single most
        important speed knob for the PLG model, which otherwise spends most
        of its time decimating distances that only feed the tail.

    Returns
    -------
    (J', K', Delta') : each of shape (D+1,)
    """
    D = len(J) - 1
    rm = determine_r_max(D)
    if r_max_cap > 0 and r_max_cap < rm:
        rm = r_max_cap

    J_new = np.zeros_like(J)
    K_new = np.zeros_like(K)
    D_new = np.zeros_like(Delta)

    # --- Head: exact renormalization for r' = 1 .. rm
    for rp in range(1, rm + 1):
        Jp, Kp, Dp = renorm_at_r(rp, J, K, Delta)
        J_new[rp] = Jp
        K_new[rp] = Kp
        D_new[rp] = Dp

    # --- Tail: power-law continuation
    if a is not None:
        aJ = a
        aKk = a if aK is None else aK
        aDd = a if aD is None else aD

        anchor_J = J_new[rm]
        anchor_K = K_new[rm]
        anchor_D = D_new[rm]
        for r in range(rm + 1, D + 1):
            ratio = rm / r
            J_new[r] = anchor_J * ratio ** aJ
            K_new[r] = anchor_K * ratio ** aKk
            D_new[r] = anchor_D * ratio ** aDd

    return J_new, K_new, D_new


# =============================================================================
# Transfer matrix at a single distance r
# =============================================================================
#
# The 4-state per-site pair "transfer matrix" T_{ij}(xi, xj) at distance r is:
#     T(xi, xj) = exp( pair_energy(xi, xj; J_r, K_r, Delta_r) )
# It is 4x4 (states: 0 = vacant, 1/2/3 = colors).
#
# We normalize so the largest entry is 1 (same convention as your Ising code).

def construct_transfer_matrix(J, K, Delta, r, normalize=True):
    if r >= len(J):
        raise ValueError(
            f"Cannot build transfer matrix: distance r={r} > max available {len(J)-1}"
        )
    Jr = J[r]
    Kr = K[r]
    Dr = Delta[r]

    # Build the log-weights first, subtract the max, then exponentiate.
    # This is the only safe path once the flow starts heading to a phase sink,
    # where couplings can easily exceed log(1e300) ~ 690.
    logT = np.zeros((4, 4), dtype=float)
    for xi in range(4):
        ti = 0 if xi == 0 else 1
        for xj in range(4):
            tj = 0 if xj == 0 else 1
            titj = ti * tj
            same = 1 if (titj == 1 and xi == xj) else 0
            logT[xi, xj] = Jr * same * titj + Kr * titj - Dr * (ti + tj)

    if normalize:
        logT = logT - np.max(logT)
        return np.exp(logT)
    else:
        return np.exp(logT)


def determine_phase_from_TM(T, threshold=0.1):
    """Heuristic phase label from the (normalized) 4x4 transfer matrix.

    Inspired by your Ising determine_phase_from_TM but adapted to the
    richer state space:
        - Disordered: diagonal color block ~ off-diagonal color block AND
          vacancy block non-negligible.
        - Ferromagnetic (Potts-ordered, dense): same-color entries ~1,
          off-color ~0, vacancy entries small.
        - Dilute / vacancy-dominated: T[0,0] dominates.
    """
    diag_color = (T[1, 1] + T[2, 2] + T[3, 3]) / 3.0
    off_color = (T[1, 2] + T[1, 3] + T[2, 3]) / 3.0
    vac = T[0, 0]

    if diag_color > 1 - threshold and off_color < threshold:
        return "potts_ordered"
    if vac > 1 - threshold and diag_color < threshold and off_color < threshold:
        return "dilute"
    if diag_color > 1 - threshold and off_color > 1 - threshold:
        return "disorder"
    return "undetermined"


# =============================================================================
# Critical coupling finder (bisection on J0 axis)
# =============================================================================

def find_Jc(a, K0=0.0, Delta0=0.0,
            Jlow=1e-2, Jhigh=1e2,
            max_steps=6, max_dist_final=9,
            tol=1e-5, growth_threshold=1e3, decay_threshold=1e-3,
            delta_long_range=True):
    """Bisection on J0 at fixed (K0, Delta0, a): finds the critical J0
    separating flows to the Potts-ordered phase sink (growth of J[1])
    from flows to the disordered phase sink (decay of J[1]).
    """
    if not (0 <= a <= 2):
        raise ValueError("a must be in [0,2]")

    # Build initial vectors of the right length
    D0 = required_initial_max_distance(max_dist_final, max_steps)

    # Local import to avoid a hard dep cycle if users rename utils
    from utils_plg import build_J, build_K, build_Delta

    def grows(J0_val):
        J = build_J(J0_val, a, D0)
        K = build_K(K0, a, D0)
        Dlt = build_Delta(Delta0, a, D0, long_range=delta_long_range)
        J1_initial = abs(J[1])
        for _ in range(max_steps):
            if abs(J[1]) > growth_threshold:
                return True
            if abs(J[1]) < decay_threshold:
                return False
            J, K, Dlt = rg_step(J, K, Dlt)
        return abs(J[1]) > J1_initial

    while (Jhigh - Jlow) > tol:
        Jmid = 0.5 * (Jlow + Jhigh)
        if grows(Jmid):
            Jhigh = Jmid
        else:
            Jlow = Jmid
    return 0.5 * (Jlow + Jhigh)
