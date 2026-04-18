"""
decimation_contiguous_plg.py
-----------------------------
Niemeyer-van Leeuwen two-cell cluster decimation for the
d=1 long-range Potts-Lattice-Gas (PLG) model.

This file is the Potts-lattice-gas analog of decimation_contiguous.py
(Ising, spin +/-1). It uses the IDENTICAL cell geometry and indexing
convention as the Ising file:

    Cell 0     occupies lattice sites [1, 2, 3]
    Cell r'    occupies lattice sites [3r'+1, 3r'+2, 3r'+3]
               == [4, 5, 6] + 3*(r' - 1)

    r_max(D)                          = (D - 2) // 3
    required_initial_max_distance(Df, n_steps): D <- 3D + 2 each step

Hamiltonian (per pair of sites i, j at separation d = |r_i - r_j|,
with the |r_i - r_j|^{-a} already absorbed into the coupling vectors):

    -beta h_{ij} = J_d * delta(s_i, s_j) * t_i t_j
                  + K_d * t_i t_j
                  - Delta_d * (t_i + t_j)

Site state encoding (4 states per site, analog of Ising's +/-1):
    0 = vacant (t=0)
    1 = color 'a' (t=1, s=a)
    2 = color 'b' (t=1, s=b)
    3 = color 'c' (t=1, s=c)

Cell-spin projection (majority rule extension of Ising's majority):
    t'_cell = 1 iff at least 2 of the 3 sites are occupied.
    s'_cell = majority color among occupied sites when t'_cell = 1,
              with a 1-1-1 tie broken by the smallest-index occupied site.
    (When t'_cell = 0, s'_cell is undefined; treated as 0.)

Renormalized couplings at distance r' are extracted from four
symmetry-distinct Boltzmann sums over the 4^6 = 4096 two-cell configs:

    R(00):   t'_L = 0, t'_R = 0
    R(a0):   t'_L = 1 (color a), t'_R = 0
    R(ab):   t'_L = 1 (color a), t'_R = 1 (color b != a)
    R(aa):   t'_L = 1 (color a), t'_R = 1 (color a)

via:
    J'_{r'}     = log R(aa) - log R(ab)
    K'_{r'}     = log R(ab) + log R(00) - 2 log R(a0)
    Delta'_{r'} = log R(00) - log R(a0)

(Potts q = 3. The color symmetry ensures R(a,0) = R(b,0) = R(c,0),
R(a,b) = R(a,c) = R(b,c), etc. We keep exactly one representative per
orbit to avoid double-counting the relative weights.)
"""

import numpy as np
from numba import njit


# =============================================================================
# Cell geometry (matches Ising contiguous)
# =============================================================================

@njit(cache=True)
def required_initial_max_distance(max_dist_final, n_steps):
    D = max_dist_final
    for _ in range(n_steps):
        D = 3 * D + 2
    return D


@njit(cache=True)
def r_max(D):
    return (D - 2) // 3


# =============================================================================
# Cell-spin projection (majority rule)
# =============================================================================

@njit(cache=True)
def project_cell(x0, x1, x2):
    """Majority-rule projection of 3 sites to (t'_cell, s'_cell)."""
    occ = 0
    if x0 != 0: occ += 1
    if x1 != 0: occ += 1
    if x2 != 0: occ += 1

    if occ < 2:
        return 0, 0

    ca = 0; cb = 0; cc = 0
    if x0 == 1:   ca += 1
    elif x0 == 2: cb += 1
    elif x0 == 3: cc += 1
    if x1 == 1:   ca += 1
    elif x1 == 2: cb += 1
    elif x1 == 3: cc += 1
    if x2 == 1:   ca += 1
    elif x2 == 2: cb += 1
    elif x2 == 3: cc += 1

    if ca > cb and ca > cc:
        return 1, 1
    if cb > ca and cb > cc:
        return 1, 2
    if cc > ca and cc > cb:
        return 1, 3

    # Tie-break: color of smallest-index occupied site.
    if x0 != 0:
        return 1, x0
    if x1 != 0:
        return 1, x1
    return 1, x2


# =============================================================================
# Pair interaction
# =============================================================================

@njit(cache=True)
def pair_energy(xi, xj, Jd, Kd, Deltad):
    """-beta h_{ij} for a single pair."""
    ti = 0 if xi == 0 else 1
    tj = 0 if xj == 0 else 1
    titj = ti * tj
    same_color = 1 if (titj == 1 and xi == xj) else 0

    return (Jd * same_color * titj
            + Kd * titj
            - Deltad * (ti + tj))


# =============================================================================
# Two-cell Hamiltonian (15 pairs)
# =============================================================================
#
# Contiguous: left_pos = [1,2,3], right_pos = [3r'+1, 3r'+2, 3r'+3].
# Intracell pair distances = {1, 2, 1}. Matches Ising's
#   intracell_energies = J1*(s0*s1 + s1*s2) + J2*(s0*s2).

@njit(cache=True)
def two_cell_log_weight(x0, x1, x2, x3, x4, x5, r_prime, J, K, Delta):
    """Sum of -beta h_{ij} over the 15 pairs in the 6-spin two-cell cluster.

    Silently drops any pair whose distance exceeds len(J)-1, matching
    the Ising file's `if d <= D` guard.
    """
    D = len(J) - 1

    pos0 = 1
    pos1 = 2
    pos2 = 3
    pos3 = 3 * r_prime + 1
    pos4 = 3 * r_prime + 2
    pos5 = 3 * r_prime + 3

    total = 0.0

    # Intra-cell 0: (0,1), (0,2), (1,2) -> distances 1, 2, 1
    d = pos1 - pos0
    if d <= D: total += pair_energy(x0, x1, J[d], K[d], Delta[d])
    d = pos2 - pos0
    if d <= D: total += pair_energy(x0, x2, J[d], K[d], Delta[d])
    d = pos2 - pos1
    if d <= D: total += pair_energy(x1, x2, J[d], K[d], Delta[d])

    # Intra-cell r': (3,4), (3,5), (4,5) -> distances 1, 2, 1
    d = pos4 - pos3
    if d <= D: total += pair_energy(x3, x4, J[d], K[d], Delta[d])
    d = pos5 - pos3
    if d <= D: total += pair_energy(x3, x5, J[d], K[d], Delta[d])
    d = pos5 - pos4
    if d <= D: total += pair_energy(x4, x5, J[d], K[d], Delta[d])

    # Inter-cell: 9 pairs
    d = pos3 - pos0
    if d <= D: total += pair_energy(x0, x3, J[d], K[d], Delta[d])
    d = pos4 - pos0
    if d <= D: total += pair_energy(x0, x4, J[d], K[d], Delta[d])
    d = pos5 - pos0
    if d <= D: total += pair_energy(x0, x5, J[d], K[d], Delta[d])
    d = pos3 - pos1
    if d <= D: total += pair_energy(x1, x3, J[d], K[d], Delta[d])
    d = pos4 - pos1
    if d <= D: total += pair_energy(x1, x4, J[d], K[d], Delta[d])
    d = pos5 - pos1
    if d <= D: total += pair_energy(x1, x5, J[d], K[d], Delta[d])
    d = pos3 - pos2
    if d <= D: total += pair_energy(x2, x3, J[d], K[d], Delta[d])
    d = pos4 - pos2
    if d <= D: total += pair_energy(x2, x4, J[d], K[d], Delta[d])
    d = pos5 - pos2
    if d <= D: total += pair_energy(x2, x5, J[d], K[d], Delta[d])

    return total


# =============================================================================
# Log-sum-exp helper
# =============================================================================

@njit(cache=True)
def _lse(buf, n):
    if n == 0:
        return -np.inf
    m = buf[0]
    for i in range(1, n):
        if buf[i] > m:
            m = buf[i]
    s = 0.0
    for i in range(n):
        s += np.exp(buf[i] - m)
    return m + np.log(s)


# =============================================================================
# The four Boltzmann sums at distance r'
# =============================================================================

@njit(cache=True)
def log_R_four(r_prime, J, K, Delta):
    """Return (log R(00), log R(a0), log R(ab), log R(aa)) at distance r'.

    Potts color symmetry implies:
        R(a,0) = R(b,0) = R(c,0)    [take a-representative only]
        R(a,a) = R(b,b) = R(c,c)    [take (a,a)-representative only]
        R(a,b) = R(a,c) = R(b,a) = ... [take (a,b)-representative only]

    and lattice reflection:
        R(0, a) = R(a, 0)           [not re-counted; using L-colored rep]

    These orbit choices keep the three extraction formulas
        exp J'  = R(aa) / R(ab)
        exp K'  = R(ab) R(00) / R(a0)^2
        exp D'  = R(00) / R(a0)
    consistent with the handwritten derivation.
    """
    BUF = 4096
    buf_00 = np.empty(BUF, dtype=np.float64); n_00 = 0
    buf_a0 = np.empty(BUF, dtype=np.float64); n_a0 = 0
    buf_ab = np.empty(BUF, dtype=np.float64); n_ab = 0
    buf_aa = np.empty(BUF, dtype=np.float64); n_aa = 0

    for x0 in range(4):
        for x1 in range(4):
            for x2 in range(4):
                tL, sL = project_cell(x0, x1, x2)
                for x3 in range(4):
                    for x4 in range(4):
                        for x5 in range(4):
                            tR, sR = project_cell(x3, x4, x5)
                            E = two_cell_log_weight(
                                x0, x1, x2, x3, x4, x5,
                                r_prime, J, K, Delta
                            )

                            if tL == 0 and tR == 0:
                                buf_00[n_00] = E; n_00 += 1
                            elif tL == 1 and tR == 0:
                                if sL == 1:
                                    buf_a0[n_a0] = E; n_a0 += 1
                            elif tL == 0 and tR == 1:
                                # (vac, color) is the L-R reflection of
                                # (color, vac); by lattice symmetry we
                                # don't count it in R(a0). This leaves
                                # R(a0) = sum over (tL=1,sL=a,tR=0) only,
                                # which is what the formulas assume.
                                pass
                            else:  # both occupied
                                if sL == 1:
                                    if sR == 1:
                                        buf_aa[n_aa] = E; n_aa += 1
                                    elif sR == 2:
                                        buf_ab[n_ab] = E; n_ab += 1
                                    # sR == 3: (a,c) orbit; skip as Potts-
                                    # equivalent to (a,b).

    log_R_00 = _lse(buf_00, n_00)
    log_R_a0 = _lse(buf_a0, n_a0)
    log_R_ab = _lse(buf_ab, n_ab)
    log_R_aa = _lse(buf_aa, n_aa)

    return log_R_00, log_R_a0, log_R_ab, log_R_aa


# =============================================================================
# Extract renormalized (J', K', Delta') at single r'
# =============================================================================

@njit(cache=True)
def renorm_at_r(r_prime, J, K, Delta):
    logR00, logRa0, logRab, logRaa = log_R_four(r_prime, J, K, Delta)
    Jp = logRaa - logRab
    Kp = logRab + logR00 - 2.0 * logRa0
    Dp = logR00 - logRa0
    return Jp, Kp, Dp
