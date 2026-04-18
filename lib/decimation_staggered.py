"""
decimation_staggered.py
----------------------------
Staggered-cell variant of the d=1 long-range PLG two-cell decimation.

    Cell 0     occupies lattice sites [1, 3, 5]   (intracell spacing 2)
    Cell r'    occupies:
         odd r':  m = 2 + 6*((r-1)//2),  sites = [m, m+2, m+4]
         even r': m = 7 + 6*((r-2)//2),  sites = [m, m+2, m+4]

    r_max(D): the largest r' s.t. max pair distance (max(right) - 1) <= D.
    required_initial_max_distance uses the odd/even alternating recursion.

Hamiltonian and cell-spin projection are the same as the contiguous PLG
variant. The only thing that changes is the geometry of sites within and
between cells.

Intracell distances are {2, 2, 4}, so the intracell pairs use J[2] twice
and J[4] once, matching the Ising staggered intracell_energies().
"""

import numpy as np
from numba import njit


# =============================================================================
# Geometry
# =============================================================================

@njit(cache=True)
def required_initial_max_distance(max_dist_final, n_steps):
    D = max_dist_final
    for _ in range(n_steps):
        if D & 1:
            D = 3 * D + 2
        else:
            D = 3 * D + 4
    return D


@njit(cache=True)
def right_pos_staggered(r):
    if r % 2 == 1:  # odd r: even-type cell
        k = (r - 1) // 2
        m = 2 + 6 * k
    else:
        k = (r - 2) // 2
        m = 7 + 6 * k
    return np.array([m, m + 2, m + 4], dtype=np.int64)


@njit(cache=True)
def r_max(D):
    r = 1
    while True:
        rp = right_pos_staggered(r)
        dmax = rp[2] - 1  # max(right) - min(left) = (m+4) - 1
        if dmax > D:
            return r - 1
        r += 1


# =============================================================================
# Projection and pair energy (identical to contiguous PLG)
# =============================================================================

@njit(cache=True)
def project_cell(x0, x1, x2):
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

    if x0 != 0:
        return 1, x0
    if x1 != 0:
        return 1, x1
    return 1, x2


@njit(cache=True)
def pair_energy(xi, xj, Jd, Kd, Deltad):
    ti = 0 if xi == 0 else 1
    tj = 0 if xj == 0 else 1
    titj = ti * tj
    same = 1 if (titj == 1 and xi == xj) else 0
    return Jd * same * titj + Kd * titj - Deltad * (ti + tj)


# =============================================================================
# Two-cell log weight (staggered geometry)
# =============================================================================

@njit(cache=True)
def two_cell_log_weight(x0, x1, x2, x3, x4, x5, r_prime, J, K, Delta):
    D = len(J) - 1

    # left_pos = [1, 3, 5], right_pos = right_pos_staggered(r_prime)
    pos0 = 1
    pos1 = 3
    pos2 = 5
    rp = right_pos_staggered(r_prime)
    pos3 = rp[0]
    pos4 = rp[1]
    pos5 = rp[2]

    total = 0.0

    # Intracell 0: distances (1,3)=2, (1,5)=4, (3,5)=2
    d = pos1 - pos0
    if d <= D: total += pair_energy(x0, x1, J[d], K[d], Delta[d])
    d = pos2 - pos0
    if d <= D: total += pair_energy(x0, x2, J[d], K[d], Delta[d])
    d = pos2 - pos1
    if d <= D: total += pair_energy(x1, x2, J[d], K[d], Delta[d])

    # Intracell r': same spacings (2, 4, 2)
    d = pos4 - pos3
    if d <= D: total += pair_energy(x3, x4, J[d], K[d], Delta[d])
    d = pos5 - pos3
    if d <= D: total += pair_energy(x3, x5, J[d], K[d], Delta[d])
    d = pos5 - pos4
    if d <= D: total += pair_energy(x4, x5, J[d], K[d], Delta[d])

    # Intercell: 9 pairs
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
# log-sum-exp
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
# The four symmetry-distinct Boltzmann sums
# =============================================================================

@njit(cache=True)
def log_R_four(r_prime, J, K, Delta):
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
                                pass
                            else:
                                if sL == 1:
                                    if sR == 1:
                                        buf_aa[n_aa] = E; n_aa += 1
                                    elif sR == 2:
                                        buf_ab[n_ab] = E; n_ab += 1

    log_R_00 = _lse(buf_00, n_00)
    log_R_a0 = _lse(buf_a0, n_a0)
    log_R_ab = _lse(buf_ab, n_ab)
    log_R_aa = _lse(buf_aa, n_aa)

    return log_R_00, log_R_a0, log_R_ab, log_R_aa


@njit(cache=True)
def renorm_at_r(r_prime, J, K, Delta):
    logR00, logRa0, logRab, logRaa = log_R_four(r_prime, J, K, Delta)
    Jp = logRaa - logRab
    Kp = logRab + logR00 - 2.0 * logRa0
    Dp = logR00 - logRa0
    return Jp, Kp, Dp
