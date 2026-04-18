# Long-range Potts-Lattice-Gas in $d=1$: Position-Space Renormalization Group

A renormalization-group implementation for the one-dimensional Potts-Lattice-Gas (PLG) model with long-range power-law interactions.

The Ising treatment flows a single coupling vector $J_r$ under a two-cell Niemeyer–van Leeuwen decimation. The PLG model carries three coupling vectors that must close on each other under RG: the Potts coupling $J_r$, a pair (biquadratic / vacancy-vacancy) coupling $K_r$, and a per-distance chemical potential $\Delta_r$. Each step of this RG applies the three-coupling recursion relation to a discretized power-law interaction vector.

---

## The model

The Hamiltonian is

$$-\beta\mathcal{H} = \sum_{r_1 \neq r_2}\left[\,J_{|r_1-r_2|}\,\delta(s_{r_1}, s_{r_2})\,t_{r_1}t_{r_2} \;+\; K_{|r_1-r_2|}\,t_{r_1}t_{r_2} \;-\; \Delta_{|r_1-r_2|}\,(t_{r_1} + t_{r_2})\right]$$

where at each site $i$:

- $t_i \in \{0, 1\}$ is an occupancy variable (1 = site occupied, 0 = vacant);
- $s_i \in \{a, b, c\}$ is a Potts color defined only when $t_i = 1$.

The bare couplings are power-law initialized, $X_r = X_0 / r^a$ for $X \in \{J, K, \Delta\}$. The range exponent $a$ controls the physics: short-range-like at large $a$, equivalent-neighbor-like at small $a$. The rigorous-results cutoffs for the Ising case ($a=2$ marks the onset of first-order behavior, $a>2$ destroys ordering above $T=0$) are expected to carry over to the Potts-lattice-gas case with modifications from the extra coupling dimensions.

### State space per site

Encoded as a single integer $x_i \in \{0, 1, 2, 3\}$:
| $x_i$ | meaning |
|---|---|
| 0 | vacant ($t_i = 0$) |
| 1 | color $a$, occupied |
| 2 | color $b$, occupied |
| 3 | color $c$, occupied |

### Cell-spin projection (majority rule)

Each RG step projects 3 physical sites in a cell to a renormalized pair $(t'_\text{cell}, s'_\text{cell})$:

- $t'_\text{cell} = 1$ iff at least 2 of the 3 sites are occupied;
- $s'_\text{cell}$ is the majority color among occupied sites when $t'_\text{cell} = 1$;
- A 1–1–1 color tie is broken by the smallest-index occupied site (convention choice).

### Extraction of renormalized couplings

At each renormalized distance $r'$, four symmetry-distinct two-cell Boltzmann sums are computed:

- $R(00)$ — both cells vacant;
- $R(a0)$ — left cell color $a$, right cell vacant;
- $R(ab)$ — left cell color $a$, right cell color $b \neq a$;
- $R(aa)$ — both cells color $a$.

Potts color symmetry ensures the other orbits (e.g., $R(b0), R(c0), R(ac)$) are degenerate; only one representative per orbit is summed. The renormalized couplings follow directly from the hand-derived relations:

$$e^{J'_{r'}} = \frac{R(aa)}{R(ab)}, \qquad e^{K'_{r'}} = \frac{R(ab)\,R(00)}{R(a0)^2}, \qquad e^{\Delta'_{r'}} = \frac{R(00)}{R(a0)}.$$

---

### Switching between contiguous and staggered geometry

Edit the top of `renormalization.py`:

```python
from decimation_contiguous import *
#from decimation_staggered import *
```

to

```python
#from decimation_contiguous import *
from decimation_staggered import *
```

Everything downstream (`rg_step`, `find_Jc`, the driver) adapts automatically.

### Finding a critical coupling

```python
from renormalization_plg import find_Jc

Jc = find_Jc(a=1.2, K0=0.0, Delta0=0.0,
             max_steps=6, max_dist_final=9,
             tol=1e-5)
print(f"J_c(a=1.2, K0=Delta0=0) = {Jc:.5f}")
```

`find_Jc` does a bisection on the $J_0$ axis at fixed $K_0, \Delta_0, a$, classifying each initial point by whether $J[1]$ grows (ordered) or decays (disordered) under repeated RG.

---

## Numerical notes and design choices

### Log-space arithmetic everywhere

The $4^6 = 4096$ two-cell configurations are summed using log-sum-exp. Once the flow is pulled toward a phase sink, coupling magnitudes easily exceed $\log(10^{300}) \approx 690$, well into `exp`-overflow territory. The transfer-matrix builder constructs the exponent matrix in log space, subtracts the maximum, and exponentiates — this is the only numerically safe path for couplings in the regime this RG visits.

### Why `K` cannot be dropped

The PLG is not closed under this decimation with only $(J, \Delta)$. Starting from $K_0 = 0$, one RG step already generates $K'_{r'} \neq 0$ at all distances: the decimation mixes the color-matching term and the pair-occupancy term into a common effective coupling, so any truncation that omits $K$ will lose information at every step. The minimal closed coupling space is three-dimensional per distance.

### $\Delta = 0$ is not a fixed line

With the majority-rule projection, setting $J = K = \Delta = 0$ does **not** yield $\Delta' = 0$. Reason: at zero bare couplings, every microscopic configuration has equal weight, but the number of microscopic configurations projecting to "cell occupied with color $a$" differs from the number projecting to "vacant cell", so the ratio $R(00)/R(a0)$ is not 1. The trivial disordered fixed point of this RG lives at a finite negative $\Delta^*$, not at $\Delta = 0$. This is a known feature of majority-rule projections on models without a $\mathbb{Z}_2$ symmetry, not an implementation bug. The analogous effect does not occur in the Ising code because $s \to -s$ makes the $J = H = 0$ line an exact fixed line of that projection.

### Speed: the `r_max_cap` knob

Each RG step exactly decimates every distance $r'$ from 1 up to a geometric `r_max` determined by the initial-vector length. Because `required_initial_max_distance` grows like $3^{n_\text{steps}}$ on each step, running `n_steps = 10` with `max_dist_final = 10` asks for an initial vector of length ~700 000 — and correspondingly the first RG step would try to decimate ~230 000 distinct distances, each requiring a 4096-configuration sum.

The driver therefore passes an `r_max_cap` argument to `rg_step` (default: `max(max_dist_final + 4, 3 * max_dist_final)`), which caps the "head" of the exact decimation at a physically sensible distance. Beyond the cap, the tail is filled in by power-law continuation $X_r = X_{r_\text{cap}} \cdot (r_\text{cap}/r)^a$, same mechanism the Ising code uses. Raising the cap trades wall-clock time linearly for additional short-distance accuracy far from where the flow actually decides the phase.

For the Ising code this cap is less critical (2 states per site → 64-config inner sum), but it matters a lot here (4 states per site → 4096-config inner sum, a 64× multiplier).

### Symmetry bookkeeping in `log_R_four`

Potts color symmetry and lattice reflection combine to give multiple microscopic orbits that should all contribute to the same $R$. The code keeps exactly one representative per orbit:

- For $R(a0)$: only `(tL=1, sL=a, tR=0)`. The reflection $(0, a)$ is tracked via left-right symmetry of the lattice, which guarantees $R(0, a) = R(a, 0)$; not double-counted.
- For $R(ab)$: only `(sL=a, sR=b)`. The $(a, c)$ orbit is equivalent by Potts permutation; not double-counted.
- For $R(aa)$: only `(sL=a, sR=a)`. The $(b, b)$ and $(c, c)$ orbits are Potts-equivalent; not double-counted.

This is the bookkeeping that makes the three extraction formulas from the handwritten notes return the right coupling ratios rather than degenerate-orbit counts.

---

## Cell geometry reference

For `decimation_contiguous.py` (lattice spacing 1 within a cell):

```
Cell 0:      sites 1 2 3
Cell r':     sites 3r'+1, 3r'+2, 3r'+3
Intracell distances:   1, 1, 2   (same as Ising contiguous)
```

For `decimation_staggered.py` (lattice spacing 2 within a cell):

```
Cell 0:       sites 1 3 5
Cell r odd:   m = 2 + 6*(r-1)/2,      sites [m, m+2, m+4]
Cell r even:  m = 7 + 6*(r-2)/2,      sites [m, m+2, m+4]
Intracell distances:   2, 2, 4   (same as Ising staggered)
```

---

## Dependencies

- `numpy`
- `numba` (for JIT compilation of the hot inner loops)
- `matplotlib`

The Numba decorators cache compiled binaries on first run, so the second invocation in a session is much faster.

---
