# Long-range Potts-Lattice-Gas in $d=1$: Position-Space Renormalization Group

A renormalization-group implementation for the one-dimensional Potts-Lattice-Gas (PLG) model with long-range power-law interactions, built as a direct analog of the long-range Ising RG code of Artun & Berker ([arXiv / *Phys. Rev. E* write-up](https://journals.aps.org), "Calculated Magnetization Curves and Hybrid-Order Phase Transition of the d=1 Ising Model with Long-Range Power-Law Interactions").

The Ising treatment flows a single coupling vector $J_r$ under a two-cell Niemeyer–van Leeuwen decimation. The PLG model carries three coupling vectors that must close on each other under RG: the Potts coupling $J_r$, a pair (biquadratic / vacancy-vacancy) coupling $K_r$, and a per-distance chemical potential $\Delta_r$. Each step of this RG applies the three-coupling recursion derived in the handwritten notes of Umut (included separately) to a discretized power-law interaction vector.

---

## The model

The Hamiltonian is

$$-\beta\mathcal{H} = \sum_{r_1 \neq r_2}\left[\,J_{|r_1-r_2|}\,\delta(s_{r_1}, s_{r_2})\,t_{r_1}t_{r_2} \;+\; K_{|r_1-r_2|}\,t_{r_1}t_{r_2} \;-\; \Delta_{|r_1-r_2|}\,(t_{r_1} + t_{r_2})\right]$$

where at each site $i$:

- $t_i \in \{0, 1\}$ is an occupancy variable (1 = site occupied, 0 = vacant);
- $s_i \in \{a, b, c\}$ is a Potts color defined only when $t_i = 1$.

The bare couplings are power-law initialized, $X_r = X_0 / r^a$ for $X \in \{J, K, \Delta\}$, exactly as in the Ising code. The range exponent $a$ controls the physics: short-range-like at large $a$, equivalent-neighbor-like at small $a$. The rigorous-results cutoffs for the Ising case ($a=2$ marks the onset of first-order behavior, $a>2$ destroys ordering above $T=0$) are expected to carry over to the Potts-lattice-gas case with modifications from the extra coupling dimensions.

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

## Files

The code is organized to mirror the Ising implementation file-for-file. If you have the Ising files `utils.py`, `decimation_contiguous.py`, `decimation_staggered.py`, `renormalization.py`, their PLG analogs are:

| PLG file | Mirrors (Ising) | Role |
|---|---|---|
| `utils_plg.py` | `utils.py` | Power-law builders `build_J`, `build_K`, `build_Delta` (index 0 unused, index $r$ for $r \geq 1$). Also exports a Numba-compatible `logsumexp`. |
| `decimation_contiguous_plg.py` | `decimation_contiguous.py` | Two-cell decimation with the contiguous geometry: cell 0 sits at lattice sites $[1, 2, 3]$, cell $r'$ at $[3r'+1, 3r'+2, 3r'+3]$. Intracell spacing = 1. `r_max(D) = (D-2)//3`. `required_initial_max_distance`: $D \leftarrow 3D + 2$ per step. |
| `decimation_staggered_plg.py` | `decimation_staggered.py` | Two-cell decimation with the staggered geometry: cell 0 at $[1, 3, 5]$, cell $r'$ at a staggered offset that alternates parity. Intracell spacing = 2. `r_max(D)` is an iterative lookup. `required_initial_max_distance` alternates $3D+2$ and $3D+4$ depending on parity. |
| `renormalization_plg.py` | `renormalization.py` | Orchestration: `rg_step`, `determine_r_max`, `construct_transfer_matrix` (4×4), `determine_phase_from_TM`, `find_Jc`. Imports from one of the two decimation files via `from decimation_{...}_plg import *`; switch by commenting one of the two import lines at the top. |
| `run_rg_flow_plg.py` | (driver script) | End-user API: `generate_rg_flow`, `extract_flows`, `plot_rg_flow`. Analog of your hand-written Ising driver. |

Each decimation file provides the same named functions used by the orchestrator: `r_max`, `required_initial_max_distance`, `renorm_at_r`, `log_R_four`, `project_cell`, `pair_energy`, `two_cell_log_weight`. This keeps the swap between contiguous and staggered to a single-line import change.

---

## Quick start

```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'lib')))

from run_rg_flow_plg import generate_rg_flow, extract_flows, plot_rg_flow

# Bare couplings at r=1 and power-law range exponent
J0, K0, Delta0 = 0.5, 0.0, 0.3
a = 1.2

# How far out in r we want the final coupling vectors to reach,
# and how many RG iterations to run
max_dist_final = 10
n_steps = 10

couplings, T_list = generate_rg_flow(
    J0, K0, Delta0, a,
    max_dist_final=max_dist_final,
    n_steps=n_steps,
    trace_TM=True,  TM_r=1,
)

flows = extract_flows(couplings, max_dist_final)

plot_rg_flow(
    flows,
    distances_to_plot=list(range(1, 10)),
    fig_name="rg_flow_plg.png",
)

# Inspect the normalized 4x4 pair transfer matrix at r=1 at each step
for k, T in enumerate(T_list):
    print(f"RG step {k}:\n", T.round(4))
```

`couplings` is a dict with keys `'J'`, `'K'`, `'Delta'`, each mapping to a list of length `n_steps + 1` containing the coupling vector at every step. `extract_flows` reshapes each entry into a rectangular array of shape `(n_steps+1, max_dist_final+1)` for plotting.

### Switching between contiguous and staggered geometry

Edit the top of `renormalization_plg.py`:

```python
from decimation_contiguous_plg import *
#from decimation_staggered_plg import *
```

to

```python
#from decimation_contiguous_plg import *
from decimation_staggered_plg import *
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

For `decimation_contiguous_plg.py` (lattice spacing 1 within a cell):

```
Cell 0:      sites 1 2 3
Cell r':     sites 3r'+1, 3r'+2, 3r'+3
Intracell distances:   1, 1, 2   (same as Ising contiguous)
```

For `decimation_staggered_plg.py` (lattice spacing 2 within a cell):

```
Cell 0:       sites 1 3 5
Cell r odd:   m = 2 + 6*(r-1)/2,      sites [m, m+2, m+4]
Cell r even:  m = 7 + 6*(r-2)/2,      sites [m, m+2, m+4]
Intracell distances:   2, 2, 4   (same as Ising staggered)
```

Both conventions are verified against the corresponding Ising files (`decimation_contiguous.py`, `decimation_staggered.py`) and produce identical `right_pos` sequences and `r_max` behavior — only the state-space per site and the pair-energy formula differ.

---

## Dependencies

- `numpy`
- `numba` (for JIT compilation of the hot inner loops)
- `matplotlib` (only in `run_rg_flow_plg.py`, for plotting)

The Numba decorators cache compiled binaries on first run, so the second invocation in a session is much faster.

---

## Attribution

Model derivation and Potts-lattice-gas setup: handwritten notes by Umut.
RG methodology and cell structure: adapted from Artun & Berker's long-range Ising treatment (*Phys. Rev. E*), itself built on the two-cell Niemeyer–van Leeuwen procedure [Niemeyer & van Leeuwen, *Physica* 71, 17 (1974); van Leeuwen, *Phys. Rev. Lett.* 34, 1056 (1975); Berker & Wortis, *Phys. Rev. B* 14, 4946 (1976)].
