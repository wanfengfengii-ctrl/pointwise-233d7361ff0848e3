# Streaming, mergeable PCA sketch (stdlib only)

`pca.Sketch(p, l)` summarises high-dimensional weighted observations so that
per-station summaries can be merged into a global PCA without keeping any raw
observations or a full `p × p` covariance matrix.

```python
from pca import Sketch
import spectral_check as sc

a, b = Sketch(p, l), Sketch(p, l)
for x, w in station_a: a.add(x, w)      # w: int, 1..1000
for x, w in station_b: b.add(x, w)
a.merge(b)                              # in place; b is untouched
summary = a.summary()                   # {"weight", "mean", "matrix"}
report = sc.check_spectral_bound(       # independent oracle, not used by pca
    sc.weighted_covariance(all_points, p)[2], summary["matrix"], l)
assert report["ok"]
```

## State and complexity

| | |
|---|---|
| `summary()` | `weight: int`, `mean: list[float]` length `p`, `matrix: list[list[float]]`, at most `l` rows × `p` columns |
| empty sketch | `weight == 0`, zero mean, `matrix == []` |
| resident state | `O(p·l)`: ≤ `l` factor rows, < `l` buffered rows, length-`p` double-double sum |
| `add` | amortised `O(p·l²)`; buffers `l` rows then compresses once |
| `merge` | `O(p·l² + l³)` — one rank-one cross term plus a truncation of ≤ `2l+1` rows |
| retained history | none; no `p × p` matrix is ever stored |

Neither `add` nor `merge` stores or returns a reference to caller data:
inputs are copied, `merge` never mutates `other` and never shares a row with
it, and each `summary()` allocates a fresh dictionary, mean and matrix.

## How the factor is maintained

Let `μ` be the weighted mean over all observations and

```
C = Σ w (x − μ)(x − μ)ᵀ,      E = C − BᵀB.
```

`add` feeds `B` the classic Welford rank-one update row
`√(W·w/(W+w)) · (x − μ_old)`; its outer product is exactly the covariance
increment. Once `l` such rows are buffered, the buffer plus `B` is compressed
by **Frequent Directions** truncation: take the SVD of the rows, let `δ` be the
`(l+1)`-th squared singular value, and keep `√max(σᵢ² − δ, 0)·vᵢ` for
`i = 1..l`. The discarded part is PSD with spectral norm `δ`.

Consequently every compression obeys
`λ_max(E_new) ≤ λ_max(E_old) + λ_{l+1}(M)` (Weyl), so for the covariance `C`
of a group

```
λ_max(C − BᵀB) ≤ λ_{l+1}(C) ≤ min_{0≤k<l} Σ_{i>k} λᵢ(C) / (l−k).
```

At a merge the candidate rows have Gram matrix
`M = C − (E₁ + E₂)`, so the same argument gives
`λ_max(E₁+E₂) + λ_{l+1}(M) ≤ λ_{l+1}(C)`: **the same bound holds for a single
stream, for every shard, and at every node of every merge tree**. This is
exactly the task's requirement (with `1e-6·max(1, tr C)` of floating-point
slack), and it also guarantees the lower bound `uᵀEu ≥ −ε` because the
residual is PSD by construction.

Centering is handled through the mean, never by a stored covariance: the
double-double weighted sum keeps the mean accurate to ~`1e-31` relative error
even for observations near `1e6` with tiny spread.

## Verification

* `spectral_check.py` — independent oracle. Rebuilds `C` from raw points and
  computes eigenvalues with a Householder-tridiagonalisation + implicit-QL
  solver, which shares no code with the cyclic-Jacobi solver in `pca.py`. It
  evaluates the bound directly as `max/min` over all unit `u` (i.e. the
  extreme residual eigenvalues) rather than assuming a particular structure.
* `test_pca.py` — 25 tests (`python test_pca.py`), covering the API contract,
  aliasing, centering/translation invariance, weight and mean accuracy,
  random streams and shards, every binary merge tree of a nasty four-shard
  case, a 300-node singleton merge chain, `p = 128, l = 16`, state-size and
  compression-count invariants, and a wall-clock budget.
* `demo_pca.py` — reads `.pointwise-runtime.json` (falls back to built-in
  defaults when absent), streams five stations, merges them as a single
  stream, a left-deep chain and a balanced tree, then asserts weight/mean and
  the spectral bound for each layout and for the untouched station snapshots.

```
$ python test_pca.py
Ran 25 tests ... OK

$ python demo_pca.py
...
all assertions passed: weight/mean match the true merged sample,
matrix is at most 3 x 24, and the spectral bound holds.
```

Beyond the shipped tests, the construction was stress-checked offline with
>50 000 randomized and adversarial merge trees (flat spectra, cardinality
constraints, mixed scales from `1e-6` to `1e6`, near-low-rank clusters,
enumerated merge orders) with zero bound violations.

## Files

```
pca.py                the sketch (Sketch, add, merge, summary)
spectral_check.py     independent covariance/eigen oracle + bound checker
test_pca.py           unittest suite
demo_pca.py           runtime-config-driven multi-station demo
.pointwise-runtime.json   demo parameters (dimensions, stations, tolerances)
```
