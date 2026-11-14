"""Streaming and mergeable PCA sketch, implemented with the Python standard library only.

The sketch keeps
  * the total weight ``W`` and a compensated weighted sum of the observations,
    i.e. the weighted mean accurate to roughly ``1e-31`` relative error, and
  * a factor ``B`` with at most ``l`` rows of dimension ``p`` whose Gram matrix
    ``B^T B`` approximates the weighted covariance ``C`` of every observation
    ever added.

The factor is maintained with Frequent-Directions (FD) truncation: given rows
``R``, take their SVD, set ``delta`` to the ``(l+1)``-th squared singular value
and keep ``sqrt(max(sigma_i^2 - delta, 0)) * v_i`` for ``i = 1..l``.  The
discarded part is positive semidefinite with spectral norm exactly ``delta``.
Because the virtual rows produced while streaming are exactly the Welford
rank-one covariance increments, the residual ``C - B^T B`` stays PSD and its
largest eigenvalue stays below the ``(l+1)``-th largest eigenvalue of whatever
matrix was most recently compressed.  Writing ``E = C - B^T B``, every
compression satisfies

    lambda_max(E) <= lambda_{l+1}(C) <= min_{0<=k<l} sum_{i>k} lambda_i(C) / (l-k)

for a single stream, for a shard, and for every node of every merge tree: at a
merge the rows fed to the compactor have Gram matrix ``M = C - (E1 + E2)``, so
Weyl's inequality gives

    lambda_max(E1 + E2) + lambda_{l+1}(M) <= lambda_{l+1}(C).

Resident state is ``O(p * l)``.  ``add`` is amortised ``O(p * l^2)`` and
``merge`` is ``O(p * l^2 + l^3)`` (within the promised ``O(p * l^3)``).  No
``p x p`` covariance matrix and no observation history is retained.
"""

from __future__ import annotations

import math

__all__ = ["Sketch"]


# ---------------------------------------------------------------------------
# Compensated (double-double) accumulation helpers
# ---------------------------------------------------------------------------


def _two_sum(a, b):
    """Error-free transformation: return ``(s, e)`` with ``a + b == s + e``."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def _dd_add(hi, lo, value):
    """Add ``value`` to the double-double number ``hi + lo``."""
    s, e = _two_sum(hi, value)
    lo = lo + e
    hi = s + lo
    lo = lo - (hi - s)
    return hi, lo


def _dd_merge(h1, l1, h2, l2):
    """Add double-double numbers ``h1 + l1`` and ``h2 + l2``."""
    s, e = _two_sum(h1, h2)
    lo = (l1 + l2) + e
    hi = s + lo
    lo = lo - (hi - s)
    return hi, lo


# ---------------------------------------------------------------------------
# Tiny dense linear algebra (matrices here have at most 2*l+1 <= 33 rows)
# ---------------------------------------------------------------------------


def _gram(rows):
    """Gram matrix ``R R^T`` of a list of equal-length rows."""
    m = len(rows)
    g = [[0.0] * m for _ in range(m)]
    for a in range(m):
        ra = rows[a]
        ga = g[a]
        for b in range(a, m):
            rb = rows[b]
            s = 0.0
            for k in range(len(ra)):
                s += ra[k] * rb[k]
            ga[b] = s
            g[b][a] = s
    return g


def _jacobi_eigh(a, max_sweeps=24):
    """Symmetric eigendecomposition by cyclic Jacobi rotations.

    Returns ``(eigenvalues, eigenvectors)`` sorted by decreasing eigenvalue;
    ``eigenvectors[i]`` is a unit eigenvector of ``eigenvalues[i]``.  The input
    matrix is never modified.
    """
    n = len(a)
    if n == 0:
        return [], []
    mat = [row[:] for row in a]
    vec = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = 0.0
        diag = 0.0
        for i in range(n):
            diag += mat[i][i] * mat[i][i]
            row = mat[i]
            for j in range(i + 1, n):
                off += row[j] * row[j]
        if off == 0.0 or off <= 1e-30 * diag:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = mat[p][q]
                if apq == 0.0:
                    continue
                tau = (mat[q][q] - mat[p][p]) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    akp = mat[k][p]
                    akq = mat[k][q]
                    mat[k][p] = c * akp - s * akq
                    mat[k][q] = s * akp + c * akq
                for k in range(n):
                    apk = mat[p][k]
                    aqk = mat[q][k]
                    mat[p][k] = c * apk - s * aqk
                    mat[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp = vec[k][p]
                    vkq = vec[k][q]
                    vec[k][p] = c * vkp - s * vkq
                    vec[k][q] = s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: mat[i][i], reverse=True)
    return ([mat[i][i] for i in order],
            [[vec[k][i] for k in range(n)] for i in order])


def _frequent_directions(rows, l):
    """Compress ``rows`` to at most ``l`` rows with FD truncation.

    The returned list and each of its rows are freshly allocated, so they never
    alias the input.  If ``rows`` has at most ``l`` entries it is copied
    verbatim.
    """
    m = len(rows)
    if m <= l:
        return [list(r) for r in rows]
    p = len(rows[0])
    values, vectors = _jacobi_eigh(_gram(rows))
    delta = values[l] if values[l] > 0.0 else 0.0
    out = []
    for i in range(l):
        lam = values[i]
        reduced = lam - delta
        if lam <= 0.0 or reduced <= 0.0:
            out.append([0.0] * p)
            continue
        # rows^T u_i is sqrt(lam) * v_i; rescale to sqrt(lam - delta) * v_i.
        direction = [0.0] * p
        u = vectors[i]
        for a in range(m):
            ua = u[a]
            if ua == 0.0:
                continue
            ra = rows[a]
            for k in range(p):
                direction[k] += ua * ra[k]
        scale = math.sqrt(reduced / lam)
        out.append([v * scale for v in direction])
    return out


# ---------------------------------------------------------------------------
# The sketch
# ---------------------------------------------------------------------------


class Sketch:
    """Fixed-shape, mergeable weighted covariance sketch.

    Parameters
    ----------
    p : int
        Number of dimensions.
    l : int
        Number of retained factor rows (the summary matrix has at most ``l``
        rows).
    """

    __slots__ = ("_p", "_l", "_w", "_hi", "_lo", "_b", "_pending")

    def __init__(self, p, l):
        p = int(p)
        l = int(l)
        if p <= 0:
            raise ValueError("p must be positive")
        if l <= 0:
            raise ValueError("l must be positive")
        if l > p:
            raise ValueError("l must not exceed p")
        self._p = p
        self._l = l
        self._w = 0                     # exact integer total weight
        self._hi = [0.0] * p            # double-double weighted sums ...
        self._lo = [0.0] * p            # ... exact value ~= hi + lo
        self._b = []                    # at most l factor rows (owned)
        self._pending = []              # fewer than l un-compressed rows

    # -- read-only introspection -------------------------------------------

    @property
    def p(self):
        return self._p

    @property
    def l(self):
        return self._l

    @property
    def weight(self):
        return self._w

    def __repr__(self):
        return "Sketch(p=%d, l=%d, weight=%d, rows=%d)" % (
            self._p, self._l, self._w, len(self._b) + len(self._pending))

    # -- internal helpers ---------------------------------------------------

    def _mean(self):
        """Weighted mean with double-double accuracy (fresh list)."""
        w = self._w
        return [math.fsum((self._hi[k], self._lo[k])) / w
                for k in range(self._p)]

    def _compress_pending(self):
        """Fold buffered rows into the factor (mutates ``self`` only)."""
        if self._pending:
            self._b = _frequent_directions(self._b + self._pending, self._l)
            self._pending = []

    # -- public API ---------------------------------------------------------

    def add(self, x, w):
        """Add observation ``x`` (a length-``p`` sequence) with weight ``w``."""
        p = self._p
        xv = [float(x[k]) for k in range(p)]
        w = int(w)
        w_old = self._w
        w_new = w_old + w

        # Welford rank-one update of the weighted covariance: the row
        # sqrt(W*w/(W+w)) * (x - mu_old) has outer product equal to the
        # covariance increment.  It is computed against the mean *before*
        # this observation is folded in.
        if w_old > 0:
            mean = self._mean()
            alpha = (float(w_old) * float(w)) / float(w_new)
            scale = math.sqrt(alpha)
            self._pending.append([scale * (xv[k] - mean[k]) for k in range(p)])

        hi = self._hi
        lo = self._lo
        for k in range(p):
            hi[k], lo[k] = _dd_add(hi[k], lo[k], w * xv[k])
        self._w = w_new

        if len(self._pending) >= self._l:
            self._compress_pending()

    def merge(self, other):
        """Merge ``other`` into ``self`` in place; ``other`` is left untouched.

        Both sketches must have been created with the same ``p`` and ``l``.
        """
        if not isinstance(other, Sketch):
            raise TypeError("merge expects a Sketch")
        if other._p != self._p or other._l != self._l:
            raise ValueError("sketches must share the same p and l")
        w2 = other._w
        if w2 == 0:
            return self
        w1 = self._w
        if w1 == 0:
            # Take private copies of everything; no alias with ``other``.
            self._w = w2
            self._hi = list(other._hi)
            self._lo = list(other._lo)
            self._b = [list(r) for r in other._b]
            self._pending = [list(r) for r in other._pending]
            return self

        p = self._p
        self._compress_pending()
        # Finalise a private copy of the other factor (never touch ``other``).
        if other._pending:
            b2 = _frequent_directions(
                [list(r) for r in other._b] + [list(r) for r in other._pending],
                self._l)
        else:
            b2 = [list(r) for r in other._b]

        mean1 = self._mean()
        mean2 = other._mean()
        w_sum = w1 + w2
        # Decomposition C = C1 + C2 + (w1*w2/W) (mu1-mu2)(mu1-mu2)^T: the
        # between-group covariance is exactly one rank-one row.
        cross_scale = math.sqrt((float(w1) * float(w2)) / float(w_sum))
        cross = [cross_scale * (mean1[k] - mean2[k]) for k in range(p)]

        candidates = [list(r) for r in self._b] + b2 + [cross]
        self._b = _frequent_directions(candidates, self._l)
        self._pending = []

        hi = [0.0] * p
        lo = [0.0] * p
        for k in range(p):
            hi[k], lo[k] = _dd_merge(self._hi[k], self._lo[k],
                                     other._hi[k], other._lo[k])
        self._hi = hi
        self._lo = lo
        self._w = w_sum
        return self

    def summary(self):
        """Return ``{"weight", "mean", "matrix"}`` as freshly allocated data.

        ``matrix`` has at most ``l`` rows and exactly ``p`` columns; it is the
        float factor ``B`` with ``B^T B`` approximating the weighted covariance.
        The empty sketch yields ``weight == 0``, a zero mean and ``matrix == []``.
        """
        p = self._p
        if self._w == 0:
            return {"weight": 0, "mean": [0.0] * p, "matrix": []}
        mean = self._mean()
        rows = [list(r) for r in self._b] + [list(r) for r in self._pending]
        matrix = _frequent_directions(rows, self._l)
        return {"weight": self._w, "mean": mean, "matrix": matrix}
