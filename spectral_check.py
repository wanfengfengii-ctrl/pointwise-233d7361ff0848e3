"""Independent spectral verification for :mod:`pca`.

Nothing here is used by the sketch itself; it exists so that tests can check
the sketch against ground truth computed by a completely different route:

* exact weighted covariance ``C`` rebuilt from the raw observations,
* eigenvalues of ``C`` and of the residual ``E = C - B^T B`` computed with a
  Householder-tridiagonalisation + implicit-QL eigensolver (``eigvalsh``),
  which shares no code with the cyclic-Jacobi solver inside ``pca``.

The verification helpers deliberately materialise the ``p x p`` matrices --
they are the test oracle, not part of the streaming sketch.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Symmetric eigensolver: Householder reduction (tred2) + implicit QL (tql2)
# ---------------------------------------------------------------------------


def _tred2(a):
    n = len(a)
    v = [row[:] for row in a]
    d = [0.0] * n
    e = [0.0] * n
    for j in range(n):
        d[j] = v[n - 1][j]
    for i in range(n - 1, 0, -1):
        scale = 0.0
        h = 0.0
        for k in range(i):
            scale += abs(d[k])
        if scale == 0.0:
            e[i] = d[i - 1]
            for j in range(i):
                d[j] = v[i - 1][j]
                v[i][j] = 0.0
                v[j][i] = 0.0
        else:
            for k in range(i):
                d[k] /= scale
                h += d[k] * d[k]
            f = d[i - 1]
            g = math.sqrt(h)
            if f > 0.0:
                g = -g
            e[i] = scale * g
            h -= f * g
            d[i - 1] = f - g
            for j in range(i):
                e[j] = 0.0
            for j in range(i):
                f = d[j]
                v[j][i] = f
                g = e[j] + v[j][j] * f
                for k in range(j + 1, i):
                    g += v[k][j] * d[k]
                    e[k] += v[k][j] * f
                e[j] = g
            f = 0.0
            for j in range(i):
                e[j] /= h
                f += e[j] * d[j]
            hh = f / (h + h)
            for j in range(i):
                e[j] -= hh * d[j]
            for j in range(i):
                f = d[j]
                g = e[j]
                for k in range(j, i):
                    v[k][j] -= f * e[k] + g * d[k]
                d[j] = v[i - 1][j]
                v[i][j] = 0.0
        d[i] = h
    for i in range(n - 1):
        v[n - 1][i] = v[i][i]
        v[i][i] = 1.0
        h = d[i + 1]
        if h != 0.0:
            for k in range(i + 1):
                d[k] = v[k][i + 1] / h
            for j in range(i + 1):
                g = 0.0
                for k in range(i + 1):
                    g += v[k][i + 1] * v[k][j]
                for k in range(i + 1):
                    v[k][j] -= g * d[k]
        for k in range(i + 1):
            v[k][i + 1] = 0.0
    for j in range(n):
        d[j] = v[n - 1][j]
        v[n - 1][j] = 0.0
    v[n - 1][n - 1] = 1.0
    e[0] = 0.0
    return d, e, v


def _tql2(d, e, v):
    n = len(d)
    for i in range(1, n):
        e[i - 1] = e[i]
    e[n - 1] = 0.0
    f = 0.0
    tst1 = 0.0
    eps = 2.0 ** -52
    for l in range(n):
        tst1 = max(tst1, abs(d[l]) + abs(e[l]))
        m = l
        while m < n - 1 and tst1 + abs(e[m]) != tst1:
            m += 1
        if m > l:
            it = 0
            while True:
                it += 1
                g = d[l]
                p = (d[l + 1] - g) / (2.0 * e[l])
                r = math.hypot(p, 1.0)
                if p < 0.0:
                    r = -r
                d[l] = e[l] / (p + r)
                d[l + 1] = e[l] * (p + r)
                dl1 = d[l + 1]
                h = g - d[l]
                for i in range(l + 2, n):
                    d[i] -= h
                f += h
                p = d[m]
                c = 1.0
                c2 = c
                c3 = c
                el1 = e[l + 1]
                s = 0.0
                s2 = 0.0
                for i in range(m - 1, l - 1, -1):
                    c3 = c2
                    c2 = c
                    s2 = s
                    g = c * e[i]
                    h = c * p
                    r = math.hypot(p, e[i])
                    e[i + 1] = s * r
                    s = e[i] / r
                    c = p / r
                    p = c * d[i] - s * g
                    d[i + 1] = h + s * (c * g + s * d[i])
                    for k in range(n):
                        h = v[k][i + 1]
                        v[k][i + 1] = s * v[k][i] + c * h
                        v[k][i] = c * v[k][i] - s * h
                p = -s * s2 * c3 * el1 * e[l] / dl1
                e[l] = s * p
                d[l] = c * p
                if not (abs(e[l]) > eps * tst1 and it < 100):
                    break
        d[l] += f
        e[l] = 0.0
    return d, v


def eigvalsh(a):
    """Eigenvalues (ascending) of a symmetric matrix given as list of rows."""
    n = len(a)
    if n == 0:
        return []
    d, e, v = _tred2(a)
    d, v = _tql2(d, e, v)
    return sorted(d)


# ---------------------------------------------------------------------------
# Ground-truth covariance and bound checks
# ---------------------------------------------------------------------------


def weighted_covariance(points, p):
    """Exact ``(W, mu, C)`` for ``points = [(x, w), ...]``.

    ``mu = sum w x / W`` (computed with :func:`math.fsum` per coordinate) and
    ``C = sum w (x - mu)(x - mu)^T``.
    """
    w_total = 0
    columns = [[] for _ in range(p)]
    for x, w in points:
        w_total += int(w)
        for k in range(p):
            columns[k].append(float(w) * float(x[k]))
    if w_total == 0:
        return 0, [0.0] * p, [[0.0] * p for _ in range(p)]
    mu = [math.fsum(columns[k]) / w_total for k in range(p)]
    cov = [[0.0] * p for _ in range(p)]
    for x, w in points:
        d = [float(x[k]) - mu[k] for k in range(p)]
        wf = float(w)
        for a in range(p):
            da = wf * d[a]
            row = cov[a]
            for b in range(a, p):
                row[b] += da * d[b]
    for a in range(p):
        for b in range(a):
            cov[a][b] = cov[b][a]
    return w_total, mu, cov


def gram(rows, p):
    """``B^T B`` for ``B`` given as a list of rows."""
    out = [[0.0] * p for _ in range(p)]
    for r in rows:
        for a in range(p):
            ra = r[a]
            row = out[a]
            for b in range(a, p):
                row[b] += ra * r[b]
    for a in range(p):
        for b in range(a):
            out[a][b] = out[b][a]
    return out


def symmetric_part(m):
    n = len(m)
    out = [row[:] for row in m]
    for a in range(n):
        for b in range(a):
            s = 0.5 * (out[a][b] + out[b][a])
            out[a][b] = out[b][a] = s
    return out


def check_spectral_bound(cov, matrix, l, eps_rel=1e-6):
    """Verify the task's spectral bound for factor ``matrix`` against ``cov``.

    Returns a dict with the residual eigenvalues, the tail-averaged right-hand
    side and the pass/fail verdict.
    """
    p = len(cov)
    lam = eigvalsh(cov)[::-1]
    trace = math.fsum(max(0.0, v) for v in lam)
    eps = eps_rel * max(1.0, trace)

    if matrix:
        approx = gram(matrix, p)
    else:
        approx = [[0.0] * p for _ in range(p)]
    residual = symmetric_part(
        [[cov[a][b] - approx[a][b] for b in range(p)] for a in range(p)])
    res_eigs = eigvalsh(residual)
    lhs_max = res_eigs[-1] if res_eigs else 0.0
    lhs_min = res_eigs[0] if res_eigs else 0.0

    # rhs(k) = sum_{i>k} lambda_i / (l-k), 1-indexed lambda, 0 <= k < l
    suffix = [0.0] * (p + 1)
    for i in range(p - 1, -1, -1):
        suffix[i] = suffix[i + 1] + lam[i]
    rhs = min(suffix[k] / (l - k) for k in range(l)) if l > 0 else 0.0

    upper_ok = lhs_max <= rhs + eps
    lower_ok = lhs_min >= -eps
    return {
        "ok": upper_ok and lower_ok,
        "upper_ok": upper_ok,
        "lower_ok": lower_ok,
        "lhs_max": lhs_max,       # max_u u^T E u
        "lhs_min": lhs_min,       # min_u u^T E u
        "rhs": rhs,
        "eps": eps,
        "trace": trace,
        "eigenvalues": lam,
        "residual_eigenvalues": res_eigs,
    }
