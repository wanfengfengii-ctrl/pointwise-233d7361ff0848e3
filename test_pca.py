"""Tests for :mod:`pca` and the independent spectral checker.

Run with ``python -m unittest -v test_pca`` (or ``python test_pca.py``).
"""

import math
import os
import random
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pca
import spectral_check as sc
from pca import Sketch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_points(rng, p, n, weight_choices=(1, 2, 7, 1000),
                  scales=(1.0, 10.0, 1e4), shift=0.0):
    pts = []
    for _ in range(n):
        scale = rng.choice(scales)
        x = [shift + rng.gauss(0.0, 1.0) * scale for _ in range(p)]
        pts.append((x, rng.choice(weight_choices)))
    return pts


def build(pts, p, l):
    sk = Sketch(p, l)
    for x, w in pts:
        sk.add(x, w)
    return sk


def build_shards(pts, p, l, k, rng):
    """Split points over ``k`` sketches then merge them in random order."""
    shards = [Sketch(p, l) for _ in range(k)]
    for x, w in pts:
        shards[rng.randrange(k)].add(x, w)
    nodes = shards
    while len(nodes) > 1:
        i = rng.randrange(len(nodes))
        a = nodes.pop(i)
        j = rng.randrange(len(nodes))
        nodes[j].merge(a)
    return nodes[0]


def check_against_oracle(testcase, pts, matrix, weight, mean, l):
    p = len(mean)
    w_exact, mu, cov = sc.weighted_covariance(pts, p)
    testcase.assertEqual(weight, w_exact)
    for k in range(p):
        tol = 1e-7 * max(1.0, abs(mu[k]))
        testcase.assertLessEqual(abs(mean[k] - mu[k]), tol,
                                 "mean[%d] off by %g" % (k, mean[k] - mu[k]))
    testcase.assertLessEqual(len(matrix), l)
    for row in matrix:
        testcase.assertEqual(len(row), p)
        for v in row:
            testcase.assertTrue(isinstance(v, float))
    report = sc.check_spectral_bound(cov, matrix, l)
    testcase.assertTrue(
        report["ok"],
        "spectral bound failed: max u^T E u = %r > rhs %r + eps %r; "
        "min = %r" % (report["lhs_max"], report["rhs"], report["eps"],
                      report["lhs_min"]))
    return report


# ---------------------------------------------------------------------------
# Contract: shapes, empty state, aliasing
# ---------------------------------------------------------------------------


class TestContract(unittest.TestCase):
    def test_empty_summary(self):
        for p, l in [(2, 2), (5, 2), (16, 4), (128, 16)]:
            sk = Sketch(p, l)
            s = sk.summary()
            self.assertEqual(s["weight"], 0)
            self.assertEqual(s["mean"], [0.0] * p)
            self.assertEqual(s["matrix"], [])

    def test_matrix_shape_and_float(self):
        rng = random.Random(1)
        for p, l in [(2, 2), (3, 2), (10, 4), (128, 16)]:
            pts = random_points(rng, p, 50)
            s = build(pts, p, l).summary()
            self.assertIs(type(s["weight"]), int)
            self.assertLessEqual(len(s["matrix"]), l)
            self.assertEqual(len(s["mean"]), p)
            for row in s["matrix"]:
                self.assertEqual(len(row), p)
                for v in row:
                    self.assertIs(type(v), float)
                    self.assertTrue(math.isfinite(v))

    def test_no_alias_from_input(self):
        sk = Sketch(3, 2)
        x = [1.0, 2.0, 3.0]
        sk.add(x, 1)
        x[0] = -1e9
        self.assertEqual(sk.summary()["mean"], [1.0, 2.0, 3.0])

    def test_summary_is_fresh(self):
        sk = build([([1.0, 2.0, 3.0], 5)], 3, 2)
        s1 = sk.summary()
        s1["mean"][0] = 999.0
        s1["matrix"].append([9.0, 9.0, 9.0])
        s1["matrix"][0][0] = 1234.0
        s2 = sk.summary()
        self.assertEqual(s2["mean"], [1.0, 2.0, 3.0])
        self.assertEqual(len(s2["matrix"]), 0)  # one observation -> no spread

    def test_summary_results_are_independent(self):
        rng = random.Random(2)
        sk = build(random_points(rng, 4, 20), 4, 2)
        a = sk.summary()
        b = sk.summary()
        self.assertIsNot(a["mean"], b["mean"])
        self.assertIsNot(a["matrix"], b["matrix"])
        for ra, rb in zip(a["matrix"], b["matrix"]):
            self.assertIsNot(ra, rb)

    def test_merge_does_not_modify_other(self):
        rng = random.Random(3)
        p, l = 5, 3
        a = build(random_points(rng, p, 30), p, l)
        b = build(random_points(rng, p, 40), p, l)
        before = (b.summary(), [list(r) for r in b._b],
                  [list(r) for r in b._pending],
                  list(b._hi), list(b._lo), b._w)
        a.merge(b)
        after = (b.summary(), [list(r) for r in b._b],
                 [list(r) for r in b._pending],
                 list(b._hi), list(b._lo), b._w)
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1], after[1])
        self.assertEqual(before[2], after[2])
        self.assertEqual(before[3], after[3])
        self.assertEqual(before[4], after[4])
        self.assertEqual(before[5], after[5])
        # no shared mutable objects with the target
        for row_b in b._b:
            for row_a in a._b:
                self.assertIsNot(row_a, row_b)
        for x, y in zip(a._hi, b._hi):
            self.assertIsNot(x, y)

    def test_merge_empty_sketches(self):
        p, l = 4, 2
        empty = Sketch(p, l)
        empty.merge(Sketch(p, l))
        self.assertEqual(empty.summary(),
                         {"weight": 0, "mean": [0.0] * p, "matrix": []})
        rng = random.Random(4)
        full = build(random_points(rng, p, 12), p, l)
        expected = full.summary()
        full.merge(Sketch(p, l))
        self.assertEqual(full.summary(), expected)
        target = Sketch(p, l)
        target.merge(full)
        self.assertEqual(target.summary(), expected)
        # adopting state must not alias the source
        target.summary()["mean"][0] = 1e9
        self.assertEqual(full.summary(), expected)

    def test_merge_validation(self):
        with self.assertRaises(TypeError):
            Sketch(3, 2).merge("not a sketch")
        with self.assertRaises(ValueError):
            Sketch(3, 2).merge(Sketch(4, 2))
        with self.assertRaises(ValueError):
            Sketch(3, 2).merge(Sketch(3, 3))

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            Sketch(0, 0)
        with self.assertRaises(ValueError):
            Sketch(3, 0)
        with self.assertRaises(ValueError):
            Sketch(3, 4)

    def test_self_merge_is_sane(self):
        rng = random.Random(5)
        pts = random_points(rng, 3, 10)
        sk = build(pts, 3, 2)
        sk.merge(sk)
        doubled = [(x, 2 * w) for x, w in pts]
        w_exact, mu, cov = sc.weighted_covariance(doubled, 3)
        self.assertEqual(sk.summary()["weight"], w_exact)
        for k in range(3):
            self.assertAlmostEqual(sk.summary()["mean"][k], mu[k], places=12)


# ---------------------------------------------------------------------------
# Weight and mean accuracy, centering behaviour
# ---------------------------------------------------------------------------


class TestWeightAndMean(unittest.TestCase):
    def test_weight_and_mean_exact_random(self):
        rng = random.Random(10)
        for trial in range(40):
            p = rng.choice([2, 3, 8, 32])
            l = min(rng.choice([2, 3, 4]), p)
            pts = random_points(rng, p, rng.choice([0, 1, 5, 50, 400]),
                                 shift=rng.choice([0.0, 1e3, -1e5]))
            build(pts, p, l).summary()  # smoke
            sk = build(pts, p, l)
            s = sk.summary()
            check_against_oracle(self, pts, s["matrix"], s["weight"],
                                 s["mean"], l)

    def test_huge_offsets_tiny_spread(self):
        rng = random.Random(11)
        for p in (3, 16, 64):
            base = [rng.uniform(-1e6, 1e6) for _ in range(p)]
            pts = []
            for _ in range(500):
                x = [base[k] + rng.gauss(0.0, 1.0) for k in range(p)]
                pts.append((x, rng.randint(1, 1000)))
            s = build(pts, p, min(4, p)).summary()
            w, mu, _ = sc.weighted_covariance(pts, p)
            self.assertEqual(s["weight"], w)
            for k in range(p):
                self.assertLessEqual(abs(s["mean"][k] - mu[k]),
                                     1e-7 * max(1.0, abs(mu[k])))

    def test_exact_cancellation(self):
        pts = [([1e6, -1e6, 0.0, 0.0], 1), ([-1e6, 1e6, 0.0, 0.0], 1)]
        s = build(pts, 4, 2).summary()
        self.assertEqual(s["weight"], 2)
        self.assertEqual(s["mean"], [0.0, 0.0, 0.0, 0.0])

    def test_translation_invariance_of_factor(self):
        """Centering: shifting every observation must not move B."""
        rng = random.Random(12)
        p, l = 6, 3
        pts = random_points(rng, p, 300, scales=(0.5, 2.0))
        s0 = build(pts, p, l).summary()
        shift = [1e5 * (1 if k % 2 else -1) for k in range(p)]
        shifted = [( [x[k] + shift[k] for k in range(p)], w) for x, w in pts]
        s1 = build(shifted, p, l).summary()
        _, _, cov0 = sc.weighted_covariance(pts, p)
        _, _, cov1 = sc.weighted_covariance(shifted, p)
        tr = math.fsum(max(0.0, v) for v in sc.eigvalsh(cov0))
        for a in range(p):
            self.assertAlmostEqual(cov0[a][a], cov1[a][a],
                                   delta=1e-6 * max(1.0, tr))
        b0 = sc.gram(s0["matrix"], p)
        b1 = sc.gram(s1["matrix"], p)
        for a in range(p):
            for b in range(p):
                self.assertLessEqual(abs(b0[a][b] - b1[a][b]),
                                     1e-6 * max(1.0, tr))
        for k in range(p):
            self.assertAlmostEqual(s1["mean"][k] - shift[k], s0["mean"][k],
                                   delta=1e-7 * max(1.0, abs(s0["mean"][k])))


# ---------------------------------------------------------------------------
# Spectral bound
# ---------------------------------------------------------------------------


class TestSpectralBound(unittest.TestCase):
    def test_random_stream_and_shards(self):
        rng = random.Random(20)
        for trial in range(60):
            p = rng.choice([2, 3, 4, 5, 8, 12, 16, 32])
            l = min(rng.choice([2, 2, 3, 4]), p)
            n = rng.choice([1, 2, 3, 10, 60, 250])
            pts = random_points(rng, p, n,
                                 scales=rng.choice(
                                     [(1.0,), (1.0, 10.0), (1e-3, 1e3),
                                      (1.0, 1e6)]))
            s = build(pts, p, l).summary()
            check_against_oracle(self, pts, s["matrix"], s["weight"],
                                 s["mean"], l)
            k = rng.randint(2, 6)
            s = build_shards(pts, p, l, k, rng).summary()
            check_against_oracle(self, pts, s["matrix"], s["weight"],
                                 s["mean"], l)

    def test_cardinality_basis(self):
        """Unit basis vectors are the worst case for flat spectra."""
        for p, l in [(4, 2), (16, 2), (32, 3), (128, 2)]:
            pts = []
            for k in range(p):
                x = [0.0] * p
                x[k] = 1.0
                pts.append((x, 1))
            s = build(pts, p, l).summary()
            check_against_oracle(self, pts, s["matrix"], s["weight"],
                                 s["mean"], l)

    def test_all_merge_trees_small(self):
        """Explicitly enumerate every binary merge tree for a nasty case."""
        rng = random.Random(21)
        p, l = 4, 2
        groups = [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1e3, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1e-2]],
            [[1e2, 1e2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0]],
        ]
        pts = [(r, 1) for g in groups for r in g]
        import itertools
        for order in itertools.permutations(range(len(groups))):
            nodes = []
            for i in order:
                sk = Sketch(p, l)
                for r in groups[i]:
                    sk.add(r, 1)
                nodes.append(sk)
            while len(nodes) > 1:
                a = nodes.pop(0)
                nodes[0].merge(a)
            s = nodes[0].summary()
            check_against_oracle(self, pts, s["matrix"], s["weight"],
                                 s["mean"], l)

    def test_clustered_near_low_rank(self):
        rng = random.Random(22)
        p, l = 8, 2
        groups = []
        for shift in (0.0, 50.0, -80.0):
            rows = []
            for _ in range(25):
                base = [shift + rng.gauss(0.0, 3.0) for _ in range(p)]
                rows.append(base)
            groups.append(rows)
        pts = [(r, rng.randint(1, 100)) for g in groups for r in g]
        s = build_shards([(r, w) for r, w in pts], p, l, 3, rng).summary()
        check_against_oracle(self, pts, s["matrix"], s["weight"],
                             s["mean"], l)

    def test_rank_deficient_exact_when_large(self):
        """If the data rank is below l, the residual is essentially zero."""
        rng = random.Random(23)
        p, l = 10, 5
        basis = [[rng.gauss(0.0, 1.0) for _ in range(2)] for _ in range(p)]
        pts = []
        for _ in range(200):
            coeff = [rng.gauss(0.0, 4.0), rng.gauss(0.0, 1.0)]
            x = [basis[k][0] * coeff[0] + basis[k][1] * coeff[1]
                 for k in range(p)]
            pts.append((x, rng.randint(1, 50)))
        s = build(pts, p, l).summary()
        report = check_against_oracle(self, pts, s["matrix"], s["weight"],
                                       s["mean"], l)
        self.assertLessEqual(report["lhs_max"], 1e-8 * max(1.0, report["trace"]))
        self.assertGreaterEqual(report["lhs_min"], -1e-8 * max(1.0, report["trace"]))

    def test_large_dimension(self):
        """p = 128, l = 16 with ten stations and a merge tree."""
        rng = random.Random(24)
        p, l = 128, 16
        pts = random_points(rng, p, 400, scales=(1.0, 100.0, 1e4))
        sk = build_shards(pts, p, l, 10, rng)
        s = sk.summary()
        self.assertEqual(len(s["matrix"]), l)
        check_against_oracle(self, pts, s["matrix"], s["weight"],
                             s["mean"], l)

    def test_long_singleton_chain(self):
        """Error must not accumulate across a deep merge chain."""
        rng = random.Random(25)
        p, l = 5, 2
        pts = random_points(rng, p, 300)
        nodes = []
        for x, w in pts:
            sk = Sketch(p, l)
            sk.add(x, w)
            nodes.append(sk)
        head = nodes[0]
        for other in nodes[1:]:
            head.merge(other)
        s = head.summary()
        check_against_oracle(self, pts, s["matrix"], s["weight"],
                             s["mean"], l)


# ---------------------------------------------------------------------------
# Complexity / space invariants
# ---------------------------------------------------------------------------


class TestComplexity(unittest.TestCase):
    def test_state_size_stays_linear(self):
        rng = random.Random(30)
        p, l = 16, 4
        sk = Sketch(p, l)
        for i in range(4000):
            sk.add([rng.gauss(0.0, 1.0) for _ in range(p)], 1)
            rows = len(sk._b) + len(sk._pending)
            self.assertLessEqual(rows, 2 * l)
        self.assertLessEqual(len(sk._b), l)
        self.assertLessEqual(len(sk._pending), l)
        # No p x p structure is retained anywhere on the instance.
        for slot in Sketch.__slots__:
            value = getattr(sk, slot)
            if isinstance(value, list) and value and isinstance(value[0], list):
                self.assertLessEqual(len(value) * len(value[0]), 2 * l * p)

    def test_add_truncates_amortised(self):
        """A batch of N adds triggers at most ~N/l + 1 compressions."""
        calls = []
        original = pca._frequent_directions

        def counting(rows, l):
            calls.append(len(rows))
            return original(rows, l)

        pca._frequent_directions = counting
        try:
            rng = random.Random(31)
            p, l, n = 6, 4, 400
            sk = Sketch(p, l)
            for _ in range(n):
                sk.add([rng.gauss(0.0, 1.0) for _ in range(p)], 1)
        finally:
            pca._frequent_directions = original
        # +1 for the summary() call's own refresh
        self.assertLessEqual(len(calls), n // l + 2)
        self.assertTrue(all(m <= 2 * l for m in calls))

    def test_merge_does_constant_compressions(self):
        calls = []
        original = pca._frequent_directions

        def counting(rows, l):
            calls.append(len(rows))
            return original(rows, l)

        rng = random.Random(32)
        p, l = 8, 3
        a = build(random_points(rng, p, 20), p, l)
        b = build(random_points(rng, p, 20), p, l)
        pca._frequent_directions = counting
        try:
            a.merge(b)
        finally:
            pca._frequent_directions = original
        # At most three passes: flushing ``self``'s buffer, finalising a copy
        # of ``other``, and compressing the 2l+1 merged candidates.
        self.assertLessEqual(len(calls), 3)
        # Each pass touches at most 2l+1 rows, so merge is O(p l^2 + l^3).
        self.assertTrue(all(m <= 2 * l + 1 for m in calls))

    def test_runtime_budget(self):
        """Generous wall-clock sanity for the largest allowed shape."""
        rng = random.Random(33)
        p, l = 128, 16
        sk = Sketch(p, l)
        start = time.perf_counter()
        for _ in range(200):
            sk.add([rng.uniform(-1e6, 1e6) for _ in range(p)], 1)
        sk.summary()
        other = Sketch(p, l)
        for _ in range(50):
            other.add([rng.uniform(-1e6, 1e6) for _ in range(p)], 1)
        for _ in range(10):
            sk.merge(other)
        self.assertLess(time.perf_counter() - start, 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
