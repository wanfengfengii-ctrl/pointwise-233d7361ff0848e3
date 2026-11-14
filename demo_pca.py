"""Demonstration: merge per-station PCA sketches and assert the guarantees.

The demo reads ``.pointwise-runtime.json`` (next to this file) for the
dimension, the sketch width, the RNG seed and the per-station generator
parameters.  If the file is missing or incomplete, built-in defaults are used,
so the script always runs.

Each station streams its observations into its own :class:`pca.Sketch`; the
station sketches are then merged pairwise in a few different trees.  For every
result the demo recomputes the exact weighted mean and covariance from the raw
observations (the oracle lives only in this script, never in the sketch) and
asserts:

* ``weight`` and ``mean`` match the true merged sample,
* the summary matrix has at most ``l`` rows and ``p`` columns,
* the spectral bound  -eps <= u^T E u <= min_k sum_{i>k} lam_i / (l-k) + eps
  holds for every unit ``u``, with ``eps = 1e-6 * max(1, tr C)``.

Run with ``python demo_pca.py``; exit status is non-zero if an assertion fails.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pca  # noqa: E402
import spectral_check  # noqa: E402
from pca import Sketch  # noqa: E402

RUNTIME_FILE = os.path.join(HERE, ".pointwise-runtime.json")

DEFAULTS = {
    "p": 24,
    "l": 3,
    "seed": 20261114,
    "stations": [
        {"name": "north-1", "n": 120, "scale": 1.0, "offset_scale": 120.0},
        {"name": "north-2", "n": 90, "scale": 4.0, "offset_scale": 0.0},
        {"name": "south-1", "n": 60, "scale": 0.25, "offset_scale": -300.0},
        {"name": "south-2", "n": 200, "scale": 2.0, "offset_scale": 75.0},
        {"name": "east-1", "n": 45, "scale": 10.0, "offset_scale": 0.0},
    ],
    "assertions": {
        "mean_tolerance": 1e-7,
        "spectral_tolerance": 1e-6,
        "max_demo_p": 32,
    },
}


def load_runtime(path=RUNTIME_FILE):
    config = json.loads(json.dumps(DEFAULTS))  # deep copy of the defaults
    source = "built-in defaults"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError) as exc:
            print("warning: cannot read %s (%s); using defaults" % (path, exc))
            return config, source
        if isinstance(loaded, dict):
            for key in ("p", "l", "seed", "stations", "assertions"):
                if key in loaded:
                    config[key] = loaded[key]
            source = path
    # Sanitise.
    p = int(config["p"])
    p = max(2, min(p, int(config["assertions"].get("max_demo_p", 32))))
    l = max(2, min(int(config["l"]), min(16, p)))
    config["p"] = p
    config["l"] = l
    config["seed"] = int(config["seed"])
    stations = [s for s in config.get("stations", []) if isinstance(s, dict)]
    if not stations:
        stations = json.loads(json.dumps(DEFAULTS["stations"]))
    config["stations"] = stations
    return config, source


def make_station_points(rng, p, station, shared_basis):
    """Synthetic station data: low-rank signal plus dense noise."""
    n = int(station.get("n", 50))
    scale = float(station.get("scale", 1.0))
    offset_scale = float(station.get("offset_scale", 0.0))
    offset = [offset_scale * math.sin(0.7 * k + 0.3) for k in range(p)]
    weights = [1, 1, 2, 5, 25]
    pts = []
    for _ in range(n):
        x = [offset[k] + scale * rng.gauss(0.0, 1.0) for k in range(p)]
        for j, direction in enumerate(shared_basis):
            amp = scale * (3.0 / (j + 1)) * rng.gauss(0.0, 1.0)
            x = [x[k] + amp * direction[k] for k in range(p)]
        pts.append((x, rng.choice(weights)))
    return pts


def random_basis(rng, p, rank):
    basis = []
    for _ in range(rank):
        v = [rng.gauss(0.0, 1.0) for _ in range(p)]
        for w in basis:
            dot = math.fsum(a * b for a, b in zip(v, w))
            v = [a - dot * b for a, b in zip(v, w)]
        norm = math.sqrt(math.fsum(a * a for a in v)) or 1.0
        basis.append([a / norm for a in v])
    return basis


def verify_summary(label, pts, summary, p, l, mean_tol, spectral_tol):
    weight, mean, matrix = summary["weight"], summary["mean"], summary["matrix"]
    true_weight, true_mean, cov = spectral_check.weighted_covariance(pts, p)

    assert weight == true_weight, (
        "%s: weight %r != %r" % (label, weight, true_weight))
    for k in range(p):
        tol = mean_tol * max(1.0, abs(true_mean[k]))
        assert abs(mean[k] - true_mean[k]) <= tol, (
            "%s: mean[%d] %r vs %r (tol %r)"
            % (label, k, mean[k], true_mean[k], tol))
    assert len(matrix) <= l, "%s: matrix has %d rows > l=%d" % (
        label, len(matrix), l)
    for row in matrix:
        assert len(row) == p, "%s: row width %d != p=%d" % (
            label, len(row), p)

    report = spectral_check.check_spectral_bound(
        cov, matrix, l, eps_rel=spectral_tol)
    assert report["ok"], (
        "%s: spectral bound violated: max=%r min=%r rhs=%r eps=%r"
        % (label, report["lhs_max"], report["lhs_min"], report["rhs"],
           report["eps"]))
    return summary, report


def verify(label, pts, sketch, p, l, mean_tol, spectral_tol):
    return verify_summary(label, pts, sketch.summary(), p, l, mean_tol,
                          spectral_tol)


def main():
    config, source = load_runtime()
    p, l = config["p"], config["l"]
    rng = random.Random(config["seed"])
    basis = random_basis(rng, p, max(1, min(l, 3)))
    mean_tol = float(config["assertions"].get("mean_tolerance", 1e-7))
    spectral_tol = float(config["assertions"].get("spectral_tolerance", 1e-6))

    print("runtime config: %s" % source)
    print("dimension p=%d, sketch width l=%d, seed=%d"
          % (p, l, config["seed"]))

    station_pts = []
    station_sketches = []
    station_summaries = []
    for station in config["stations"]:
        pts = make_station_points(rng, p, station, basis)
        sketch = Sketch(p, l)
        for x, w in pts:
            sketch.add(x, w)
        station_pts.append(pts)
        station_sketches.append(sketch)
        station_summaries.append(sketch.summary())
        print("  station %-8s observations=%4d weight=%d rows=%d"
              % (station.get("name", "?"), len(pts), sketch.weight,
                 len(station_summaries[-1]["matrix"])))

    all_pts = [item for pts in station_pts for item in pts]
    results = []

    # 1. One combined stream.
    combined = Sketch(p, l)
    for x, w in all_pts:
        combined.add(x, w)
    results.append(("single stream", verify(
        "single stream", all_pts, combined, p, l, mean_tol, spectral_tol)))

    # 2. A left-deep chain of station merges.
    chain = Sketch(p, l)
    chain.merge(station_sketches[0])
    for sketch in station_sketches[1:]:
        chain.merge(sketch)
    results.append(("left-deep merge chain", verify(
        "left-deep merge chain", all_pts, chain, p, l, mean_tol,
        spectral_tol)))

    # 3. A balanced pairwise merge tree.  Each internal node is a fresh sketch
    #    that absorbs its two children, so the station sketches stay untouched.
    nodes = list(station_sketches)
    while len(nodes) > 1:
        merged = []
        for i in range(0, len(nodes) - 1, 2):
            parent = Sketch(p, l)
            parent.merge(nodes[i])
            parent.merge(nodes[i + 1])
            merged.append(parent)
        if len(nodes) % 2:
            merged.append(nodes[-1])
        nodes = merged
    results.append(("balanced merge tree", verify(
        "balanced merge tree", all_pts, nodes[0], p, l, mean_tol,
        spectral_tol)))

    # 4. The snapshots taken from the source stations are still valid, and
    #    the sources were never modified by the merges above.
    for station, pts, sketch, snapshot in zip(
            config["stations"], station_pts, station_sketches,
            station_summaries):
        name = station.get("name", "?")
        verify_summary("station %s snapshot" % name, pts, snapshot, p, l,
                       mean_tol, spectral_tol)
        assert sketch.summary() == snapshot, (
            "station %s was modified by a merge" % name)

    print()
    print("verified %d merge layouts; worst spectral slack per layout:"
          % len(results))
    for label, (summary, report) in results:
        top = report["eigenvalues"][:max(1, min(l, len(report["eigenvalues"])))]
        print("  %-24s rows=%2d weight=%6d  max u^T E u=%12.6g  "
              "bound=%12.6g  eps=%.3g  top eigenvalues=%s"
              % (label, len(summary["matrix"]), summary["weight"],
                 report["lhs_max"], report["rhs"] + report["eps"],
                 report["eps"], ", ".join("%.4g" % v for v in top)))

    print()
    print("all assertions passed: weight/mean match the true merged sample, "
          "matrix is at most %d x %d, and the spectral bound holds."
          % (l, p))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print("ASSERTION FAILED: %s" % exc, file=sys.stderr)
        sys.exit(1)
