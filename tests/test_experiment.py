import json
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import f, ncf

from weak_iv import IVDesign
from weak_iv.experiment import estimate, fixed_instruments, run_experiment


def test_fixed_design_centering_orthogonality_and_reproducibility():
    z = fixed_instruments(200, 123)
    assert_allclose(z.mean(axis=0), 0, atol=1e-15)
    assert_allclose(z.T @ z, 200 * np.eye(2), atol=1e-12)
    assert np.array_equal(z, fixed_instruments(200, 123))


def test_monte_carlo_units_and_empty_conditional_group():
    values = np.array([0, 1, 0, 1], dtype=float)
    result = estimate(values)
    assert result["estimate"] == .5
    assert result["count"] == 4
    assert_allclose(result["mcse"], values.std(ddof=1) / 2)
    assert estimate([]) == {"estimate": None, "mcse": None, "count": 0}


def test_ar_coverage_under_gaussian_null_without_relevance():
    n, repetitions = 70, 2000
    design = IVDesign(fixed_instruments(n, 32))
    rng = np.random.default_rng(71)
    indicators = []
    for _ in range(repetitions):
        v, error = rng.normal(size=(2, n))
        x = v  # Excluded instruments have zero first-stage relevance.
        y = x + .8 * v + .6 * error
        indicators.append(design.fit(y, x).ar_confidence_set().contains(1))
    # Finite-Monte-Carlo check, not proof of the reference distribution.
    assert abs(np.mean(indicators) - .95) < 5 * np.sqrt(.95 * .05 / repetitions)


def test_invalid_exclusion_coverage_matches_noncentral_f_reference():
    n, repetitions, delta = 70, 2000, .35
    z = fixed_instruments(n, 32)
    design = IVDesign(z)
    rng = np.random.default_rng(93)
    indicators = []
    for _ in range(repetitions):
        v, error = rng.normal(size=(2, n))
        x = z[:, 0] + v
        y = x + delta * z[:, 1] + .8 * v + .6 * error
        indicators.append(design.fit(y, x).ar_confidence_set().contains(1))
    noncentrality = float((delta * z[:, 1]) @ (delta * z[:, 1]))
    expected = ncf.cdf(f.isf(.05, 2, n - 3), 2, n - 3, noncentrality)
    assert abs(np.mean(indicators) - expected) < 5 * np.sqrt(expected * (1 - expected) / repetitions)


def test_summary_counts_pairing_and_conditional_coverage():
    summary = run_experiment(replicates=100, n=50, seed=61)
    valid_rates = []
    for row in summary["scenarios"]:
        counts = row["ar_set_shape_counts"]
        assert sum(counts.values()) == 100
        assert row["paired_ar_minus_wald_coverage"]["count"] == 100
        conditional_total = sum(
            values["estimate"] * values["count"]
            for values in row["coverage_conditional_on_ar_shape"].values() if values["count"]
        ) / 100
        assert_allclose(conditional_total, row["ar_coverage"]["estimate"])
        if row["exclusion_restriction_valid_by_construction"]:
            valid_rates.append(row["ar_coverage"]["estimate"])
    # At the true beta the AR residual is the SAME u in every valid scenario.
    assert len(set(valid_rates)) == 1
    assert summary == run_experiment(replicates=100, n=50, seed=61)


@pytest.mark.parametrize("replicates", [0, 1, True, 2.5])
def test_invalid_replication_counts(replicates):
    with pytest.raises(ValueError):
        run_experiment(replicates=replicates)


@pytest.mark.parametrize("n", [0, 9, True, 10.5])
def test_invalid_sample_sizes(n):
    with pytest.raises(ValueError):
        fixed_instruments(n, 1)


def test_cli_outputs_valid_json_and_png(tmp_path):
    subprocess.run([
        sys.executable, "-W", "error", "-m", "weak_iv.experiment",
        "--replicates", "30", "--n", "40", "--seed", "31", "--output", str(tmp_path),
    ], check=True, capture_output=True, text=True)
    text = (tmp_path / "summary.json").read_text()
    assert "Infinity" not in text and "NaN" not in text
    result = json.loads(text)
    assert len(result["scenarios"]) == 5
    assert result["replicates"] == 30
    for name in ("coverage_and_geometry.png", "confidence_set_shapes.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
