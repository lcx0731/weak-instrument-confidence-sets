import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import f, norm

from weak_iv import IVDesign, solve_quadratic_leq


@pytest.mark.parametrize("coefficients,kind,intervals", [
    ((1, -3, 2), "bounded", ((1, 2),)),
    ((-1, 3, -2), "two_rays", ((-math.inf, 1), (2, math.inf))),
    ((1, 0, 1), "empty", ()),
    ((-1, 0, -1), "all_real", ((-math.inf, math.inf),)),
    ((1, -2, 1), "bounded", ((1, 1),)),
    ((-1, 2, -1), "all_real", ((-math.inf, math.inf),)),
    ((0, 2, -4), "left_ray", ((-math.inf, 2),)),
    ((0, -2, 4), "right_ray", ((2, math.inf),)),
    ((0, 0, 1), "empty", ()),
    ((0, 0, 0), "all_real", ((-math.inf, math.inf),)),
    ((0, 0, -1), "all_real", ((-math.inf, math.inf),)),
    ((1, 2, 0), "bounded", ((-2, 0),)),
])
def test_every_quadratic_geometry(coefficients, kind, intervals):
    result = solve_quadratic_leq(*coefficients)
    assert result.kind == kind
    assert result.intervals == intervals
    a, b, c = coefficients
    for value in np.linspace(-5, 5, 101):
        assert result.contains(value) == ((a * value + b) * value + c <= 0)


def test_small_nonzero_quadratic_coefficient_not_treated_as_linear():
    result = solve_quadratic_leq(1e-20, 1, 1)
    assert result.kind == "bounded"
    assert_allclose(result.intervals[0], (-1e20, -1), rtol=1e-15)
    assert not result.contains(-1e21)


def test_stable_roots_with_large_coefficient_range():
    result = solve_quadratic_leq(1, -1e150, 1)
    assert_allclose(result.intervals[0], (1e-150, 1e150), rtol=1e-15, atol=0)


def test_nearly_repeated_roots_retain_sign_of_discriminant():
    below = solve_quadratic_leq(1, -2, np.nextafter(1.0, 0))
    above = solve_quadratic_leq(1, -2, np.nextafter(1.0, 2))
    assert below.kind == "bounded"
    assert below.intervals[0][0] < 1 < below.intervals[0][1]
    assert above.kind == "empty"


@pytest.mark.parametrize("coefficients", [(math.nan, 0, 0), (1, math.inf, 1), (1e-320, 1, 1)])
def test_invalid_or_unrepresentable_quadratic(coefficients):
    with pytest.raises(ValueError):
        solve_quadratic_leq(*coefficients)


def test_safe_infinite_endpoint_encoding():
    result = solve_quadratic_leq(-1, 0, 1)
    assert result.to_dict() == {"kind": "two_rays", "unbounded": True, "intervals": [[None, -1.0], [1.0, None]]}
    assert solve_quadratic_leq(1, 0, 1).to_dict()["intervals"] == []
    with pytest.raises(ValueError):
        result.contains(math.inf)


def sample(seed=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(90, 2))
    controls = rng.normal(size=(90, 1))
    v, eta = rng.normal(size=(2, 90))
    x = .7 * z[:, 0] + .3 * controls[:, 0] + v
    y = 1.2 * x + controls[:, 0] + .6 * v + .8 * eta
    return z, controls, y, x


def test_ar_f_matches_independent_nested_regressions():
    z, controls, y, x = sample()
    design = IVDesign(z, controls)
    fitted = design.fit(y, x)
    restricted = np.column_stack((np.ones(len(x)), controls))
    unrestricted = np.column_stack((restricted, z))
    for beta in (-3, 0, 1.2, 2, 8):
        response = y - beta * x
        e_restricted = response - restricted @ np.linalg.lstsq(restricted, response, rcond=None)[0]
        e_full = response - unrestricted @ np.linalg.lstsq(unrestricted, response, rcond=None)[0]
        rss0, rss1 = e_restricted @ e_restricted, e_full @ e_full
        expected = ((rss0 - rss1) / 2) / (rss1 / (len(x) - 4))
        actual = fitted.ar_test(beta)
        assert_allclose(actual["statistic"], expected, rtol=1e-12)
        assert_allclose(actual["p_value"], f.sf(expected, 2, len(x) - 4), rtol=1e-12)


def test_wald_matches_two_stage_matrix_reference():
    z, controls, y, x = sample()
    w = np.column_stack((np.ones(len(x)), controls))
    instruments = np.column_stack((w, z))
    xhat = instruments @ np.linalg.lstsq(instruments, x, rcond=None)[0]
    second_stage = np.column_stack((w, xhat))
    reference = np.linalg.lstsq(second_stage, y, rcond=None)[0]
    structural_regressors = np.column_stack((w, x))
    residual = y - structural_regressors @ reference
    sigma2 = (residual @ residual) / (len(x) - 3)
    e_slope = np.array([0, 0, 1])
    se = np.sqrt(sigma2 * np.linalg.solve(second_stage.T @ second_stage, e_slope)[-1])
    result = IVDesign(z, controls).fit(y, x).wald()
    assert_allclose(result["estimate"], reference[-1], atol=1e-13)
    assert_allclose(result["standard_error"], se, atol=1e-13)
    assert_allclose(result["interval"], reference[-1] + np.array([-1, 1]) * norm.isf(.025) * se, atol=1e-13)


@pytest.mark.parametrize("seed", [2, 17, 42, 99])
def test_analytic_inversion_matches_test_acceptance(seed):
    z, controls, y, x = sample(seed)
    fitted = IVDesign(z, controls).fit(y, x)
    result = fitted.ar_confidence_set()
    for beta in np.linspace(-20, 20, 301):
        assert result.contains(beta) == fitted.ar_test(beta)["accepted"]
    for interval in result.intervals:
        for endpoint in interval:
            if math.isfinite(endpoint):
                test = fitted.ar_test(endpoint)
                assert_allclose(test["statistic"], test["critical_value"], rtol=1e-10)


def test_instrument_basis_and_nuisance_shift_invariance():
    z, controls, y, x = sample()
    base = IVDesign(z, controls).fit(y, x)
    changed = IVDesign(z @ [[2, .3], [.1, .6]], controls).fit(
        y + 2 + 3 * controls[:, 0], x - 4 + .7 * controls[:, 0]
    )
    assert_allclose(base.ar_coefficients(), changed.ar_coefficients(), atol=1e-12)
    assert_allclose(base.ar_confidence_set().intervals, changed.ar_confidence_set().intervals, atol=1e-12)
    assert_allclose(base.wald()["estimate"], changed.wald()["estimate"], atol=1e-12)


def test_outcome_shift_by_endogenous_shifts_confidence_set():
    z, controls, y, x = sample()
    design = IVDesign(z, controls)
    before = design.fit(y, x).ar_confidence_set()
    after = design.fit(y + 3 * x, x).ar_confidence_set()
    assert before.kind == after.kind == "bounded"
    assert_allclose(np.array(before.intervals) + 3, after.intervals, atol=1e-12)


def test_first_stage_f_and_unboundedness_threshold_agree():
    z, controls, y, x = sample()
    rng = np.random.default_rng(1)
    design = IVDesign(z, controls)
    for strength in (0, .02, .1, .4, 1, 3):
        fitted = design.fit(y, rng.normal(size=len(y)) + strength * z[:, 0])
        a = fitted.ar_coefficients()[0]
        threshold = f.isf(.05, design.k, fitted.residual_df)
        assert (a < 0) == (fitted.first_stage_f() < threshold)
        assert fitted.ar_confidence_set().unbounded == (a < 0)


def test_zero_variation_is_explicit():
    z, _, _, _ = sample()
    fitted = IVDesign(z).fit(np.zeros(len(z)), np.zeros(len(z)))
    assert fitted.ar_confidence_set().kind == "all_real"
    assert fitted.ar_test(1)["degenerate_residual"]
    assert fitted.ar_test(1)["accepted"]
    with pytest.raises(ValueError, match="zero fitted"):
        fitted.wald()


@pytest.mark.parametrize("alpha", [-.1, 0, 1, 1.1, math.nan])
def test_invalid_alpha(alpha):
    z, controls, y, x = sample()
    fitted = IVDesign(z, controls).fit(y, x)
    for method in (fitted.ar_confidence_set, fitted.wald):
        with pytest.raises(ValueError):
            method(alpha)


def test_rank_and_dimensions_are_checked():
    z, controls, y, x = sample()
    invalid = [np.ones((90, 1)), np.column_stack((z, z[:, 0])), np.ones((2, 2)), np.array([[math.nan]])]
    for instruments in invalid:
        with pytest.raises(ValueError):
            IVDesign(instruments)
    with pytest.raises(ValueError):
        IVDesign(z, np.ones(90))
    with pytest.raises(ValueError):
        IVDesign(controls, controls)
    with pytest.raises(ValueError):
        IVDesign(z, controls[:-1])
    design = IVDesign(z)
    for outcome in (y[:-1], y[:, None], np.full_like(y, np.nan)):
        with pytest.raises(ValueError):
            design.fit(outcome, x)
    with pytest.raises(ValueError):
        design.fit(y, x).ar_test(math.inf)
