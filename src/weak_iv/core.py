"""QR-based IV statistics and analytic (not grid-truncated) AR inversion."""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache
import math

import numpy as np
from scipy.stats import f, norm


def _alpha(alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return float(alpha)


@lru_cache(maxsize=64)
def _critical(alpha, numerator_df, denominator_df):
    return float(f.isf(_alpha(alpha), numerator_df, denominator_df))


@dataclass(frozen=True)
class ConfidenceSet:
    """Closed real intervals; infinite endpoints denote genuinely open tails."""

    intervals: tuple[tuple[float, float], ...]

    @property
    def kind(self):
        if not self.intervals:
            return "empty"
        if len(self.intervals) == 2:
            return "two_rays"
        low, high = self.intervals[0]
        if low == -math.inf and high == math.inf:
            return "all_real"
        if low == -math.inf:
            return "left_ray"
        if high == math.inf:
            return "right_ray"
        return "bounded"

    @property
    def unbounded(self):
        return any(not math.isfinite(a) or not math.isfinite(b) for a, b in self.intervals)

    def contains(self, value):
        if not math.isfinite(value):
            raise ValueError("membership requires a finite real value")
        return any(a <= value <= b for a, b in self.intervals)

    def to_dict(self):
        # JSON null in a lower endpoint means -infinity, in an upper +infinity.
        return {
            "kind": self.kind,
            "unbounded": self.unbounded,
            "intervals": [[a if math.isfinite(a) else None, b if math.isfinite(b) else None]
                          for a, b in self.intervals],
        }


def solve_quadratic_leq(a, b, c):
    """Solve a*t^2 + b*t + c <= 0, retaining all rays and degenerate cases.

    Decimal arithmetic (80 digits) stabilizes discriminants and quadratic roots
    for the supplied float coefficients. No 'small a' tolerance changes a
    quadratic into a line. This cannot recover precision already lost when
    computing input coefficients. Unrepresentable finite roots raise an error.
    """
    coefficients = tuple(float(v) for v in (a, b, c))
    if not all(math.isfinite(v) for v in coefficients):
        raise ValueError("quadratic coefficients must be finite")
    with localcontext() as ctx:
        ctx.prec = 80
        a, b, c = (Decimal.from_float(v) for v in coefficients)

        def endpoint(value):
            result = float(value)
            if not math.isfinite(result) or (result == 0 and value != 0):
                raise ValueError("finite root is outside floating-point endpoint range")
            return result

        if a == 0:
            if b == 0:
                return ConfidenceSet(((-math.inf, math.inf),) if c <= 0 else ())
            root = endpoint(-c / b)
            return ConfidenceSet(((-math.inf, root),) if b > 0 else ((root, math.inf),))
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return ConfidenceSet(() if a > 0 else ((-math.inf, math.inf),))
        if discriminant == 0:
            if a < 0:
                return ConfidenceSet(((-math.inf, math.inf),))
            root = endpoint(-b / (2 * a))
            return ConfidenceSet(((root, root),))
        # One root from a non-cancelling numerator; the other from their product.
        sqrt_d = discriminant.sqrt()
        q = -(b + (sqrt_d if b >= 0 else -sqrt_d)) / 2
        low, high = sorted((endpoint(q / a), endpoint(c / q)))
        if a > 0:
            return ConfidenceSet(((low, high),))
        return ConfidenceSet(((-math.inf, low), (high, math.inf)))


def _matrix(value, name, rows=None):
    result = np.array(value, dtype=float, copy=True)
    if result.ndim == 1:
        result = result[:, None]
    if (result.ndim != 2 or min(result.shape) < 1
            or not np.all(np.isfinite(result)) or (rows is not None and result.shape[0] != rows)):
        raise ValueError(f"{name} must be a finite matrix with the required row count")
    return result


def _basis(matrix, name):
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError(f"{name} must have full column rank")
    return np.linalg.qr(matrix, mode="reduced")[0]


class IVDesign:
    """Fixed excluded instruments and optional included exogenous controls.

    An intercept is always added: do not include one in controls. The classical
    AR reference is exact only when structural errors are iid Gaussian and
    independent of this fixed design. Endogenous x may correlate with errors.
    """

    def __init__(self, instruments, controls=None):
        z = _matrix(instruments, "instruments")
        self.n, self.k = z.shape
        nuisance = np.ones((self.n, 1))
        if controls is not None:
            nuisance = np.column_stack((nuisance, _matrix(controls, "controls", self.n)))
        self.p = nuisance.shape[1]
        if self.n <= self.p + self.k:
            raise ValueError("positive residual degrees of freedom are required")
        self.q_controls = _basis(nuisance, "intercept and controls")
        # Check joint rank before residualization so an instrument in the
        # control span cannot be mistaken for a tiny but independent direction.
        _basis(np.column_stack((nuisance, z)), "controls and excluded instruments")
        z_residual = z - self.q_controls @ (self.q_controls.T @ z)
        self.q_instruments = _basis(z_residual, "residualized instruments")

    def fit(self, outcome, endogenous):
        y, x = (np.asarray(v, dtype=float) for v in (outcome, endogenous))
        if (y.shape != (self.n,) or x.shape != (self.n,)
                or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x))):
            raise ValueError("outcome and endogenous must be finite vectors matching the design")
        columns = np.column_stack((y, x))
        residualized = columns - self.q_controls @ (self.q_controls.T @ columns)
        projected_coordinates = self.q_instruments.T @ residualized
        unexplained = residualized - self.q_instruments @ projected_coordinates
        return IVFit(self.n, self.p, self.k, residualized, projected_coordinates, unexplained)


@dataclass(frozen=True)
class IVFit:
    n: int
    p: int
    k: int
    residualized: np.ndarray
    projected: np.ndarray
    unexplained: np.ndarray

    @property
    def residual_df(self):
        return self.n - self.p - self.k

    def _f_statistic(self, projected, unexplained):
        explained_ss = float(projected @ projected)
        residual_ss = float(unexplained @ unexplained)
        if residual_ss == 0:
            # Closure convention for a perfect zero residual, a null-probability
            # event under the nondegenerate Gaussian model.
            return (0.0 if explained_ss == 0 else math.inf), True
        return (explained_ss / self.k) / (residual_ss / self.residual_df), False

    def first_stage_f(self):
        return self._f_statistic(self.projected[:, 1], self.unexplained[:, 1])[0]

    def ar_test(self, beta, alpha=.05):
        if not math.isfinite(beta):
            raise ValueError("beta must be finite")
        critical = _critical(_alpha(alpha), self.k, self.residual_df)
        statistic, degenerate = self._f_statistic(
            self.projected[:, 0] - beta * self.projected[:, 1],
            self.unexplained[:, 0] - beta * self.unexplained[:, 1],
        )
        return {
            "statistic": statistic,
            "critical_value": critical,
            "p_value": float(f.sf(statistic, self.k, self.residual_df)),
            "accepted": statistic <= critical,
            "degenerate_residual": degenerate,
        }

    def ar_coefficients(self, alpha=.05):
        critical = _critical(_alpha(alpha), self.k, self.residual_df)
        tau = self.k * critical / self.residual_df
        gram = self.projected.T @ self.projected - tau * (self.unexplained.T @ self.unexplained)
        return float(gram[1, 1]), float(-2 * gram[0, 1]), float(gram[0, 0])

    def ar_confidence_set(self, alpha=.05):
        return solve_quadratic_leq(*self.ar_coefficients(alpha))

    def wald(self, alpha=.05):
        """Conventional homoskedastic 2SLS with normal, not exact-t, limits."""
        alpha = _alpha(alpha)
        denominator = float(self.projected[:, 1] @ self.projected[:, 1])
        if denominator <= 0:
            raise ValueError("2SLS is undefined with zero fitted endogenous variation")
        beta = float(self.projected[:, 1] @ self.projected[:, 0]) / denominator
        residual = self.residualized[:, 0] - beta * self.residualized[:, 1]
        variance = float(residual @ residual) / (self.n - self.p - 1)
        se = math.sqrt(variance / denominator)
        half_width = float(norm.isf(alpha / 2)) * se
        if not all(math.isfinite(v) for v in (beta, se, half_width, beta - half_width, beta + half_width)):
            raise ValueError("2SLS result is outside floating-point range")
        return {"estimate": beta, "standard_error": se, "interval": [beta - half_width, beta + half_width]}
