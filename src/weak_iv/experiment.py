"""Fixed-design, seeded Monte Carlo study; not an empirical causal claim."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import f, ncf

from .core import IVDesign


SCENARIOS = (
    ("zero_strength", 0.0, 0.0),
    ("weak", 4.0, 0.0),
    ("intermediate", 16.0, 0.0),
    ("strong", 100.0, 0.0),
    ("invalid_exclusion", 100.0, 0.35),
)
KINDS = ("bounded", "two_rays", "all_real", "empty", "left_ray", "right_ray")


def fixed_instruments(n, seed):
    """Two centered, orthogonal columns with Z'Z = n I (up to roundoff)."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 10:
        raise ValueError("n must be an integer at least 10")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    raw = rng.standard_normal((n, 2))
    centered = raw - raw.mean(axis=0)
    return np.linalg.qr(centered, mode="reduced")[0] * np.sqrt(n)


def estimate(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"estimate": None, "mcse": None, "count": 0}
    return {
        "estimate": float(values.mean()),
        "mcse": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
        "count": len(values),
    }


def run_experiment(replicates=3000, n=200, seed=20261011):
    if not isinstance(replicates, (int, np.integer)) or isinstance(replicates, bool) or replicates < 2:
        raise ValueError("replicates must be an integer at least 2")
    z = fixed_instruments(n, seed)
    design = IVDesign(z)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    innovations = rng.standard_normal((replicates, n, 2))
    beta, rho, alpha = 1.0, .8, .05
    critical = float(f.isf(alpha, design.k, design.n - design.p - design.k))
    denominator_df = design.n - design.p - design.k
    results, examples = [], {}
    for name, concentration, direct_effect in SCENARIOS:
        ar_coverage, wald_coverage, first_stage, shapes = [], [], [], []
        bounded_lengths, wald_lengths = [], []
        for draw, normals in enumerate(innovations):
            v = normals[:, 0]
            u = rho * v + np.sqrt(1 - rho**2) * normals[:, 1]
            x = z[:, 0] * np.sqrt(concentration / n) + v
            y = beta * x + direct_effect * z[:, 1] + u
            fitted = design.fit(y, x)
            confidence = fitted.ar_confidence_set(alpha)
            conventional = fitted.wald(alpha)
            ar_coverage.append(confidence.contains(beta))
            low, high = conventional["interval"]
            wald_coverage.append(low <= beta <= high)
            wald_lengths.append(high - low)
            first_stage.append(fitted.first_stage_f())
            shapes.append(confidence.kind)
            if confidence.kind == "bounded":
                a, b = confidence.intervals[0]
                bounded_lengths.append(b - a)
            # First observed occurrence of each shape, not a search for an
            # attractive estimate. These illustrations are not a coverage sample.
            if confidence.kind not in examples:
                examples[confidence.kind] = {
                    "scenario": name, "zero_based_draw": draw,
                    "ar_set": confidence.to_dict(),
                    "quadratic_coefficients": list(fitted.ar_coefficients(alpha)),
                    "first_stage_f": first_stage[-1],
                    "wald": conventional,
                    "ar_test_at_truth": fitted.ar_test(beta, alpha),
                }
        ar_values, wald_values = np.asarray(ar_coverage), np.asarray(wald_coverage)
        shapes_array = np.asarray(shapes)
        shape_counts = {kind: int(np.sum(shapes_array == kind)) for kind in KINDS}
        results.append({
            "scenario": name,
            "concentration_parameter": concentration,
            "direct_effect_on_second_instrument": direct_effect,
            "exclusion_restriction_valid_by_construction": direct_effect == 0,
            "analytic_ar_coverage_under_generator": (
                1 - alpha if direct_effect == 0 else
                float(ncf.cdf(critical, design.k, denominator_df, n * direct_effect**2))
            ),
            "analytic_ar_unbounded_probability_under_generator": (
                1 - alpha if concentration == 0 else
                float(ncf.cdf(critical, design.k, denominator_df, concentration))
            ),
            "ar_coverage": estimate(ar_values),
            "wald_coverage": estimate(wald_values),
            "paired_ar_minus_wald_coverage": estimate(ar_values.astype(float) - wald_values),
            "ar_set_shape_counts": shape_counts,
            "ar_unbounded_fraction": sum(shape_counts[k] for k in ("two_rays", "all_real", "left_ray", "right_ray")) / replicates,
            "coverage_conditional_on_ar_shape": {kind: estimate(ar_values[shapes_array == kind]) for kind in KINDS},
            "bounded_ar_length_median_conditional_on_bounded": float(np.median(bounded_lengths)) if bounded_lengths else None,
            "wald_length_median": float(np.median(wald_lengths)),
            "first_stage_f_median": float(np.median(first_stage)),
            "first_stage_f_quantiles_10_90": np.quantile(first_stage, [.1, .9]).tolist(),
        })
    return {
        "data": "Synthetic fixed-design linear IV model; no private or external observations.",
        "replicates": replicates, "n": n, "seed": seed,
        "true_beta": beta, "error_correlation": rho, "alpha": alpha,
        "excluded_instruments": 2, "included_controls": "intercept only",
        "simulation_design": "Same fixed instruments and paired innovations across scenarios; independent draws within each scenario.",
        "endpoint_encoding": "A null lower endpoint is -infinity; a null upper endpoint is +infinity. An empty list is an empty set.",
        "instrument_gram_divided_by_n": (z.T @ z / n).tolist(),
        "scenarios": results,
        "first_occurrence_examples": examples,
    }


def make_figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rows = summary["scenarios"]
    positions = np.arange(len(rows))
    labels = ["λ = 0", "λ = 4", "λ = 16", "λ = 100", "λ = 100\ninvalid exclusion"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for offset, field, label, color in (
        (-.17, "ar_coverage", "AR confidence set", "#227a78"),
        (.17, "wald_coverage", "2SLS normal-Wald interval", "#b76738"),
    ):
        axes[0].bar(positions + offset, [r[field]["estimate"] for r in rows], width=.34,
                    yerr=[2 * r[field]["mcse"] for r in rows], color=color, capsize=3, label=label)
    axes[0].axhline(.95, color="#202c40", linestyle="--", label="Nominal 95%")
    axes[0].set(title="Unconditional coverage (± 2 MCSE)", ylabel="Coverage", ylim=(0, 1.07))
    axes[0].legend(fontsize=8, loc="lower left")
    bottom = np.zeros(len(rows))
    for kind, label, color in (
        ("bounded", "Bounded", "#227a78"),
        ("two_rays", "Two unbounded rays", "#688eaf"),
        ("all_real", "Whole real line", "#b8c9d5"),
        ("empty", "Empty", "#b76738"),
        ("left_ray", "Left ray", "#bd91b7"),
        ("right_ray", "Right ray", "#b7a57a"),
    ):
        values = np.array([r["ar_set_shape_counts"][kind] / summary["replicates"] for r in rows])
        if np.any(values):
            axes[1].bar(positions, values, bottom=bottom, color=color, label=label)
            bottom += values
    axes[1].set(title="AR set geometry, with no truncation", ylabel="Fraction of simulations", ylim=(0, 1.07))
    axes[1].legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_xticks(positions, labels)
        ax.set_xlabel("Instrument concentration λ; last case breaks an assumption")
    fig.suptitle(f"Weak-instrument inference · n={summary['n']} · {summary['replicates']:,} paired draws per scenario")
    fig.savefig(output / "coverage_and_geometry.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, kind in zip(axes.flat, ("all_real", "two_rays", "bounded", "empty")):
        example = summary["first_occurrence_examples"].get(kind)
        if example is None:
            ax.text(.5, .5, f"{kind}: not observed in this run", ha="center", va="center", transform=ax.transAxes)
            continue
        endpoints = [v for interval in example["ar_set"]["intervals"] for v in interval if v is not None]
        support = [summary["true_beta"] - 2, summary["true_beta"] + 2] + endpoints
        low, high = min(support), max(support)
        margin = .15 * (high - low)
        grid = np.linspace(low - margin, high + margin, 1201)
        a, b, c = example["quadratic_coefficients"]
        values = (a * grid + b) * grid + c
        values /= max(float(np.max(np.abs(values))), 1e-12)
        ax.plot(grid, values, color="#202c40")
        ax.fill_between(grid, -1.1, 1.1, where=values <= 0, color="#227a78", alpha=.18,
                        label="Accepted values in viewport")
        ax.axhline(0, color="#707070", linewidth=.8)
        ax.axvline(summary["true_beta"], color="#b76738", linestyle="--", label="Generator β = 1")
        ax.set(title=f"{kind.replace('_', ' ').title()} · {example['scenario']} draw {example['zero_based_draw']}",
               xlabel="Candidate coefficient β", ylabel="Rescaled quadratic q(β)", ylim=(-1.1, 1.1))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("First occurrence of each shape: accept q(β) ≤ 0\nFinite viewports only; complete unbounded endpoints are stored in JSON", fontsize=12)
    fig.savefig(output / "confidence_set_shapes.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--replicates", type=int, default=3000)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.replicates < 2 or args.n < 10 or args.seed < 0:
        parser.error("require replicates >= 2, n >= 10, seed >= 0")
    summary = run_experiment(args.replicates, args.n, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    make_figures(summary, args.output)
    print(f"Wrote summary.json and two figures to {args.output}")


if __name__ == "__main__":
    main()
