"""Study 3: power and sample size (gate G3).

Sweeps relative lift x per-arm n, and compares simulated power with two references:
- exact power, by enumerating both binomial arms (no approximation), which the simulator
  must match at every cell, small n included;
- closed-form normal-approximation power, whose gap to exact is formula error, reported
  rather than hidden.
Then validates required_sample_size() by round trip: solve n for 80% power, simulate at
that n, and check empirical power is 0.80 +/- 0.01.

Agreement is judged per cell by an exact binomial test of the simulated rejection count
against the reference probability, which stays valid in saturated cells (power ~ 1)
where a normal z with the empirical SE is 0/0. The gates are calibrated for 56 cells:
- no cell may have p < 0.01 / (number of cells) (Bonferroni: <= 1% chance of a false fail);
- the chi-square sum of z^2 over well-conditioned cells (>= 5 expected rejections and
  >= 5 expected non-rejections) must have p > 0.01, catching small errors spread over
  many cells that no single cell reveals.

Run: python -m studies.study3_power   (exit code 1 if G3 fails)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from absim.config import DEFAULT
from absim.exact import exact_rejection_probability
from absim.mc import rate_estimate
from absim.power import closed_form_power, power_sweep, required_sample_size
from absim.rng import make_rng
from absim.simulate import run_trials

LIFTS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
SIZES = (500, 1_000, 2_000, 5_000, 10_000, 15_000, 25_000, 50_000)
N_TRIALS = 20_000  # per grid cell
ROUND_TRIP_LIFTS = (0.05, 0.10, 0.20, 0.30)
ROUND_TRIP_TRIALS = 100_000
GATE_P = 0.01
RESULTS = Path(__file__).resolve().parents[1] / "results"


def compare_to_reference(rate: float, n_trials: int, reference: float) -> dict:
    """Exact binomial test of a simulated rate against a reference probability, plus a z-score
    that uses the reference's SE (so a saturated cell is not divided by zero)."""
    ref = float(np.clip(reference, 0.0, 1.0))
    hits = int(round(rate * n_trials))
    se_ref = math.sqrt(ref * (1 - ref) / n_trials)
    if se_ref > 0:
        z = (rate - ref) / se_ref
    else:
        z = 0.0 if hits == round(ref * n_trials) else math.inf
    return {
        "p": float(binomtest(hits, n_trials, ref).pvalue),
        "z": z,
        "well_conditioned": n_trials * ref >= 5 and n_trials * (1 - ref) >= 5,
    }


def agreement(cells: pd.DataFrame, ref_col: str, tag: str) -> tuple[pd.DataFrame, dict]:
    checks = pd.DataFrame(
        [compare_to_reference(r.power_sim, r.n_trials, getattr(r, ref_col)) for r in cells.itertuples()],
        index=cells.index,
    )
    well = checks[checks.well_conditioned]
    stat = float((well.z**2).sum())
    stats = {
        "cells": len(checks),
        "min_binomial_p": float(checks.p.min()),
        "bonferroni_threshold": GATE_P / len(checks),
        "well_conditioned_cells": len(well),
        "chi2": stat,
        "chi2_p": float(chi2.sf(stat, len(well))),
        "max_abs_z_well_conditioned": float(well.z.abs().max()),
    }
    cols = checks.rename(columns={"p": f"binom_p_{tag}", "z": f"z_{tag}", "well_conditioned": f"well_conditioned_{tag}"})
    return cols, stats


def main() -> int:
    cfg = DEFAULT
    RESULTS.mkdir(exist_ok=True)

    grid = power_sweep(LIFTS, SIZES, p_c=cfg.p_c, alpha=cfg.alpha, n_trials=N_TRIALS)
    grid["closed_form_minus_exact"] = grid.power_closed_form - grid.power_exact
    exact_cols, vs_exact = agreement(grid, "power_exact", "vs_exact")
    grid = grid.join(exact_cols)
    large = grid[grid.n_per_arm >= 1_000]
    cf_cols, vs_closed_form = agreement(large, "power_closed_form", "vs_closed_form")
    grid = grid.join(cf_cols)
    grid.to_csv(RESULTS / "study3_power_grid.csv", index=False)

    trips = []
    for lift in ROUND_TRIP_LIFTS:
        mde = cfg.p_c * lift
        n = required_sample_size(mde, cfg.target_power, cfg.alpha, cfg.p_c)
        batch = run_trials(make_rng("study3_round_trip", relative_lift=lift), ROUND_TRIP_TRIALS, cfg.p_c, cfg.p_c + mde, n)
        est = rate_estimate(batch.p[:, 0] < cfg.alpha)
        trips.append({
            "relative_lift": lift,
            "abs_mde": mde,
            "required_n_per_arm": n,
            "power_sim": est["rate"],
            "se": est["se"],
            "ci_lo": est["lo"],
            "ci_hi": est["hi"],
            "power_closed_form_at_n": float(closed_form_power(cfg.p_c, cfg.p_c + mde, n, cfg.alpha)),
            "power_exact_at_n": exact_rejection_probability(n, cfg.p_c, cfg.p_c + mde, cfg.alpha),
            "n_trials": ROUND_TRIP_TRIALS,
        })
    trips = pd.DataFrame(trips)
    trips.to_csv(RESULTS / "study3_round_trip.csv", index=False)

    null = grid[(grid.relative_lift == 0.0) & (grid.n_per_arm >= 1_000)]
    null_p = [binomtest(int(round(r.power_sim * r.n_trials)), r.n_trials, cfg.alpha).pvalue for r in null.itertuples()]
    worst_cf = grid.loc[grid.closed_form_minus_exact.abs().idxmax()]

    gates = {
        "sim_vs_exact_no_cell_below_bonferroni": vs_exact["min_binomial_p"] >= vs_exact["bonferroni_threshold"],
        "sim_vs_exact_chi2_p_above_0.01": vs_exact["chi2_p"] > GATE_P,
        "sim_vs_closed_form_n_ge_1000_no_cell_below_bonferroni": vs_closed_form["min_binomial_p"] >= vs_closed_form["bonferroni_threshold"],
        "sim_vs_closed_form_n_ge_1000_chi2_p_above_0.01": vs_closed_form["chi2_p"] > GATE_P,
        "round_trip_power_within_0.01_at_all_mdes": bool(np.all((trips.power_sim - cfg.target_power).abs() <= 0.01)),
        "zero_lift_column_is_alpha_n_ge_1000": min(null_p) >= GATE_P / len(null_p),
    }
    summary = {
        "n_trials_per_cell": N_TRIALS,
        "sim_vs_exact": vs_exact,
        "sim_vs_closed_form_n_ge_1000": vs_closed_form,
        "max_abs_sim_minus_closed_form_n_ge_1000": float((large.power_sim - large.power_closed_form).abs().max()),
        "max_abs_closed_form_minus_exact": {
            "value": float(worst_cf.closed_form_minus_exact),
            "relative_lift": float(worst_cf.relative_lift),
            "n_per_arm": int(worst_cf.n_per_arm),
        },
        "zero_lift_min_binomial_p_n_ge_1000": float(min(null_p)),
        "gates": {k: bool(v) for k, v in gates.items()},
        "G3_pass": bool(all(gates.values())),
    }
    (RESULTS / "study3_summary.json").write_text(json.dumps(summary, indent=2))

    print("Simulated power (rows: relative lift, columns: n per arm)")
    print(grid.pivot(index="relative_lift", columns="n_per_arm", values="power_sim").round(3).to_string())
    print("\nClosed form minus exact power")
    print(grid.pivot(index="relative_lift", columns="n_per_arm", values="closed_form_minus_exact").round(4).to_string())
    print("\n(sim - exact) / SE of exact")
    print(grid.pivot(index="relative_lift", columns="n_per_arm", values="z_vs_exact").round(2).to_string())
    print("\nRound trip (target power 0.80)")
    print(trips[["relative_lift", "required_n_per_arm", "power_sim", "se", "power_exact_at_n", "power_closed_form_at_n"]].round(4).to_string(index=False))
    for label, s in (("sim vs exact, all cells", vs_exact), ("sim vs closed form, n >= 1000", vs_closed_form)):
        print(f"\n{label}: min binomial p {s['min_binomial_p']:.4f} (Bonferroni threshold {s['bonferroni_threshold']:.5f}); "
              f"chi2 {s['chi2']:.1f} on {s['well_conditioned_cells']} well-conditioned cells, p = {s['chi2_p']:.3f}; "
              f"max |z| {s['max_abs_z_well_conditioned']:.2f}")
    print(f"max |sim - closed form| at n >= 1000: {summary['max_abs_sim_minus_closed_form_n_ge_1000']:.4f}")
    print(f"largest closed-form error vs exact: {worst_cf.closed_form_minus_exact:+.4f} "
          f"at lift {worst_cf.relative_lift}, n {int(worst_cf.n_per_arm)}")
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("G3", "PASS" if summary["G3_pass"] else "FAIL")
    return 0 if summary["G3_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
