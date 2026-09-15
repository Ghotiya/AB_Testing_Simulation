"""Study 2: the cost of peeking (gate G2).

For each K in LOOK_COUNTS, the experiment is tested at K equally spaced looks and stopped
at the first p < threshold. Three rules are compared:
- uncorrected: alpha at every look (the failure mode);
- Bonferroni: alpha / K at every look (the planned fix);
- Pocock: one constant boundary calibrated so the overall rate is exactly alpha
  (secondary), which measures how much power Bonferroni gives away by ignoring the
  correlation between nested looks.
Under H0 this gives the true false-positive rate; under the design alternative it gives
power, compared with the fixed-horizon test on the same trials.

One stream per case is simulated on the union of all look schedules, so every K and every
rule see the same trials (common random numbers). Differences between rules are
therefore paired, with paired standard errors.

The simulated null curves are also checked against the large-sample Gaussian
recursive-integration reference (absim/theory.py), which shares no code with the simulator.

Run: python -m studies.study2_peeking   (exit code 1 if G2 fails)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from absim.config import DEFAULT, LOOK_COUNTS
from absim.mc import mean_estimate, rate_estimate
from absim.peeking import apply_stopping
from absim.rng import make_rng
from absim.simulate import look_schedule, run_trials_at_looks, union_look_schedule
from absim.theory import gaussian_crossing_probability

N_TRIALS = 50_000
RESULTS = Path(__file__).resolve().parents[1] / "results"
METHODS = {"uncorrected": None, "bonferroni": "bonferroni", "pocock": "pocock"}
COMPARISONS = (
    ("uncorrected", "fixed"),
    ("bonferroni", "fixed"),
    ("pocock", "fixed"),
    ("bonferroni", "uncorrected"),
    ("pocock", "bonferroni"),
)
Z_CAP = 3.5  # 27 null cells: ~1% chance that any exceeds 3.5 SE by chance


def main() -> int:
    cfg = DEFAULT
    RESULTS.mkdir(exist_ok=True)
    N = cfg.n_per_arm
    union = union_look_schedule(N, LOOK_COUNTS)

    rows, paired = [], []
    for case, p_t in (("null", cfg.p_c), ("alt", cfg.p_t)):
        full = run_trials_at_looks(make_rng("study2", case=case), N_TRIALS, cfg.p_c, p_t, union)
        fixed = full.p[:, -1] < cfg.alpha  # fixed-horizon test on the same trials
        for K in LOOK_COUNTS:
            batch = full.select_looks(look_schedule(N, K))
            rejected = {"fixed": fixed}
            for name, corr in METHODS.items():
                out = apply_stopping(batch, cfg.alpha, corr)
                rejected[name] = out.rejected
                est = rate_estimate(out.rejected)
                theory = np.nan
                if case == "null":
                    z_crit = norm.ppf(1 - out.alpha_per_look / 2)
                    theory = gaussian_crossing_probability(batch.n / N, z_crit)
                rows.append({
                    "case": case,
                    "K": K,
                    "method": name,
                    "alpha_per_look": out.alpha_per_look,
                    "z_crit_per_look": float(norm.ppf(1 - out.alpha_per_look / 2)),
                    "rate": est["rate"],
                    "se": est["se"],
                    "ci_lo": est["lo"],
                    "ci_hi": est["hi"],
                    "gaussian_theory": theory,
                    "mean_n_at_stop": float(out.n_at_stop.mean()),
                    "n_trials": N_TRIALS,
                })
            for a, b in COMPARISONS:
                est = mean_estimate(rejected[a].astype(int) - rejected[b].astype(int))
                paired.append({"case": case, "K": K, "comparison": f"{a}_minus_{b}", "diff": est["mean"],
                               "se": est["se"], "ci_lo": est["lo"], "ci_hi": est["hi"]})
        del full

    df = pd.DataFrame(rows)
    pdf = pd.DataFrame(paired)
    df.to_csv(RESULTS / "study2_peeking.csv", index=False)
    pdf.to_csv(RESULTS / "study2_paired_differences.csv", index=False)

    def curve(case, method):
        return df[(df.case == case) & (df.method == method)].set_index("K")

    def diff(case, comparison):
        return pdf[(pdf.case == case) & (pdf.comparison == comparison)].set_index("K")

    unc, bon, poc = curve("null", "uncorrected"), curve("null", "bonferroni"), curve("null", "pocock")
    study1 = json.loads((RESULTS / "study1_summary.json").read_text())["null_fpr"]
    k1 = unc.loc[1]
    k1_gap = abs(k1.rate - study1["rate"])
    k1_tol = 2 * np.hypot(k1.se, study1["se"])

    null_all = df[df.case == "null"]
    theory_z = ((null_all.rate - null_all.gaussian_theory) / null_all.se).abs()
    bon_cost = diff("alt", "bonferroni_minus_fixed")
    pocock_alpha_z = ((poc.rate - cfg.alpha) / poc.se).abs()

    gates = {
        "K1_reproduces_study1_fpr": bool(k1_gap <= k1_tol),
        "uncorrected_fpr_increasing_in_K": bool(np.all(np.diff(unc.rate.values) > 0)),
        "bonferroni_fpr_at_most_alpha": bool(np.all(bon.rate.values <= cfg.alpha + 2 * bon.se.values))
        and bool(np.all(bon.rate.values[1:] < cfg.alpha)),
        "bonferroni_fpr_decreasing_in_K": bool(np.all(np.diff(bon.rate.values) < 0)),
        "bonferroni_power_cost_paired_ci_below_zero": bool(np.all(bon_cost[bon_cost.index > 1].ci_hi < 0)),
        "pocock_fpr_within_3.5se_of_alpha_every_K": bool(pocock_alpha_z.max() <= Z_CAP),
        "null_curves_within_3.5se_of_gaussian_theory": bool(theory_z.max() <= Z_CAP),
    }
    summary = {
        "n_trials": N_TRIALS,
        "look_counts": list(LOOK_COUNTS),
        "K1_vs_study1": {"study2_K1": float(k1.rate), "study1": study1["rate"], "gap": float(k1_gap), "tolerance": float(k1_tol)},
        "max_abs_z_vs_gaussian_theory": float(theory_z.max()),
        "max_abs_z_pocock_vs_alpha": float(pocock_alpha_z.max()),
        "gates": gates,
        "G2_pass": all(gates.values()),
    }
    (RESULTS / "study2_summary.json").write_text(json.dumps(summary, indent=2))

    print("False-positive rate under H0 (simulated ± MC SE, Gaussian theory in brackets)")
    print(f"{'K':>3} | {'uncorrected':>24} | {'Bonferroni':>24} | {'Pocock':>24} | {'Pocock z':>8}")
    for K in LOOK_COUNTS:
        cells = " | ".join(f"{c.loc[K].rate:.4f} ±{c.loc[K].se:.4f} [{c.loc[K].gaussian_theory:.4f}]" for c in (unc, bon, poc))
        print(f"{K:>3} | {cells} | {poc.loc[K].z_crit_per_look:>8.3f}")

    alt = {m: curve("alt", m) for m in METHODS}
    print(f"\nPower under the design alternative (fixed horizon on the same trials: {alt['uncorrected'].loc[1].rate:.4f})")
    print(f"{'K':>3} | {'uncorr':>6} {'Bonf':>6} {'Pocock':>6} | {'Bonf - fixed (95% CI)':>26} | "
          f"{'Pocock - fixed (95% CI)':>26} | {'Pocock - Bonf (95% CI)':>26}")
    for K in LOOK_COUNTS:
        def fmt(comp):
            r = diff("alt", comp).loc[K]
            return f"{r['diff']:+.4f} ({r.ci_lo:+.4f}, {r.ci_hi:+.4f})"
        print(f"{K:>3} | {alt['uncorrected'].loc[K].rate:.4f} {alt['bonferroni'].loc[K].rate:.4f} {alt['pocock'].loc[K].rate:.4f} | "
              f"{fmt('bonferroni_minus_fixed'):>26} | {fmt('pocock_minus_fixed'):>26} | {fmt('pocock_minus_bonferroni'):>26}")

    print(f"\nK=1 vs Study 1 FPR: {k1.rate:.4f} vs {study1['rate']:.4f} (gap {k1_gap:.4f}, tolerance {k1_tol:.4f})")
    print(f"Max |sim - Gaussian theory| / SE across null cells: {theory_z.max():.2f}")
    print(f"Max |Pocock FPR - alpha| / SE: {pocock_alpha_z.max():.2f}")
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("G2", "PASS" if summary["G2_pass"] else "FAIL")
    return 0 if summary["G2_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
