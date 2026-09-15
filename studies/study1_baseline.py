"""Study 1: baseline validation of the simulator (hard gate G1).

Fixed-horizon test at the design point. Under a true null the FPR must be 0.05 and the
p-values must follow the test's exact (enumerated) null distribution; under the design alternative the power must be 0.80. A small-n corner
where the z-test's normal approximation is known to be poor is reported as a limitation
of the test, not the simulator.

Run: python -m studies.study1_baseline   (exit code 1 if G1 fails)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest

from absim.config import DEFAULT
from absim.exact import exact_null_pvalue_distribution, ks_against
from absim.mc import mean_estimate, rate_estimate
from absim.power import closed_form_power
from absim.rng import make_rng
from absim.simulate import run_trials

N_TRIALS = 100_000
SMALL_N, SMALL_P = 200, 0.02  # small-n corner where the z-test is known to be conservative
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    cfg = DEFAULT
    RESULTS.mkdir(exist_ok=True)

    # Null at the design point.
    null = run_trials(make_rng("study1", case="null"), N_TRIALS, cfg.p_c, cfg.p_c, cfg.n_per_arm)
    p_null = null.p[:, 0]
    fpr = rate_estimate(p_null < cfg.alpha)
    fpr_other_alphas = {str(a): rate_estimate(p_null < a) for a in (0.01, 0.10)}
    exact_null = exact_null_pvalue_distribution(cfg.n_per_arm, cfg.p_c)
    ks_exact = ks_against(p_null, exact_null)
    ks_uniform = kstest(p_null, "uniform")
    exact_size = float(exact_null.cdf_left(cfg.alpha))
    exact_tie_atom = float(1 - exact_null.cdf_left(1.0))
    sim_tie_share = rate_estimate(p_null == 1.0)
    z_mean = mean_estimate(null.z[:, 0])
    z_sd = float(null.z[:, 0].std(ddof=1))

    counts, edges = np.histogram(p_null, bins=20, range=(0, 1))
    pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "count": counts}).to_csv(
        RESULTS / "study1_null_pvalue_hist.csv", index=False
    )

    # Alternative at the design point.
    alt = run_trials(make_rng("study1", case="design_power"), N_TRIALS, cfg.p_c, cfg.p_t, cfg.n_per_arm)
    power = rate_estimate(alt.p[:, 0] < cfg.alpha)
    power_theory = float(closed_form_power(cfg.p_c, cfg.p_t, cfg.n_per_arm, cfg.alpha))
    delta_hat = mean_estimate(alt.delta_hat[:, 0])

    # Small-n corner (limitation of the z-test).
    small = run_trials(make_rng("study1", case="small_n"), N_TRIALS, SMALL_P, SMALL_P, SMALL_N)
    small_fpr = rate_estimate(small.p[:, 0] < cfg.alpha)
    small_degenerate = float(((small.x_c + small.x_t)[:, 0] == 0).mean())
    small_exact = exact_null_pvalue_distribution(SMALL_N, SMALL_P)
    small_exact_size = float(small_exact.cdf_left(cfg.alpha))
    small_ks_exact = ks_against(small.p[:, 0], small_exact)

    # The KS gate compares against the exact discrete null of this test, not U(0,1):
    # U(0,1) is rejected at 100k trials by the tie atom at p = 1 even for a perfect
    # simulator. The U(0,1) distance is reported and explained by that atom.
    gates = {
        "fpr_ci_contains_alpha": fpr["lo"] <= cfg.alpha <= fpr["hi"],
        "ks_vs_exact_null_p_above_0.01": ks_exact["pvalue"] > 0.01,
        "power_within_0.01_of_target": abs(power["rate"] - cfg.target_power) <= 0.01,
    }
    summary = {
        "design": {"p_c": cfg.p_c, "p_t": cfg.p_t, "alpha": cfg.alpha, "n_per_arm": cfg.n_per_arm, "n_trials": N_TRIALS},
        "null_fpr": fpr,
        "null_fpr_other_alphas": fpr_other_alphas,
        "null_exact_size": exact_size,
        "null_ks_vs_exact": ks_exact,
        "null_ks_vs_uniform": {"statistic": float(ks_uniform.statistic), "pvalue": float(ks_uniform.pvalue)},
        "null_tie_atom_exact": exact_tie_atom,
        "null_tie_share_sim": sim_tie_share,
        "null_z_mean": z_mean,
        "null_z_sd": z_sd,
        "design_power": power,
        "design_power_closed_form": power_theory,
        "design_delta_hat_mean": delta_hat,
        "true_delta": cfg.delta,
        "small_n": {
            "n_per_arm": SMALL_N,
            "p": SMALL_P,
            "fpr": small_fpr,
            "exact_size": small_exact_size,
            "ks_vs_exact": small_ks_exact,
            "share_degenerate": small_degenerate,
        },
        "gates": gates,
        "G1_pass": all(gates.values()),
    }
    (RESULTS / "study1_summary.json").write_text(json.dumps(summary, indent=2, default=bool))

    print(f"Null FPR at alpha=0.05: {fpr['rate']:.4f} (MC SE {fpr['se']:.4f}, 95% CI {fpr['lo']:.4f}-{fpr['hi']:.4f})")
    for a, est in fpr_other_alphas.items():
        print(f"Null FPR at alpha={a}: {est['rate']:.4f} (SE {est['se']:.4f})")
    print(f"Exact size of the test at alpha=0.05: {exact_size:.5f}")
    print(f"KS vs exact discrete null: D={ks_exact['statistic']:.5f}, p={ks_exact['pvalue']:.3f}")
    print(f"KS vs U(0,1): D={ks_uniform.statistic:.5f}, p={ks_uniform.pvalue:.2g} "
          f"(tie atom at p=1: exact {exact_tie_atom:.5f}, simulated {sim_tie_share['rate']:.5f} SE {sim_tie_share['se']:.5f})")
    print(f"Null z: mean {z_mean['mean']:+.4f} (SE {z_mean['se']:.4f}), sd {z_sd:.4f}")
    print(f"Design power: {power['rate']:.4f} (SE {power['se']:.4f}); closed form {power_theory:.4f}")
    print(f"Mean delta_hat under H1: {delta_hat['mean']:.6f} (SE {delta_hat['se']:.6f}), true {cfg.delta:.6f}")
    print(f"Small-n corner n={SMALL_N}, p={SMALL_P}: FPR {small_fpr['rate']:.4f} (SE {small_fpr['se']:.4f}), degenerate {small_degenerate:.4%}; "
          f"exact size {small_exact_size:.4f}, KS vs exact p={small_ks_exact['pvalue']:.3f}")
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("G1", "PASS" if summary["G1_pass"] else "FAIL")
    return 0 if summary["G1_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
