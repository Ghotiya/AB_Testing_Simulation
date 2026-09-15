"""Study 4: winner's curse from early stopping (gate G4).

For each true relative lift (fixed-horizon power spanning ~0.2 to ~0.98, plus the null)
one stream of 50,000 trials is simulated on the union of all look schedules. Every K
then sees the same trials (common random numbers). Uncorrected stop-at-first-significance
is applied, and the effect reported by declared winners is compared with the truth.

Decomposition:
  selection-only bias      = bias at K = 1 (fixed horizon, no peeking)
  incremental peeking bias = bias(K) - bias(K = 1), with a Poisson-bootstrap CI
The K = 1 winners' mean is checked against its exact enumerated value at every lift.

Run: python -m studies.study4_winners_curse   (exit code 1 if G4 fails)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from absim.config import DEFAULT, LOOK_COUNTS
from absim.exact import exact_rejection_probability, exact_winner_summary
from absim.peeking import apply_stopping
from absim.rng import make_rng
from absim.simulate import look_schedule, run_trials_at_looks, union_look_schedule
from absim.winners_curse import bootstrap_conditional_mean_difference, summarize_winners, truncated_normal_winner_mean

LIFTS = (0.0, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14)
N_TRIALS = 50_000
N_BOOT = 400
Z_CAP = 3.5
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    cfg = DEFAULT
    RESULTS.mkdir(exist_ok=True)
    N = cfg.n_per_arm
    union = union_look_schedule(N, LOOK_COUNTS)

    rows = []
    for lift in LIFTS:
        p_t = cfg.p_c * (1 + lift)
        delta = p_t - cfg.p_c
        full = run_trials_at_looks(make_rng("study4", relative_lift=lift), N_TRIALS, cfg.p_c, p_t, union)
        exact = exact_winner_summary(N, cfg.p_c, p_t, cfg.alpha)
        fixed_power = exact_rejection_probability(N, cfg.p_c, p_t, cfg.alpha)
        boot_rng = make_rng("study4_bootstrap", relative_lift=lift)

        outcomes = {K: apply_stopping(full.select_looks(look_schedule(N, K)), cfg.alpha) for K in LOOK_COUNTS}
        base = outcomes[1]
        base_winners = base.rejected & (base.delta_hat_at_stop > 0)
        for K, out in outcomes.items():
            row = {
                "relative_lift": lift,
                "true_delta": delta,
                "fixed_horizon_power_exact": fixed_power,
                "K": K,
                "n_trials": N_TRIALS,
                **summarize_winners(out, delta),
            }
            winners = out.rejected & (out.delta_hat_at_stop > 0)
            inc = bootstrap_conditional_mean_difference(
                out.delta_hat_at_stop, winners, base.delta_hat_at_stop, base_winners, boot_rng, N_BOOT
            ) if K > 1 else {"diff": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0}
            row.update({
                "incremental_mean_vs_K1": inc["diff"],
                "incremental_se": inc["se"],
                "incremental_lo": inc["lo"],
                "incremental_hi": inc["hi"],
            })
            if delta > 0:
                row.update({
                    "incremental_rel": inc["diff"] / delta,
                    "incremental_rel_lo": inc["lo"] / delta,
                    "incremental_rel_hi": inc["hi"] / delta,
                })
            if K == 1:
                row.update({
                    "exact_winners_mean": exact["mean_delta_hat_winners"],
                    "exact_p_winner": exact["p_winner"],
                    "truncated_normal_winners_mean": truncated_normal_winner_mean(cfg.p_c, p_t, N, cfg.alpha),
                })
            # a bias is only claimed where its CI excludes zero
            row["winners_bias_ci_excludes_zero"] = bool(row["winners_lo"] > delta or row["winners_hi"] < delta)
            rows.append(row)
        del full, outcomes

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "study4_winners_curse.csv", index=False)

    k1 = df[df.K == 1].set_index("relative_lift")
    k1_z = (k1.winners_mean - k1.exact_winners_mean) / k1.winners_se
    positive = df[df.relative_lift > 0]
    rel = positive.pivot(index="relative_lift", columns="K", values="winners_rel_bias")
    lowest, highest = min(l for l in LIFTS if l > 0), max(LIFTS)

    gates = {
        "K1_winners_mean_matches_exact_every_lift_abs_z_le_3.5": bool(k1_z.abs().max() <= Z_CAP),
        "K1_selection_bias_positive_at_lowest_power": bool(k1.loc[lowest].winners_rel_lo > 0),
        "K1_selection_bias_small_at_highest_power_lt_5pct": bool(k1.loc[highest].winners_rel_bias < 0.05),
        "winners_bias_decreasing_in_true_effect_every_K": bool(np.all(np.diff(rel.values, axis=0) < 0)),
    }
    summary = {
        "n_trials": N_TRIALS,
        "lifts": list(LIFTS),
        "look_counts": list(LOOK_COUNTS),
        "fixed_horizon_power_exact": {str(l): float(k1.loc[l].fixed_horizon_power_exact) for l in LIFTS},
        "K1_max_abs_z_vs_exact": float(k1_z.abs().max()),
        "gates": gates,
        "G4_pass": all(gates.values()),
    }
    (RESULTS / "study4_summary.json").write_text(json.dumps(summary, indent=2))

    power = k1.fixed_horizon_power_exact
    print("Relative bias of declared winners' reported effect (rows: lift [fixed-horizon power], columns: K)")
    shown = rel.copy()
    shown.index = [f"{l:.2f} [{power.loc[l]:.2f}]" for l in rel.index]
    print((shown * 100).round(1).to_string())
    print("\nIncremental bias from peeking, K minus K=1, relative to true effect, % (95% bootstrap CI)")
    for lift in (lowest, 0.10, highest):
        sub = positive[(positive.relative_lift == lift) & (positive.K > 1)]
        cells = ", ".join(f"K={r.K}: {100*r.incremental_rel:+.1f} ({100*r.incremental_rel_lo:+.1f}, {100*r.incremental_rel_hi:+.1f})"
                          for r in sub.itertuples() if r.K in (2, 5, 14, 28))
        print(f"  lift {lift:.2f}: {cells}")
    print("\nWrong-sign share among significant results (rows: lift, columns: K), %")
    print((df.pivot(index="relative_lift", columns="K", values="wrong_sign_share_of_significant") * 100).round(2).to_string())
    print("\nFull-horizon estimate of the same winners, relative bias % (rows: lift, columns: K)")
    print((positive.pivot(index="relative_lift", columns="K", values="winners_full_horizon_rel_bias") * 100).round(1).to_string())
    null = df[df.relative_lift == 0].set_index("K")
    print("\nNull (no true effect): share of trials declaring a winner and their mean reported delta_hat")
    for K in (1, 5, 14, 28):
        print(f"  K={K}: winners {null.loc[K].share_winners:.4f}, mean delta_hat {null.loc[K].winners_mean:.5f} "
              f"(SE {null.loc[K].winners_se:.5f}); the design MDE is {cfg.delta:.3f}")
    print("\nK=1 winners' mean: simulated vs exact vs truncated-normal theory")
    for lift in LIFTS:
        r = k1.loc[lift]
        print(f"  lift {lift:.2f}: sim {r.winners_mean:.6f} (SE {r.winners_se:.6f}), exact {r.exact_winners_mean:.6f}, "
              f"normal theory {r.truncated_normal_winners_mean:.6f}, z vs exact {k1_z.loc[lift]:+.2f}")
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("G4", "PASS" if summary["G4_pass"] else "FAIL")
    return 0 if summary["G4_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
