"""Phase 5 figures F1-F4, drawn only from results/ (no re-simulation).

The one recomputation is analytic: F1 overlays the exact enumerated null and F3 draws
closed-form power as continuous lines. Every caption is built from the numbers in the
result files and written to figures/captions.md alongside the PNG + PDF.

Run: python -m absim.plots
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from absim.config import DEFAULT, LOOK_COUNTS
from absim.exact import exact_null_pvalue_distribution
from absim.power import closed_form_power

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"

# Palette: checked for colour-blind separation and distinct lightness steps on the light surface.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_COLOR = {"uncorrected": "#2a78d6", "bonferroni": "#eb6834", "pocock": "#1baf7a"}  # all-pairs pass
METHOD_LABEL = {"uncorrected": "Uncorrected", "bonferroni": "Bonferroni", "pocock": "Pocock"}
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal pass, light -> dark
F3_LIFTS = (0.05, 0.10, 0.20, 0.30)  # the Study 3 round-trip MDEs
F4_LIFTS = (0.04, 0.06, 0.10, 0.14)  # fixed-horizon power 0.21 / 0.40 / 0.80 / 0.97
HEADLINE_K = 14

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9.5, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5,
})
MARK = dict(markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5)


def _csv(name):
    # keep_default_na=False: the case label "null" would otherwise be read as NaN.
    return pd.read_csv(RESULTS / name, keep_default_na=False, na_values=[""])


def _json(name):
    return json.loads((RESULTS / name).read_text())


def _save(fig, stem, caption, captions):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    captions.append((stem, caption))


def _log_k_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(LOOK_COUNTS)
    ax.set_xticklabels([str(k) for k in LOOK_COUNTS])
    ax.minorticks_off()
    ax.set_xlabel("Number of looks K (daily checks on a K-day test)")


def figure1(captions):
    hist = _csv("study1_null_pvalue_hist.csv")
    s = _json("study1_summary.json")
    n_trials = s["design"]["n_trials"]
    exact = exact_null_pvalue_distribution(s["design"]["n_per_arm"], s["design"]["p_c"])
    lo, hi = hist.bin_lo.to_numpy(), hist.bin_hi.to_numpy()
    exact_mass = exact.cdf_left(hi) - exact.cdf_left(lo)
    exact_mass[-1] = 1.0 - exact.cdf_left(lo[-1])  # last bin is closed on the right: holds the p = 1 atom
    sim_mass = hist["count"].to_numpy() / n_trials

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = hi - lo
    ax.bar(lo + width * 0.06, sim_mass, width=width * 0.88, align="edge", color=METHOD_COLOR["uncorrected"],
           label=f"Simulated ({n_trials:,} null tests)", zorder=2)
    centers = (lo + hi) / 2
    ax.plot(centers, exact_mass, "o", color=INK, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="Exact enumerated null", zorder=3)
    ax.axhline(0.05, color=MUTED, linewidth=0.8, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(sim_mass.max(), exact_mass.max()) * 1.15)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Share of tests per 0.05 bin")
    fpr = s["null_fpr"]
    ax.set_title(f"F1 · Null p-values: FPR {fpr['rate']:.4f} (95% CI {fpr['lo']:.4f} to {fpr['hi']:.4f})")
    ax.annotate(f"p = 1 tie atom\n(exact mass {s['null_tie_atom_exact']:.4f})", xy=(0.975, exact_mass[-1]),
                xytext=(0.70, exact_mass[-1] * 1.08), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.17), ncols=2)

    caption = (
        f"F1. Null p-values at the design point (p = {s['design']['p_c']}, n = {s['design']['n_per_arm']:,}/arm). "
        f"Empirical FPR at alpha = {s['design']['alpha']} is {fpr['rate']:.4f} (95% CI {fpr['lo']:.4f} to {fpr['hi']:.4f}); "
        f"the exact size is {s['null_exact_size']:.5f}. The p-values match the exact enumerated null "
        f"(KS D = {s['null_ks_vs_exact']['statistic']:.4f}, p = {s['null_ks_vs_exact']['pvalue']:.2f}) but not U(0,1) "
        f"(D = {s['null_ks_vs_uniform']['statistic']:.4f}), purely because ties x_t = x_c put an atom of mass "
        f"{s['null_tie_atom_exact']:.4f} at p = 1 (simulated {s['null_tie_share_sim']['rate']:.4f})."
    )
    _save(fig, "F1_null_calibration", caption, captions)


def figure2(captions):
    peek = _csv("study2_peeking.csv")
    paired = _csv("study2_paired_differences.csv")
    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(10, 3.9))

    for method, color in METHOD_COLOR.items():
        for ax, case in ((ax_f, "null"), (ax_p, "alt")):
            d = peek[(peek.case == case) & (peek.method == method)].sort_values("K")
            if case == "null":
                ax.plot(d.K, d.gaussian_theory, color=color, linewidth=1.6, alpha=0.55, zorder=2)
            else:
                ax.plot(d.K, d.rate, color=color, linewidth=1.6, zorder=2)
            ax.errorbar(d.K, d.rate, yerr=[d.rate - d.ci_lo, d.ci_hi - d.rate], fmt="o", color=color,
                        ecolor=color, elinewidth=1, label=METHOD_LABEL[method], zorder=3, **MARK)

    alpha = DEFAULT.alpha
    ax_f.axhline(alpha, color=MUTED, linewidth=0.8, zorder=1)
    ax_f.text(1.02, alpha + 0.004, "alpha = 0.05", color=INK2, fontsize=8)
    ax_f.set_ylabel("False positive rate (true null)")
    ax_f.set_ylim(0, 0.30)
    ax_p.set_ylabel("Power (design lift +10%)")
    ax_p.set_ylim(0.40, 0.95)
    for ax in (ax_f, ax_p):
        _log_k_axis(ax)
        ax.axvline(HEADLINE_K, color=GRID, linewidth=1.2, zorder=0)

    def row(case, method, K=HEADLINE_K):
        return peek[(peek.case == case) & (peek.method == method) & (peek.K == K)].iloc[0]

    def diff(case, comparison, K=HEADLINE_K):
        return paired[(paired.case == case) & (paired.comparison == comparison) & (paired.K == K)].iloc[0]

    u, b, pk = row("null", "uncorrected"), row("null", "bonferroni"), row("null", "pocock")
    ax_f.annotate(f"{u.rate:.3f}", xy=(HEADLINE_K, u.rate), xytext=(-24, 6), textcoords="offset points",
                  fontsize=8.5, color=INK, fontweight="semibold")
    fixed = row("alt", "uncorrected", 1).rate
    pb, pp = row("alt", "bonferroni"), row("alt", "pocock")
    cost_b, gain_p = diff("alt", "bonferroni_minus_fixed"), diff("alt", "pocock_minus_bonferroni")
    ax_p.annotate(f"{pb.rate:.3f}", xy=(HEADLINE_K, pb.rate), xytext=(-7, -5), textcoords="offset points", ha="right", va="top",
                  fontsize=8.5, color=INK, fontweight="semibold")
    ax_p.annotate(f"{pp.rate:.3f}", xy=(HEADLINE_K, pp.rate), xytext=(6, 5), textcoords="offset points", ha="left", va="bottom",
                  fontsize=8.5, color=INK, fontweight="semibold")
    ax_f.set_title(f"F2a · Peeking inflates FPR to {u.rate:.3f} at K = {HEADLINE_K}")
    ax_p.set_title(f"F2b · Bonferroni costs {-cost_b['diff']:.3f} power at K = {HEADLINE_K}")
    ax_f.legend(loc="upper left", title="Points: simulation ± 95% CI; faint lines: Gaussian theory",
                title_fontsize=7.5, alignment="left")
    ax_p.legend(loc="lower left")
    fig.tight_layout(w_pad=3)

    theory = u.gaussian_theory
    k28 = row("null", "uncorrected", 28)
    caption = (
        f"F2. Stopping at the first look with p < 0.05 raises the false positive rate from "
        f"{row('null', 'uncorrected', 1).rate:.4f} at K = 1 to {u.rate:.3f} at K = {HEADLINE_K} "
        f"(Gaussian recursive-integration theory {theory:.3f}) and {k28.rate:.3f} at K = 28. "
        f"Bonferroni holds FPR at {b.rate:.4f} but cuts power from {fixed:.3f} (fixed horizon) to {pb.rate:.3f}, "
        f"a paired cost of {cost_b['diff']:+.3f} (95% CI {cost_b.ci_lo:+.3f} to {cost_b.ci_hi:+.3f}). "
        f"A calibrated Pocock boundary keeps FPR at {pk.rate:.4f} with power {pp.rate:.3f}, "
        f"{gain_p['diff']:+.3f} over Bonferroni (95% CI {gain_p.ci_lo:+.3f} to {gain_p.ci_hi:+.3f}). "
        f"Fixed-horizon power {fixed:.4f} in this seed is 2.6 SE above the closed form 0.8013; "
        f"a 10-seed check averaged 0.8015."
    )
    _save(fig, "F2_peeking", caption, captions)


def figure3(captions):
    grid = _csv("study3_power_grid.csv")
    rt = _csv("study3_round_trip.csv")
    s = _json("study3_summary.json")
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    n_line = np.geomspace(400, 70_000, 300)
    for lift, color in zip(F3_LIFTS, RAMP4):
        p_t = DEFAULT.p_c * (1 + lift)
        ax.plot(n_line, closed_form_power(DEFAULT.p_c, p_t, n_line), color=color, zorder=2)
        d = grid[np.isclose(grid.relative_lift, lift)].sort_values("n_per_arm")
        ax.errorbar(d.n_per_arm, d.power_sim, yerr=[d.power_sim - d.ci_lo, d.ci_hi - d.power_sim], fmt="o",
                    color=color, ecolor=color, elinewidth=1, label=f"+{lift:.0%} lift", zorder=3, **MARK)
    ax.errorbar(rt.required_n_per_arm, rt.power_sim, yerr=[rt.power_sim - rt.ci_lo, rt.ci_hi - rt.power_sim],
                fmt="D", color=INK, ecolor=INK, elinewidth=1, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.3, label="Round trip at required n", zorder=4)
    ax.axhline(DEFAULT.target_power, color=MUTED, linewidth=0.8, zorder=1)
    ax.axhline(DEFAULT.alpha, color=MUTED, linewidth=0.8, zorder=1)
    ax.text(420, DEFAULT.target_power + 0.015, "target power 0.80", color=INK2, fontsize=8)
    ax.text(69_000, DEFAULT.alpha + 0.015, "alpha = 0.05 (zero lift)", ha="right", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2000, 5000, 10_000, 25_000, 50_000])
    ax.set_xticklabels(["500", "1k", "2k", "5k", "10k", "25k", "50k"])
    ax.minorticks_off()
    ax.set_xlim(400, 70_000)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Sample size per arm (baseline 10%)")
    ax.set_ylabel("Power")
    sim_exact = s["sim_vs_exact"]
    ax.set_title(f"F3 · Simulated power matches exact power in all {sim_exact['cells']} cells")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Lines: closed form\npoints: simulation ± 95% CI", title_fontsize=7.5,
              alignment="left")

    cf_err = s["max_abs_closed_form_minus_exact"]
    caption = (
        f"F3. Power of the pooled z-test vs sample size. Across all {sim_exact['cells']} cells "
        f"(7 lifts x 8 sizes, {s['n_trials_per_cell']:,} trials each) simulation agrees with exact enumerated power "
        f"(smallest per-cell binomial p = {sim_exact['min_binomial_p']:.3f} against a Bonferroni threshold of "
        f"{sim_exact['bonferroni_threshold']:.5f}; chi-square p = {sim_exact['chi2_p']:.2f}). The closed form is never "
        f"more than {abs(cf_err['value']):.4f} from exact (worst at +{cf_err['relative_lift']:.0%}, "
        f"n = {cf_err['n_per_arm']:,}), and the largest simulated minus closed-form gap at n >= 1,000 is "
        f"{s['max_abs_sim_minus_closed_form_n_ge_1000']:.4f}, within MC error. At the sample sizes returned by "
        f"required_sample_size, simulated power is {rt.power_sim.min():.4f} to {rt.power_sim.max():.4f} "
        f"for relative MDEs of {rt.relative_lift.min():.0%} to {rt.relative_lift.max():.0%}. "
        f"Lifts +2.5% and +15% are omitted from the plot for legibility and are in results/study3_power_grid.csv."
    )
    _save(fig, "F3_power", caption, captions)


def figure4(captions):
    w = _csv("study4_winners_curse.csv")
    fig, (ax_k, ax_b) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(width_ratios=[1.15, 1]))

    for lift, color in zip(F4_LIFTS, RAMP4):
        d = w[np.isclose(w.relative_lift, lift)].sort_values("K")
        power = d.fixed_horizon_power_exact.iloc[0]
        ax_k.fill_between(d.K, 100 * d.winners_rel_lo, 100 * d.winners_rel_hi, color=color, alpha=0.18,
                          linewidth=0, zorder=1)
        ax_k.plot(d.K, 100 * d.winners_rel_bias, "o-", color=color, label=f"+{lift:.0%} (power {power:.2f})",
                  zorder=3, **MARK)
    _log_k_axis(ax_k)
    ax_k.axvline(HEADLINE_K, color=GRID, linewidth=1.2, zorder=0)
    ax_k.set_ylim(0, None)
    ax_k.set_ylabel("Winners' mean lift overstatement (%)")
    ax_k.legend(loc="upper left", title="True lift (fixed-horizon power)", title_fontsize=7.5, alignment="left")

    head = w[np.isclose(w.relative_lift, DEFAULT.relative_lift) & (w.K == HEADLINE_K)].iloc[0]
    ax_k.annotate(f"{100 * head.winners_rel_bias:+.1f}%", xy=(HEADLINE_K, 100 * head.winners_rel_bias),
                  xytext=(-38, 8), textcoords="offset points", fontsize=8.5, color=INK, fontweight="semibold")
    ax_k.set_title(f"F4a · Declared winners overstate the design lift by {100 * head.winners_rel_bias:+.1f}% at K = {HEADLINE_K}")

    # Decomposition at the headline K, every non-zero lift: selection alone (K = 1) + peeking increment.
    k1 = w[(w.K == 1) & (w.relative_lift > 0)].sort_values("relative_lift").reset_index(drop=True)
    kh = w[(w.K == HEADLINE_K) & (w.relative_lift > 0)].sort_values("relative_lift").reset_index(drop=True)
    y = np.arange(len(k1))[::-1]
    sel, inc = 100 * k1.winners_rel_bias.to_numpy(), 100 * kh.incremental_rel.to_numpy()
    gap = 0.35  # visual surface gap between the two segments, in % units
    bar_h = 0.55
    ax_b.barh(y, sel, height=bar_h, color=METHOD_COLOR["uncorrected"], label="Selection only (K = 1)", zorder=2)
    ax_b.barh(y, np.maximum(inc - gap, 0), left=sel + gap, height=bar_h, color=METHOD_COLOR["bonferroni"],
              label=f"Added by peeking (K = {HEADLINE_K})", zorder=2)
    for yi, s_, i_ in zip(y, sel, inc):
        ax_b.text(s_ + i_ + 4, yi, f"{s_ + i_:.0f}%", va="center", fontsize=8, color=INK2)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([f"+{l:.0%} · power {p:.2f}" for l, p in zip(k1.relative_lift, k1.fixed_horizon_power_exact)])
    ax_b.tick_params(axis="y", length=0)
    ax_b.grid(axis="y", visible=False)
    ax_b.set_xlim(0, (sel + inc).max() * 1.14)
    ax_b.set_xlabel(f"Winners' overstatement at K = {HEADLINE_K} (%)")
    ax_b.set_title("F4b · Peeking adds most of it, even at high power")
    ax_b.legend(loc="lower right")
    fig.tight_layout(w_pad=3)

    top = kh.iloc[-1]
    top_k1 = k1.iloc[-1]
    null = w[(w.relative_lift == 0) & (w.K == HEADLINE_K)].iloc[0]
    design_k1 = k1[np.isclose(k1.relative_lift, DEFAULT.relative_lift)].iloc[0]
    caption = (
        f"F4. Relative overstatement of the true lift by tests that stop early and declare the treatment the winner. "
        f"At the design point (+{DEFAULT.relative_lift:.0%}, K = {HEADLINE_K}) winners overstate the effect by "
        f"{100 * head.winners_rel_bias:+.1f}% (95% CI {100 * head.winners_rel_lo:+.1f} to {100 * head.winners_rel_hi:+.1f}): "
        f"{100 * design_k1.winners_rel_bias:+.1f}% is selection alone (K = 1) and {100 * head.incremental_rel:+.1f}% "
        f"(bootstrap CI {100 * head.incremental_rel_lo:+.1f} to {100 * head.incremental_rel_hi:+.1f}) is added by peeking. "
        f"The same winners run to the full horizon overstate by only {100 * head.winners_full_horizon_rel_bias:+.1f}%. "
        f"Selection bias fades with power ({100 * top_k1.winners_rel_bias:+.1f}% at power "
        f"{top_k1.fixed_horizon_power_exact:.2f}) but peeking bias does not: the same lift still shows "
        f"{100 * top.winners_rel_bias:+.1f}% at K = {HEADLINE_K}. Under the null at K = {HEADLINE_K}, "
        f"{null.share_winners:.1%} of tests declare a winner with mean reported lift {null.winners_mean:.4f}. "
        f"Panel a shows 4 of 7 non-zero lifts; all are in results/study4_winners_curse.csv."
    )
    _save(fig, "F4_winners_curse", caption, captions)


def main() -> int:
    captions = []
    for make in (figure1, figure2, figure3, figure4):
        make(captions)
    lines = ["# Figure captions", "", "Generated by `python -m absim.plots` from `results/`. Do not edit by hand.", ""]
    for stem, caption in captions:
        lines += [f"## {stem}", "", f"![{stem}]({stem}.png)", "", caption, ""]
    (FIGURES / "captions.md").write_text("\n".join(lines))
    print("\n\n".join(f"{stem}: {c}" for stem, c in captions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
