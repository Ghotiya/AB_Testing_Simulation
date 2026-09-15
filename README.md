# Quantifying the Cost of Bad Experiment Design

**A Monte Carlo study of peeking, power and effect-size bias in A/B testing**

Author: Agam Ghotiya · License: MIT

A/B tests fail quietly. A dashboard checked every morning, a sample size picked by feel,
or a test stopped the moment it "wins" all produce confident conclusions that are wrong
more often than the nominal 5% suggests. This project measures *how often* and *by how
much*. It simulates hundreds of thousands of experiments where the true effect is known,
then checks every result against exact theory.

The study is simulation only. That is deliberate: a real dataset gives one realization of
the world, while properties such as the false positive rate under peeking only show up
when the true effect is known (often zero, by construction) and the same experiment is run
thousands of times.

---

## Key findings

**Design point used throughout:** 10% baseline conversion, true +10% relative lift
(0.10 to 0.11), α = 0.05 two-sided, 14,800 users per arm. That is the 80%-power sample
size (14,751) rounded up. At about 1,057 users per arm per day it is a 14-day test, so
**checking the results every morning means K = 14 looks**.

| Question | Answer | |
|---|---|---|
| Is the simulator correct? | Null false positive rate **0.0503** (95% CI 0.0489 to 0.0516); the full p-value distribution matches the exact null (KS p = 0.83) | F1 |
| What does daily peeking cost? | False positives rise from 5% to **22.1%** (theory 22.0%), and to 27.7% with twice-daily checks | F2 |
| Does Bonferroni fix it? | Yes, but it overcorrects (FPR 2.15%) and costs **25.5 points of power** (0.806 to 0.551). A calibrated Pocock boundary holds 5% and wins back **12.4 points** | F2 |
| Is textbook power right? | Simulated power matches exact power in **56/56** grid cells; the closed form is never more than 0.0012 off; the sample-size calculator hits 0.799 to 0.802 power at four MDEs | F3 |
| Are early winners' effects trustworthy? | No. Tests stopped early on significance overstate a real +10% lift by **+62.9%**: +12.4% from selecting significant results, **+50.5% from peeking**. Run to full length, the same tests overstate by only +7.9% | F4 |

### 1. The simulator is calibrated

![F1](figures/F1_null_calibration.png)

Over 100,000 null experiments the false positive rate is 0.0503. The exact size of the
test, found by enumerating every outcome, is 0.05001. Design-point power is 0.8010 against
a closed form of 0.8013.

A useful subtlety came up here. A KS test of the p-values against U(0,1) *rejects*
(D = 0.0079), even though the simulator is correct. When both arms convert the same number
of users, z = 0 and p = 1 exactly. That spike has probability 0.0077 (0.0079 simulated).
It is a property of any test on counts, so validation uses the **exact enumerated null
distribution** instead, where KS gives p = 0.83. At small samples (200 per arm, 2%
baseline) the z-test is conservative, with exact size 0.0447, and the simulation
reproduces that (0.0445).

### 2. Peeking inflates false positives, and the fix has a price

![F2](figures/F2_peeking.png)

The analyst checks at K equally spaced points on one accumulating data stream and stops
at the first p < 0.05.
- **False positives:** 22.1% at K = 14, against Gaussian recursive-integration theory at
  22.0%. Every simulated point lies within 1.06 SE of theory.
- **Why not 51%?** Fourteen independent 5% tests would give 1 − 0.95¹⁴ = 51%. The looks
  share data, though: looks i and j are correlated √(nᵢ/nⱼ), about 0.96 for consecutive
  days near the end. So the inflation grows much more slowly than K.
- **Bonferroni** (α/K per look, |z| > 2.914 at K = 14) ignores that correlation. It
  pushes false positives down to 2.15% and costs −0.255 power (paired 95% CI −0.259 to
  −0.251).
- **Pocock** uses one constant threshold calibrated to exactly 5% (|z| > 2.615). It keeps
  false positives at 4.97% with power 0.674, +0.124 over Bonferroni. It is still −0.132
  below the fixed-horizon test: interim looks are never free.

### 3. Power and the sample-size calculator

![F3](figures/F3_power.png)

The grid is 7 relative lifts (0 to 30%) × 8 sample sizes (500 to 50,000 per arm), with
20,000 experiments per cell. Each cell is compared with **exact** power, enumerating both
binomial arms, and with the usual normal-approximation formula.
- **Agreement:** every cell passes an exact binomial test (smallest p = 0.016, Bonferroni
  threshold 0.00018; χ² p = 0.72).
- **The sample-size calculator** (`required_sample_size`) returns 57,763 / 14,751 /
  3,841 / 1,774 users per arm for MDEs of 5 / 10 / 20 / 30%. Simulating 100,000
  experiments at each size gives power 0.7991 / 0.8017 / 0.8010 / 0.8005.
- **Halving the MDE needs about 4× the users** (3.9× from 10% to 5%).

### 4. The winner's curse under early stopping

![F4](figures/F4_winners_curse.png)

A "declared winner" is a test that stops significant with the treatment ahead. Two
separate biases are measured on the same experiments:
- **Selection:** even with no peeking (K = 1), keeping only significant results keeps the
  experiments whose estimate came out high.
- **Peeking:** the extra bias from stopping early, with a bootstrap CI.

Findings:
- **At the design point with daily checks**, winners report the lift 62.9% too high (95%
  CI 62.2 to 63.7), so a real +10% is reported as about +16%. Of that, 12.4 points are
  selection and 50.5 are peeking (CI 49.8 to 51.3). Winners stop after 5,856 users per
  arm on average, 40% of the plan.
- **Bias is largest where power is lowest:** +276% at power 0.21 vs +47% at power 0.92
  (K = 14).
- **Peeking bias does not vanish at high power.** At 97% power, selection bias is only
  +1.6%, yet daily peeking still gives +37.5%. Early stops happen at small samples, where
  power at that look is low.
- **Under a true null**, daily peeking declares a winner 11.1% of the time, with an
  average reported lift of 0.0175, which is 1.75× the effect the test was designed to
  detect.
- **Wrong-sign results:** at +4% lift, 7.4% of significant results have the wrong sign
  with daily peeking, against 0.4% at a fixed horizon.

### The one-line story

Uncontrolled optional stopping causes two distinct failures from the same root cause. It
creates effects that do not exist, and it inflates effects that do. A pre-registered
sample size and a calibrated stopping boundary fix the first, at a measurable power cost.

---

## How the study is built

**Test statistic.** A pooled two-proportion z-test, two-sided, without continuity
correction, as production A/B platforms compute it. When the pooled rate is 0 or 1 the
standard error is 0, and the test returns p = 1 (not significant) rather than NaN.

**Simulation.**
- **Nested looks:** every experiment is one cumulative data stream built from binomial
  increments, so interim looks are nested on the same accumulating data. Drawing fresh
  samples at each look is the most common way to get a peeking study silently wrong: it
  makes the looks independent and exaggerates the inflation.
- **Common random numbers:** Studies 2 and 4 simulate each case once on the union of all
  look schedules. Every K and every stopping rule sees the same experiments, so all
  comparisons are paired.
- **Seeding:** reproducible with `numpy.random.SeedSequence` (master seed 20260912). Each
  study and configuration gets its own stream, so any cell can be re-run alone and
  reproduces bit for bit.
- **Two code paths:** a readable scalar `run_single_trial()` and a vectorized
  `run_trials()`, which a test requires to agree exactly.

**Monte Carlo sizes.** Every rate is reported with its MC standard error, and no claim is
smaller than about twice that error.

| Study | Experiments |
|---|---|
| 1 Baseline | 100,000 null + 100,000 alternative, plus 100,000 at small n |
| 2 Peeking | 50,000 null + 50,000 alternative, shared across K ∈ {1, 2, 3, 5, 7, 10, 14, 20, 28} and three stopping rules |
| 3 Power | 20,000 per grid cell (56 cells); 100,000 per round-trip MDE |
| 4 Winner's curse | 50,000 per lift {0, 4, 5, 6, 8, 10, 12, 14}% (power 0.21 to 0.97), shared across all K; 400 bootstrap resamples |

**Correctness gates.** Each study script exits with code 1 if its gate fails, and the
gates are designed to be able to fail.

| Gate | Checks |
|---|---|
| G0 (tests) | z-test matches statsmodels to < 1e-10 on 10,000 random inputs including edge cases; scalar and vectorized paths agree exactly; looks are nested; degenerate looks never give NaN |
| G1 | FPR CI contains 0.05; KS vs the exact null p > 0.01; power within ±0.01 of 0.80 |
| G2 | K = 1 reproduces Study 1; uncorrected FPR rises with K; Bonferroni stays ≤ α and falls with K; null curves within 3.5 SE of Gaussian theory; Pocock within 3.5 SE of α at every K; power cost has a paired CI |
| G3 | exact binomial test per cell against exact power, with a Bonferroni min-p across cells and a χ² test; round trip within ±0.01 at four MDEs; zero lift returns α |
| G4 | K = 1 winners' mean matches exact enumeration at every lift; selection bias > 0 at low power and < 5% at high power; bias falls with effect size at every K |

Three references keep the checks independent of the simulator:
- `absim/exact.py` enumerates both binomial arms for exact null distributions, exact power
  and exact winners' means.
- `absim/theory.py` computes Gaussian crossing probabilities by recursive integration and
  reproduces the published Armitage (1969) and Pocock (1977) tables in the tests.
- statsmodels is used only inside tests.

**Two lessons from building the gates.**
- "Within 2 SE at every one of 56 cells" fails about half the time by chance, which is
  why G3 uses exact tests with a multiple-comparison correction.
- The intuition "winner's-curse bias vanishes at K = 1 and at high power" is wrong on both
  counts: selection bias exists without peeking, and peeking bias survives high power. G4
  tests what is actually true.

---

## Running it

Requirements: Python 3.12 and about 400 MB for the environment. CPU only, no GPU, no
downloads. [uv](https://docs.astral.sh/uv/) is recommended, but plain `pip` works.

```bash
git clone https://github.com/Ghotiya/AB_Testing_Simulation.git
cd AB_Testing_Simulation

# environment (exact versions that produced the committed results)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
source .venv/bin/activate
#   without uv:  python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.lock

# everything: tests, the four gated studies, then the figures (~40 s on one core)
python -m studies.run_all
```

`run_all` prints one `== <step> ok` line per step and ends with `== ALL STEPS PASSED`. It
stops at the first failure. Seeding is deterministic, so the files it writes to
`results/` and the figure PNGs match the committed ones byte for byte.

Individual steps:

```bash
python -m pytest -q                        # 46 tests
python -m studies.study1_baseline          # G1   -> results/study1_*
python -m studies.study2_peeking           # G2   -> results/study2_*
python -m studies.study3_power             # G3   -> results/study3_*
python -m studies.study4_winners_curse     # G4   -> results/study4_*  (~25 s, the bootstrap)
python -m absim.plots                      # figures/ from results/ only, no re-simulation
```

On a shared machine, `scripts/run_bg.sh <name> <command...>` runs a job detached with
`nohup` at the lowest CPU priority, with no GPU visible and one BLAS thread. It logs to
`logs/<name>.log`:

```bash
scripts/run_bg.sh all python -m studies.run_all
tail -f logs/all.log
```

## Using the library

```python
from absim.config import DEFAULT
from absim.peeking import run_with_peeking
from absim.power import closed_form_power, required_sample_size
from absim.rng import make_rng

required_sample_size(mde=0.01, power=0.80, alpha=0.05, p_baseline=0.10)   # 14751 users per arm
closed_form_power(0.10, 0.11, 14_800)                                     # 0.8013

# 50,000 null experiments, checked daily for 14 days, stopping at the first p < 0.05
out = run_with_peeking(14, correction=None,          # or "bonferroni" / "pocock"
                       rng=make_rng("demo"), n_trials=50_000,
                       p_c=0.10, p_t=0.10, n_per_arm=DEFAULT.n_per_arm)
```

| Module | Contents |
|---|---|
| `absim/stats.py` | vectorized pooled two-proportion z-test |
| `absim/simulate.py` | `run_single_trial` (readable reference), `run_trials` (vectorized), look schedules, `TrialBatch` |
| `absim/peeking.py` | `run_with_peeking(num_looks, correction=None)`: uncorrected, Bonferroni, Pocock |
| `absim/power.py` | `closed_form_power`, `required_sample_size`, `power_sweep` |
| `absim/winners_curse.py` | `estimate_winners_curse_bias`, winners' summaries, bootstrap CIs |
| `absim/exact.py` | exact enumerated null distribution, exact power, exact winners' mean |
| `absim/theory.py` | Gaussian recursive integration for repeated looks, Pocock constants |
| `absim/mc.py` | rates and means with Monte Carlo CIs |
| `absim/plots.py` | figures F1 to F4, captions built from the result files |
| `absim/config.py`, `absim/rng.py` | design parameters, reproducible seeding |

## Repository layout

```
absim/        the simulation library
studies/      study1_baseline … study4_winners_curse (each gated), run_all
tests/        pytest suite (46 tests)
results/      CSV + JSON outputs of every study
figures/      F1–F4 as PNG and PDF, captions.md with the numeric caption of each
scripts/      run_bg.sh launcher
requirements.txt / requirements.lock
```

Result files:

| File | Contents |
|---|---|
| `study1_summary.json`, `study1_null_pvalue_hist.csv` | calibration, KS tests, tie mass, design power, small-n check |
| `study2_peeking.csv`, `study2_paired_differences.csv` | FPR and power per K and stopping rule with CIs and theory; paired differences |
| `study3_power_grid.csv`, `study3_round_trip.csv` | simulated vs exact vs closed-form power per cell; calculator round trip |
| `study4_winners_curse.csv` | per lift × K: winners' bias, selection vs peeking split, full-horizon counterfactual, wrong-sign share |
| `study*_summary.json` | each study's gate checks |

## Limitations and next steps

- **Scope of the setup:**
  - Proportions only, with a pooled z-test.
  - Users are i.i.d. with constant daily traffic: no day-of-week or novelty effects,
    network interference or sample-ratio mismatch.
  - Looks are equally spaced, with 50/50 allocation.
- **Not simulated:**
  - Sequential methods beyond Bonferroni and Pocock. O'Brien–Fleming / Lan–DeMets
    α-spending, which spend little α early, and mSPRT / always-valid p-values are the
    natural next steps.
  - Continuous metrics such as revenue per user. The peeking and selection mechanisms
    depend on the correlation between nested looks, which comes from the central limit
    theorem rather than the Bernoulli family, so they should carry over. Heavy-tailed
    metrics would be worse calibrated at early looks.
  - Real-data validation, by design.
