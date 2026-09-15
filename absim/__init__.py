"""absim: Monte Carlo simulation of two-arm binomial A/B tests under known ground truth."""

from absim.config import DEFAULT, LOOK_COUNTS, DesignConfig
from absim.power import closed_form_power, required_sample_size
from absim.rng import make_rng
from absim.simulate import TrialBatch, look_schedule, run_single_trial, run_trials
from absim.stats import two_proportion_ztest

__all__ = [
    "DEFAULT",
    "LOOK_COUNTS",
    "DesignConfig",
    "TrialBatch",
    "closed_form_power",
    "look_schedule",
    "make_rng",
    "required_sample_size",
    "run_single_trial",
    "run_trials",
    "two_proportion_ztest",
]
