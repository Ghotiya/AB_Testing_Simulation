"""Design parameters every study inherits. Rationale for each value is in README.md."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DesignConfig:
    p_c: float = 0.10  # baseline conversion rate
    relative_lift: float = 0.10  # design MDE, relative
    alpha: float = 0.05  # two-sided
    target_power: float = 0.80
    n_per_arm: int = 14_800  # closed-form 14,751 rounded up

    @property
    def p_t(self) -> float:
        return self.p_c * (1 + self.relative_lift)

    @property
    def delta(self) -> float:
        return self.p_t - self.p_c

    def with_(self, **changes) -> "DesignConfig":
        return replace(self, **changes)


DEFAULT = DesignConfig()

# Number of equally spaced interim looks swept in Studies 2 and 4.
LOOK_COUNTS = (1, 2, 3, 5, 7, 10, 14, 20, 28)
