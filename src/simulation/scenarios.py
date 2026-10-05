"""Policy comparison and what-if scenario runner for the digital twin.

Every policy / scenario is run on the SAME seeded patient streams (common
random numbers) over several replications, and results are reported as
mean +/- 95% confidence interval, so differences are real rather than noise.

Owner: P4 (Simulation Engineer)
Reference: MASTER_PROMPT §8.1, §8.2, §16 Cmd 9

Usage:
    python -m src.simulation.scenarios --policies all --tier tier_1 \\
        --days 5 --replications 10 --output reports/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.scheduler.policies import get_policy
from src.simulation.engine import RadiologySimulation
from src.simulation.entities import CONFIG_DIR, SimulationSettings
from src.utils.logger import get_logger

logger = get_logger(__name__)

POLICY_NAMES: list[str] = ["fcfs", "priority", "sjf", "wave", "radqueue_ai", "radqueue_noshow"]
POLICY_DISPLAY_NAMES: dict[str, str] = {
    "fcfs": "FCFS",
    "priority": "Priority",
    "sjf": "SJF",
    "wave": "Wave",
    "radqueue_ai": "RadQueue AI",
    "radqueue_noshow": "RadQueue + No-Show",
}
HEADLINE_KPIS: list[str] = [
    "avg_wait", "p90_wait", "max_wait", "std_wait", "emergency_avg_wait",
    "emergency_within_limit_rate", "urgent_within_limit_rate", "starvation_rate",
    "utilization", "patients_per_day", "avg_turnaround",
]
CI_Z: float = 1.96
DEFAULT_REPLICATIONS: int = 10
DEFAULT_DAYS: int = 5


def run_single(policy_name: str, settings: SimulationSettings, seed: int) -> dict[str, float]:
    """Run one replication of one policy and return its KPI summary."""
    sim = RadiologySimulation(get_policy(policy_name), settings=settings, seed=seed)
    summary = sim.run().summary()
    summary.update({"policy": policy_name, "seed": seed, "scenario": settings.scenario_name})
    return summary


def _aggregate(runs: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean and 95% CI half-width of every KPI per group."""
    kpis = [c for c in runs.columns if c not in {"policy", "seed", "scenario"} and runs[c].dtype != object]
    grouped = runs.groupby(group_col, sort=False)[kpis]
    mean = grouped.mean()
    ci = grouped.std(ddof=1).fillna(0.0) * CI_Z / np.sqrt(grouped.count().clip(lower=1))
    ci.columns = [f"{c}_ci95" for c in ci.columns]
    return pd.concat([mean, ci], axis=1).reset_index()


def run_comparison(
    policies: list[str] | None = None,
    settings: SimulationSettings | None = None,
    replications: int = DEFAULT_REPLICATIONS,
    base_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare scheduling policies on identical patient streams.

    Args:
        policies: Policy names (default: all six).
        settings: Department settings (default: Tier 1, 5 days).
        replications: Number of independent seeded replications.
        base_seed: First seed; replication r uses ``base_seed + r``.

    Returns:
        (per-run DataFrame, aggregated mean/CI DataFrame indexed by policy).
    """
    policies = policies or POLICY_NAMES
    settings = settings or SimulationSettings.from_config("tier_1", days=DEFAULT_DAYS)
    rows = [
        run_single(policy, settings, base_seed + r)
        for r in range(replications)
        for policy in policies
    ]
    runs = pd.DataFrame(rows)
    summary = _aggregate(runs, "policy")
    summary.insert(1, "policy_name", summary["policy"].map(POLICY_DISPLAY_NAMES))
    return runs, summary


def load_whatif_scenarios(config_dir: Path = CONFIG_DIR) -> list[dict[str, Any]]:
    """Return the what-if presets defined in simulation_config.yaml."""
    with open(config_dir / "simulation_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f).get("whatif_scenarios", [])


def run_whatif(
    scenario: dict[str, Any],
    policy: str = "radqueue_ai",
    settings: SimulationSettings | None = None,
    replications: int = DEFAULT_REPLICATIONS,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Compare a what-if scenario against the baseline under one policy.

    Args:
        scenario: Scenario dict (see ``SimulationSettings.with_scenario``).
        policy: Scheduling policy to use for both arms.
        settings: Baseline settings.
        replications: Seeded replications per arm.
        base_seed: First seed.

    Returns:
        DataFrame with one row per KPI: baseline, scenario, delta, delta_pct.
    """
    base = settings or SimulationSettings.from_config("tier_1", days=DEFAULT_DAYS)
    variant = base.with_scenario(scenario)
    rows = [
        run_single(policy, arm, base_seed + r)
        for r in range(replications)
        for arm in (base, variant)
    ]
    agg = _aggregate(pd.DataFrame(rows), "scenario").set_index("scenario")
    baseline_row, scenario_row = agg.loc[base.scenario_name], agg.loc[variant.scenario_name]
    kpis = [k for k in HEADLINE_KPIS if k in agg.columns]
    out = pd.DataFrame({
        "kpi": kpis,
        "baseline": [baseline_row[k] for k in kpis],
        "scenario": [scenario_row[k] for k in kpis],
    })
    out["delta"] = out["scenario"] - out["baseline"]
    out["delta_pct"] = np.where(out["baseline"].abs() > 0, 100 * out["delta"] / out["baseline"].abs(), np.nan)
    out.insert(0, "scenario_name", variant.scenario_name)
    return out


def save_comparison_figures(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save static (matplotlib) policy-comparison charts to ``output_dir``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("avg_wait", "Average queue wait (min)"),
        ("p90_wait", "90th percentile wait (min)"),
        ("emergency_avg_wait", "Emergency average wait (min)"),
        ("utilization", "Machine utilisation"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4.5))
    for ax, (kpi, title) in zip(axes, panels):
        if kpi not in summary.columns:
            continue
        ax.bar(summary["policy_name"], summary[kpi], yerr=summary.get(f"{kpi}_ci95"), capsize=4, color="#4C78A8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Scheduling policy comparison (mean ± 95% CI, common random numbers)")
    fig.tight_layout()
    path = output_dir / "scheduling_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [path]


def main() -> None:
    """CLI: run policy comparison + all what-if presets and write reports."""
    parser = argparse.ArgumentParser(description="Run digital-twin policy comparison and what-if scenarios")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "simulation_config.yaml",
                        help="Path to simulation_config.yaml (its directory must hold the other configs)")
    parser.add_argument("--policies", type=str, default="all", help="'all' or comma-separated policy names")
    parser.add_argument("--tier", type=str, default="tier_1", choices=["tier_1", "tier_2", "tier_3"])
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whatif-policy", type=str, default="radqueue_ai")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()

    policies = POLICY_NAMES if args.policies == "all" else [p.strip() for p in args.policies.split(",")]
    settings = SimulationSettings.from_config(args.tier, days=args.days, config_dir=args.config.parent)
    args.output.mkdir(parents=True, exist_ok=True)

    runs, summary = run_comparison(policies, settings, args.replications, args.seed)
    runs.to_csv(args.output / "simulation_runs.csv", index=False)
    summary.to_csv(args.output / "simulation_comparison.csv", index=False)
    logger.info(f"Policy comparison written to {args.output / 'simulation_comparison.csv'}")

    whatif = pd.concat(
        [run_whatif(s, args.whatif_policy, settings, args.replications, args.seed)
         for s in load_whatif_scenarios(args.config.parent)],
        ignore_index=True,
    )
    whatif.to_csv(args.output / "whatif_comparison.csv", index=False)
    logger.info(f"What-if results written to {args.output / 'whatif_comparison.csv'}")

    try:
        save_comparison_figures(summary, args.output / "figures")
    except ImportError as exc:
        logger.warning(f"matplotlib not installed; skipping figures: {exc}")

    cols = ["policy_name", "avg_wait", "p90_wait", "emergency_avg_wait", "emergency_within_limit_rate", "utilization"]
    print(summary[[c for c in cols if c in summary.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
