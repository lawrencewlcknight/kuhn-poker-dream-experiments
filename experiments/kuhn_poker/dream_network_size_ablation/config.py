"""Configuration for the DREAM network-size ablation."""

from pathlib import Path

from dream_poker.constants import (
    DEFAULT_SEEDS_5,
    EXPLOITABILITY_THRESHOLD,
    KUHN_AVERAGE_POLICY_VALUE_TARGET,
    KUHN_GAME_VALUE_P0,
    SMOKE_TEST_SEEDS,
    THESIS_SEEDS_10,
)


def network_variant(
    variant_id: str,
    label: str,
    layers: list[int],
    description: str,
) -> dict:
    """Create a variant that applies one hidden architecture to all DREAM networks."""
    return {
        "variant_id": variant_id,
        "label": label,
        "policy_network_layers": list(layers),
        "advantage_network_layers": list(layers),
        "baseline_network_layers": list(layers),
        "network_architecture": "x".join(str(width) for width in layers),
        "network_depth": len(layers),
        "network_max_width": max(layers),
        "network_hidden_units": sum(layers),
        "description": description,
    }


EXPERIMENT_CONFIG = {
    "experiment_name": "kuhn_poker_dream_network_size_ablation",
    "game_name": "kuhn_poker",
    "algorithm": "DREAM-style OpenSpiel network-size ablation",
    "num_iterations": 175,
    "num_traversals": 160,
    "evaluation_interval": 25,
    "policy_network_train_every": 25,
    "policy_network_train_steps": 100,
    "advantage_network_train_steps": 50,
    "baseline_network_train_steps": 50,
    "policy_network_layers": [32, 32],
    "advantage_network_layers": [32, 32],
    "baseline_network_layers": [32, 32],
    "learning_rate": 0.003,
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "batch_size_baseline": 1024,
    "advantage_memory_capacity": int(1e6),
    "strategy_memory_capacity": int(1e6),
    "baseline_memory_capacity": int(1e6),
    "epsilon": 0.06,
    "compute_exploitability": True,
    "isolate_policy_training_rng": True,
    "seeds": [1234, 2025, 31415],
    "optional_development_seeds_5": DEFAULT_SEEDS_5,
    "optional_thesis_seeds_10": THESIS_SEEDS_10,
    "kuhn_game_value_player_0": KUHN_GAME_VALUE_P0,
    "average_policy_value_target": KUHN_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "output_root": Path("outputs") / "dream_network_size_ablation",
}


BASELINE_VARIANT = "arch_2x32_exp_baseline"


NETWORK_SIZE_VARIANTS = [
    network_variant(
        "arch_1x16",
        "1x16",
        [16],
        "Tiny shallow network",
    ),
    network_variant(
        "arch_2x16",
        "2x16",
        [16, 16],
        "Same depth as baseline with narrower hidden layers",
    ),
    network_variant(
        "arch_1x32",
        "1x32",
        [32],
        "Baseline width with one hidden layer",
    ),
    network_variant(
        "arch_2x32_exp_baseline",
        "2x32",
        [32, 32],
        "Aligned DREAM baseline architecture",
    ),
    network_variant(
        "arch_2x64",
        "2x64",
        [64, 64],
        "Same depth as baseline with wider hidden layers",
    ),
    network_variant(
        "arch_2x128",
        "2x128",
        [128, 128],
        "Very wide two-hidden-layer network",
    ),
    network_variant(
        "arch_3x32",
        "3x32",
        [32, 32, 32],
        "Baseline width with an additional hidden layer",
    ),
    network_variant(
        "arch_3x64",
        "3x64",
        [64, 64, 64],
        "Deeper and wider high-capacity network",
    ),
]


SMOKE_TEST_CONFIG_OVERRIDES = {
    "seeds": SMOKE_TEST_SEEDS,
    "num_iterations": 10,
    "num_traversals": 50,
    "policy_network_train_steps": 20,
    "advantage_network_train_steps": 20,
    "baseline_network_train_steps": 20,
    "policy_network_train_every": 5,
    "evaluation_interval": 5,
    "output_root": Path("outputs") / "smoke_tests" / "dream_network_size_ablation",
}
