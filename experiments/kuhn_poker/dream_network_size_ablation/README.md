# DREAM network-size ablation

This experiment tests how DREAM performance in Kuhn poker changes as the hidden-layer architecture is varied while holding the rest of the aligned baseline training protocol fixed.

Each variant applies the same hidden architecture to all three DREAM network families: the average-policy network, both player-specific advantage networks, and both player-specific learned baseline/control-variate networks.

The default variants are:

- `arch_1x16`: hidden layers `[16]`;
- `arch_2x16`: hidden layers `[16, 16]`;
- `arch_1x32`: hidden layers `[32]`;
- `arch_2x32_exp_baseline`: hidden layers `[32, 32]`, matching the aligned DREAM baseline;
- `arch_2x64`: hidden layers `[64, 64]`;
- `arch_2x128`: hidden layers `[128, 128]`;
- `arch_3x32`: hidden layers `[32, 32, 32]`;
- `arch_3x64`: hidden layers `[64, 64, 64]`.

The output tables include depth, maximum width, total hidden units, and parameter counts for the constructed PyTorch networks.

For cloud runs, avoid running all variants as one Batch job. The full sweep can exceed a 48-hour Batch timeout. Use named variant sets instead:

- `width_sweep`: `arch_2x16`, `arch_2x32_exp_baseline`, `arch_2x64`, `arch_2x128`;
- `depth_sweep`: `arch_1x32`, `arch_2x32_exp_baseline`, `arch_3x32`;
- `capacity_extremes`: `arch_1x16`, `arch_2x32_exp_baseline`, `arch_3x64`.

Each set includes the baseline variant so paired-difference summaries remain valid.

## Run

```bash
python -m experiments.kuhn_poker.dream_network_size_ablation.run
```

## Split cloud-sized runs

```bash
python -m experiments.kuhn_poker.dream_network_size_ablation.run \
  --variant-set width_sweep \
  --output-root outputs/cloud/dream_network_size_width

python -m experiments.kuhn_poker.dream_network_size_ablation.run \
  --variant-set depth_sweep \
  --output-root outputs/cloud/dream_network_size_depth

python -m experiments.kuhn_poker.dream_network_size_ablation.run \
  --variant-set capacity_extremes \
  --output-root outputs/cloud/dream_network_size_capacity_extremes
```

## Smoke test

```bash
python -m experiments.kuhn_poker.dream_network_size_ablation.run \
  --seeds 1234,2025 \
  --iterations 10 \
  --traversals 50 \
  --policy-network-train-steps 20 \
  --advantage-network-train-steps 20 \
  --baseline-network-train-steps 20 \
  --evaluation-interval 5 \
  --variant-set capacity_extremes \
  --output-root outputs/smoke_tests/dream_network_size_ablation
```

Key outputs include `checkpoint_curves_by_variant.csv`, `seed_variant_summary.csv`, `aggregate_summary_by_variant.csv`, `paired_differences_vs_baseline.csv`, `paired_difference_summary.json`, `multiseed_curves_by_variant.npz`, and plots under `plots/`.
