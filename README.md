# MARL-Structured-Language-Priors

This directory is a standalone extraction of the Chapter 5 prototype. It packages
the structured language-prior interface, tabular and deep high-level reinforcement
learners, and the maritime search-and-rescue simulator used for evaluation.

## Structure

- `planner/`: structured task graph schema and cognitive-prior adapter
- `policy/`: tabular and deep actor-critic high-level policy learners
- `env/`: safety-aware adapter on top of the simulator
- `simulator/msar_sim/`: maritime UAV-USV simulator extracted from Chapter 4
- `scripts/`: mock pipeline, prototype artifact generation, and formal experiments
- `results/`: CSV/JSON/TEX summaries
- `figures/`: generated figures for the paper

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_mock_pipeline.py
.venv/bin/python scripts/generate_prototype_results.py
.venv/bin/python scripts/run_formal_experiments.py
.venv/bin/python scripts/run_extension_experiments.py
.venv/bin/python scripts/run_reward_sensitivity_experiments.py
.venv/bin/python scripts/run_deep_rl_realism_experiments.py
```

If Python reports a certificate error when installing dependencies on macOS,
retry with the system certificate path:

```bash
SSL_CERT_FILE=/etc/ssl/cert.pem .venv/bin/python -m pip install -r requirements.txt
```

The scripts also work with an already configured Python environment:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_mock_pipeline.py
python3 scripts/generate_prototype_results.py
python3 scripts/run_formal_experiments.py
python3 scripts/run_extension_experiments.py
python3 scripts/run_reward_sensitivity_experiments.py
python3 scripts/run_deep_rl_realism_experiments.py
```

## Main Outputs

- `results/prototype_metrics.csv`
- `results/prototype_results_table.tex`
- `results/formal_results.csv`
- `results/formal_summary.json`
- `results/formal_results_table.tex`
- `figures/prototype_curves.png`
- `figures/formal_training_curves.png`
- `figures/formal_regime_comparison.png`
- `results/rule_shift_results.csv`
- `results/platform_failure_results.csv`
- `results/safety_ablation_results.csv`
- `results/llm_parse_eval.csv`
- `results/deep_rl_comparison.csv`
- `results/realism_layer_results.csv`
- `results/reward_sensitivity_results.csv`
- `results/marl_auv_quantitative_summary.csv`
- `figures/rule_shift_adaptation.png`
- `figures/platform_failure_comparison.png`
- `figures/safety_ablation.png`
- `figures/deep_rl_comparison.png`
- `figures/realism_layer_comparison.png`

## Notes

This code focuses on high-level decision interfaces rather than low-level vehicle
control. The contribution is the connection from language-derived cognitive priors
to safer high-level coordination decisions in heterogeneous maritime SAR.

## Current Evidence Boundary

The current formal experiment supports a safety and robustness claim, not a pure
speed claim. The full stack improves violation rate, formation quality,
communication load, and stress-regime composite score, while accepting longer
mission time than the fixed-rule baseline.

## Next Experiments

The submission-oriented extensions in `EXPERIMENT_EXTENSION_PLAN.md` are now
implemented in `scripts/run_extension_experiments.py`:

- rule-shift adaptation
- platform-failure recovery
- safety-projection ablation
- offline schema parsing evaluation for language-like SAR instructions

The Equation (4) weight-sensitivity check is implemented in
`scripts/run_reward_sensitivity_experiments.py`. It does not retrain policies;
it recomputes the reported stress-regime composite score from the raw formal
metrics under nominal, rescue-heavy, safety-heavy, communication-heavy, and
speed-first scalarizations.

The same extension script was also run on remote host `happy`; the key CSV files
match the local run up to floating-point formatting.

The deep-RL and realism-layer extension is implemented in
`scripts/run_deep_rl_realism_experiments.py`. It trains a PPO-style centralized
actor-critic over the high-level action bundle, compares it with the tabular full
stack, and evaluates a stress-regime perturbation layer with hydrodynamic load,
sensor error, and packet drops. This script was run on remote host `happy` with
the `happy` conda environment. PyTorch imported successfully; CUDA fell back to
CPU because the installed NVIDIA driver is older than the CUDA build bundled
with that environment.

The realism-layer script now writes the explicit perturbation protocol into
`results/deep_rl_realism_summary.json`, including the sea-state/communication
sampling formulas, action modifiers, and metric-transfer equations.
