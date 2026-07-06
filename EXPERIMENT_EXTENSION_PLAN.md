# Experiment Extension Plan

This file lists the minimum experiments needed to turn the Chapter 5 prototype
into a stronger standalone paper.

## Current Reproducible Baseline

Run from `第3项工作/code`:

```bash
python3 scripts/run_mock_pipeline.py
python3 scripts/generate_prototype_results.py
python3 scripts/run_formal_experiments.py
```

Current outputs:

- `results/prototype_metrics.csv`
- `results/formal_results.csv`
- `results/formal_summary.json`
- `figures/prototype_curves.png`
- `figures/formal_training_curves.png`
- `figures/formal_regime_comparison.png`

The current formal experiment supports this claim:

> The full stack improves safety and robustness, especially in stress regimes,
> but it does not minimize mission time.

## E1 Rule-Shift Adaptation

Goal: evaluate whether cognitive priors help when operational rules change.

Status: implemented and executed in `scripts/run_extension_experiments.py`.

Implementation sketch:

1. Add a new script `scripts/run_rule_shift_experiments.py`.
2. Train for 420 episodes.
3. At episode 210, change the schema reward template from search-oriented to
   protection-oriented.
4. Measure adaptation steps, post-shift score, and violation rate.

Expected outputs:

- `results/rule_shift_results.csv`
- `figures/rule_shift_adaptation.png`

## E2 Platform-Failure Recovery

Goal: evaluate role reallocation under UAV/USV failures.

Status: implemented and executed in `scripts/run_extension_experiments.py`.

Implementation sketch:

1. Add a wrapper in `env/chapter4_adapter.py` or a new script that removes one
   UAV or one USV from each evaluation scenario.
2. Evaluate standard and stress regimes separately.
3. Compare fixed rule, RL only, LLM rule only, and full stack.

Expected outputs:

- `results/platform_failure_results.csv`
- `figures/platform_failure_comparison.png`

## E3 Safety-Projection Ablation

Goal: isolate whether the execution-layer safety projection contributes real
benefit.

Status: implemented and executed in `scripts/run_extension_experiments.py`.

Compare:

- cognitive prior + Q-learning + safety projection
- cognitive prior + Q-learning without safety projection
- Q-learning + safety projection without cognitive prior
- Q-learning only

Primary metrics:

- violation rate
- formation error
- rescue success
- composite score

Expected outputs:

- `results/safety_ablation_results.csv`
- `figures/safety_ablation.png`

## E4 Live LLM Parsing Evaluation

Goal: replace the static task schema with real language-to-JSON parsing
evidence.

Status: partially implemented as an offline schema parsing sanity check in
`scripts/run_extension_experiments.py`. A live API-backed parser is still future
work.

Implementation sketch:

1. Prepare 20-50 maritime SAR instructions in `planner/prompts/`.
2. Ask the LLM to emit JSON matching `planner/task_graph_schema.json`.
3. Validate JSON fields and count conflicts.
4. Report task-node accuracy, hard-constraint accuracy, reward-template
   consistency, and invalid-output rate.

Expected outputs:

- `results/llm_parse_eval.csv`
- `results/llm_parse_summary.json`

## Submission-Oriented Claim Boundary

Use these claims:

- The framework provides a schema-validated interface from language-derived
  task priors to high-level SAR coordination.
- Cognitive priors improve stress-regime safety and robustness when coupled with
  learning and safety projection.
- The method trades a small amount of mission speed for better formation quality
  and lower violation rate.

Avoid this claim:

- The method is globally faster than fixed rules.
