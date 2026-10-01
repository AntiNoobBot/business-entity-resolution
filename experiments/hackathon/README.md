# Historical experiment workspace

These Python scripts are byte-identical copies of the source available at archival time. Numbered files `00`–`23` form the early data/EDA/blocking/feature/model workflow. `norm.py` and `metric.py` are shared helpers. `state_map.json` is the learned aggregate alias map.

The later `upgrade_trial_02/src/` files keep their original relative nesting. Many scripts expect this directory to be the working directory or derive this directory from their location. The corresponding data, environments, models, and predictions are absent.

Start with the [reproduction guide](../../docs/reproducibility.md). Reports are stored separately in [reports/historical](../../reports/historical/) so exploratory runs do not overwrite the evidence archive. Do not interpret script availability as end-to-end reproducibility or as proof that an experiment reached the scored submission.

