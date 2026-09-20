# UDDPG-Net

Experimental code associated with **UDDPG-Net** for reinforcement-learning-based resource allocation.

This repository contains the comparison baselines, controlled ablation experiments, reward-weight sensitivity analysis, exploration-noise analysis, interpretability utilities, and supporting simulation environments used in the UDDPG-Net study.

## Repository Structure

```text
UDDPG-Net/
├── README.md
├── requirements.txt
├── .gitignore
└── SRC/
    ├── ddpg_ga_keras3_same_env.py
    ├── sac_same_env.py
    ├── maddpg_keras3_same_env.py
    ├── structured_mapping_ablation_modified.py
    ├── fast_eam_weight_sensitivity.py
    ├── noise_ablation_same_env.py
    ├── run_shap_udddpg.py
    ├── Fairness_Energy_Aware_Resource_Allocation.py.py
    ├── PPO_Resource_Allocation.py
    ├── PPO_Resource_Allocation_Experiment.py
    ├── enviroment.py
    ├── envNEW.py
    ├── fadNEW.py
    ├── fading K varying.py
    ├── spe NEW.py
    ├── special_case.py
    └── energy_efficiency_comparison_ddpg.py
```

## Main Scripts

| Script | Purpose |
|---|---|
| `ddpg_ga_keras3_same_env.py` | DDPG-GA hybrid comparison baseline using Keras 3 / TensorFlow 2.x |
| `sac_same_env.py` | SAC comparison baseline under the same simplified environment |
| `maddpg_keras3_same_env.py` | MADDPG comparison baseline under matched experimental settings |
| `structured_mapping_ablation_modified.py` | Controlled ablation of the structured hybrid-action mapping used in UDDPG-Net |
| `fast_eam_weight_sensitivity.py` | Sensitivity analysis for Energy- and Fairness-Aware Mechanism (EAM) reward weights |
| `noise_ablation_same_env.py` | Exploration-noise ablation: no noise, Gaussian, OU, and Gaussian+OU |
| `run_shap_udddpg.py` | SHAP-based interpretability analysis for trained UDDPG-Net actor/critic checkpoints |
| `energy_efficiency_comparison_ddpg.py` | Energy-efficiency comparison/visualization utility |

The remaining scripts contain earlier DDPG/PPO experiments, physical/simulation environment utilities, and auxiliary experiments retained for reproducibility and comparison.

---

## 1. DDPG-GA Hybrid Baseline

The DDPG-GA baseline combines deterministic policy learning with critic-guided evolutionary refinement:

1. The DDPG actor generates a seed action.
2. A bounded genetic algorithm searches around the seed action.
3. The DDPG critic value \(Q(s,a)\) is used as GA fitness.
4. The highest-valued candidate is selected for execution.
5. The selected transition is stored in replay memory and used for DDPG updates.

The implementation includes a safety fallback: when the best GA candidate has a lower critic value than the original actor action, the original DDPG action is retained.

Default settings include:

- \(K=10\) users/links;
- 2000 training episodes;
- 100 steps per episode;
- replay-buffer capacity = 100,000;
- batch size = 64;
- actor learning rate = 0.002;
- critic learning rate = 0.004;
- \(\gamma=0.9\);
- \(\tau=0.01\);
- three hidden layers with 128 units per layer.

Run:

```bash
python SRC/ddpg_ga_keras3_same_env.py
```

---

## 2. Same-Environment Baselines

### SAC

```bash
python SRC/sac_same_env.py
```

### MADDPG

```bash
python SRC/maddpg_keras3_same_env.py
```

These scripts are designed to preserve comparable simulation settings and random-environment trajectories where applicable.

---

## 3. Structured Hybrid-Action Mapping Ablation

`structured_mapping_ablation_modified.py` evaluates the contribution of the structured hybrid-action mapping used in UDDPG-Net.

The manuscript-oriented setting uses:

- \(K=10\) active links;
- \(B=5\) acoustic subbands;
- raw action dimension

\[
K(2B+1)=110.
\]

The structured mapping contains:

1. power-control normalization;
2. softmax subband preference;
3. discrete subband selection;
4. selected power/subband coupling;
5. energy-feasibility projection.

A quick single-seed run can be launched with:

```bash
python SRC/structured_mapping_ablation_modified.py --single-seed
```

A multi-seed run can be launched with:

```bash
python SRC/structured_mapping_ablation_modified.py --seeds 0,1,2,3,4
```

Custom training settings can be provided, for example:

```bash
python SRC/structured_mapping_ablation_modified.py \
  --episodes 300 \
  --steps 100 \
  --eval-episodes 20 \
  --eval-steps 100
```

---

## 4. EAM Reward-Weight Sensitivity

The EAM sensitivity script examines the effects of throughput, energy, fairness, and feasibility-violation weights.

Run the fast profile:

```bash
python SRC/fast_eam_weight_sensitivity.py --profile fast
```

Run the paper-oriented profile:

```bash
python SRC/fast_eam_weight_sensitivity.py --profile paper
```

The script exports CSV summaries and figures including reward-weight and throughput-fairness trade-off results.

---

## 5. Exploration-Noise Ablation

The exploration-noise experiment compares:

- no exploration noise;
- Gaussian noise;
- Ornstein-Uhlenbeck (OU) noise;
- combined Gaussian + OU noise.

Run:

```bash
python SRC/noise_ablation_same_env.py
```

The variants share the same DDPG implementation, environment trajectories, reward definition, replay settings, and training horizon.

---

## 6. SHAP Analysis

`run_shap_udddpg.py` provides SHAP-based interpretation for trained UDDPG-Net actor and critic models.

The default network assumptions in this analysis script are:

- \(K=10\);
- state dimension = 24;
- action dimension = 10;
- hidden dimension = 128;
- Actor: 24-128-128-128-10;
- Critic: (24+10)-128-128-128-1.

The primary interpretation target is

\[
f_Q(s)=Q(s,\mu(s)).
\]

Example:

```bash
python SRC/run_shap_udddpg.py \
  --checkpoints checkpoint_seed0.pt checkpoint_seed1.pt \
  --states states_seed0.npz states_seed1.npz \
  --output-dir shap_results
```

The checkpoint and state paths should be replaced with the files generated by the corresponding UDDPG-Net training runs.

---

## 7. Simulation Environments and Legacy Experiments

The repository also contains earlier or auxiliary scripts:

- `enviroment.py`
- `envNEW.py`
- `fadNEW.py`
- `fading K varying.py`
- `spe NEW.py`
- `special_case.py`
- `Fairness_Energy_Aware_Resource_Allocation.py.py`
- `PPO_Resource_Allocation.py`
- `PPO_Resource_Allocation_Experiment.py`

Some of these scripts use legacy TensorFlow 1.x-style interfaces such as `tensorflow.compat.v1`, `tf.Session`, or `tf.layers`. They may require a separate legacy Python/TensorFlow environment and are not guaranteed to run under the Keras 3 environment used by the newer comparison scripts.

---

## 8. Installation

### Current TensorFlow/Keras experiments

Recommended:

- Python 3.10+
- TensorFlow 2.16+
- Keras 3
- NumPy
- SciPy
- Matplotlib
- pandas

Install:

```bash
pip install -r requirements.txt
```

### SHAP analysis

The provided `requirements.txt` also includes PyTorch and SHAP for the interpretability script.

### Legacy scripts

Older TensorFlow/PPO files should preferably be executed in a separate legacy environment because they rely on APIs that are not fully compatible with Keras 3.

---

## 9. Reproducibility

Several current experiment scripts explicitly separate random-number streams for:

- environment trajectories;
- policy exploration;
- replay-buffer sampling;
- genetic-algorithm search;
- model initialization.

The DDPG-GA implementation also stores fixed environment trajectories so repeated runs with the same environment seed can use the same state-transition sequence. This is intended to improve fairness across algorithm comparisons.

For publication-quality experiments, use the same:

- random seed set;
- training budget;
- environment trajectories;
- reward parameters;
- physical-layer parameters;
- evaluation protocol

as reported in the manuscript.

---

## 10. Generated Outputs

Depending on the script, generated files can include:

- `.npz` experiment results;
- periodic checkpoints;
- actor/critic weight files;
- CSV summary tables;
- training curves;
- ablation figures;
- SHAP importance tables and plots.

Most generated outputs are written under `data/` or a script-specific output directory.

---

## 11. Citation

If you use this repository in academic work, please cite the corresponding UDDPG-Net paper.

```bibtex
@article{UDDPGNet2026,
  title   = {UDDPG-Net},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026}
}
```

Please replace the placeholder bibliographic information with the final paper metadata after publication.

---

## 12. Code Availability

The repository is intended to support reproducibility, controlled comparison, ablation analysis, and interpretation of the UDDPG-Net experiments.

For questions about the code or experimental settings, please open an Issue in the GitHub repository.

Repository:

```text
https://github.com/wangjianpinghist/UDDPG-Net
```
