# Vision-Based Autonomous Driving via Deep Reinforcement Learning in Continuous & Discrete Action Spaces

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0+-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-v2.0+-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end continuous-to-discrete deep reinforcement learning system trained on raw raster visual feeds within Gymnasium's Box2D racing simulator (`CarRacing-v3`). Optimized using Proximal Policy Optimization (PPO), this project details the iterative progression across four architectural generations—diagnosing and overcoming full-throttle policy saturation, reward shaping pathologies, and steering latency to produce a robust production agent with a **0.0% failure rate** and consistent **900+ environment-solved performance**.

---

## Visual Demonstrations

### Failure vs. Production Mastery

| V1 Baseline: Continuous Throttle Saturation | V4 Production: Threshold Braking & Apex Clipping |
| :---: | :---: |
| <img src="assets/old_videos/eval_run-episode-0.mp4" width="360" alt="V1 failure" /> | <img src="assets/v4_videos/v4_showcase.gif" width="360" alt="V4 Showcase" /> |
| *V1 locks throttle into Turn 1, slides off into grass, and spins indefinitely.* | *V4 executes active 5% braking, traces racing lines, and completes circuits in ~770 frames.* |

*A side-by-side synchronized comparison is compiled at `assets/v4_diagnostics/v1_vs_v4_comparison.gif`.*

---

## 1. Quantitative Performance Matrix

Empirical benchmarks collected deterministically across unseen procedural track seeds:

| Evaluation Metric | V1 Baseline | V2 Shaped Continuous | V3 FrameSkip Ablation | V4 Production (Discrete) |
| :--- | :---: | :---: | :---: | :---: |
| **Action Space** | Continuous R^3 | Continuous R^3 | Continuous R^3 | **Discrete (5 Primitives)** |
| **Physics Horizon** | 60 Hz (1 frame) | 60 Hz (1 frame) | 30 Hz (k=2 skip) | **60 Hz (1 frame)** |
| **Mean Reward (± std)** | 12.51 (Collapsed) | 681.04 ± 263.78 | 573.65 ± 282.40 | **833.48 ± 111.04** |
| **Score Std Dev (std)** | N/A | ± 263.78 | ± 282.40 | **± 111.04 (-57.9% variance)** |
| **Median Reward** | 12.51 | 839.39 | 645.34 | **885.12** |
| **Peak Single Lap** | 403.30 | 934.00 | 892.96 | **929.30** |
| **Worst-Case Floor** | < 0.0 | 65.52 | 18.38 | **569.70 (Zero spinouts)** |
| **>= 800 High Performance** | 0.0% | 60.0% | 46.7% | **75.0% - 85.0%** |
| **>= 900 Solved Threshold** | 0.0% | 6.7% | 6.7% | **30.0% - 40.0%** |
| **Braking Execution Rate** | 0.002 | 0.017 (feigned) | 0.017 | **4.96% (921 active pulses)** |
| **Episode Length on >= 900** | Never cleared | 1000 steps | 500 steps | **768 - 859 steps (True completion)** |
| **Critic Explained Variance** | ~ 0.55 | ~ 0.65 | ~ 0.60 | **0.88 - 0.93** |

---

## 2. Iterative Engineering Progression

```text

[V1: Continuous Baseline - 602k Steps]
  ├── Observation: (4 x 84 x 96) CHW Grayscale Stack
  └── Flaw: Full-throttle saturation; validation collapsed from peak 403.30 down to 12.51.
         │
         ▼
[V2: Centrifugal Continuous Shaping - 627k Steps]
  ├── Intervention: Angular slip penalties + stagnation truncation.
  └── Flaw: Discovered throttle lift-off (~0.36 gas) rather than braking; failed acute hairpins (65 pts).
         │
         ▼
[V3: Frame-Skipping Experiment (k=2) - 150k Steps]
  ├── Hypothesis: Extend lookahead horizon to allow early deceleration.
  └── Result: Steering latency caused inner-curb overcorrection; median dropped to 645.34.
         │
         ▼
[V4: Discrete Production Policy - 1,000,000 Steps]
  ├── Architecture: Discrete 5-action space + 60 Hz control + LR schedule + Backtrack Callback.
  └── Result: Active braking reached 4.96%, mean return rose to 833.48, worst track rose to 569.70.
```

---

## 3. Failure Mode & Root Cause Analysis

### V1 Policy Collapse: The Full-Throttle Local Minimum
In early training, high policy entropy allowed V1 to explore moderate speeds, reaching a validation peak of **`403.30`** at step 230,000. However, because straightaways grant positive rewards for rapid tile visits while grass penalties are delayed, gradient ascent pushed the actor head into saturation: **100% continuous throttle ($gas \approx 1.0$)**. 

During training rollouts, exploratory action noise $\sigma \cdot \epsilon$ occasionally saved the car from spinning. During deterministic evaluation ($\mu(s)$), the absence of noise caused the car to blast full-throttle into Turn 1, skid into the grass within 80 frames, and accumulate continuous step penalties ($-0.1$/frame) for the remainder of the 1,000-step episode—collapsing validation returns to **`12.51`**.

### V2 & V3 Limitations: The Continuous Action Trap & Steering Latency
* **V2 (Reward Shaping):** Penalizing high-speed steering successfully curbed full-throttle behavior, bringing average gas down to `0.363` and boosting median return to `839.39`. However, the agent discovered that **lifting off the throttle** was sufficient to avoid the penalty on wide curves, completely avoiding the brake (`brake = 0.017`). On acute $180^\circ$ hairpins following straights, lift-off alone could not shed enough velocity, leading to severe outlier scores (`65.52`, `252.11`).
* **V3 (Frame Skipping $k=2$):** Repeating continuous actions across 2 physics frames ($33.3\text{ ms}$) doubled temporal lookahead but introduced severe steering latency. Holding near-maximum steering ($\vert{}steer\vert{} = 0.934$) across multi-frame physics integration caused the vehicle to hook into inner curbs on S-bends, dropping median reward down to `645.34`.

---

## 4. The V4 Discrete Action Space Architecture

To overcome the continuous local minimum where throttle release substitutes for braking, V4 reformulates vehicle control into **5 discrete primitive actions**:

$$\mathcal{A} = \{\text{Coast}, \text{Steer Left}, \text{Steer Right}, \text{Full Gas}, \text{Threshold Brake}\}$$

| Action Index | Primitive Command | Vector Sent to Box2D Engine | Execution Ratio (20 Tracks) |
| :---: | :--- | :---: | :---: |
| `0` | Coast / Do Nothing | `[ 0.0, 0.0, 0.0]` | **6.01%** |
| `1` | Steer Left | `[-1.0, 0.0, 0.0]` | **9.95%** |
| `2` | Steer Right | `[ 1.0, 0.0, 0.0]` | **16.83%** |
| `3` | Full Gas | `[ 0.0, 1.0, 0.0]` | **62.25%** |
| `4` | Threshold Brake | `[ 0.0, 0.0, 0.8]` | **4.96%** |

### Why Categorical PPO Solved the Environment
1. **Explicit Threshold Braking:** By removing the continuous Gaussian covariance between gas and brake, PPO easily assigned positive advantage to discrete braking pulses before high-curvature entries. The agent executed **921 active braking commands** (4.96% of all actions), completely eliminating corner fly-offs.
2. **True Circuit Completion:** In earlier versions, high-scoring laps survived all 1,000 steps without finishing the loop. In V4, top laps cross all $N$ track tiles and terminate early on victory in **768 to 859 steps**.
3. **Critic Stability:** Categorical Softmax transitions dramatically reduced environment stochasticity, driving the critic network's **explained variance up to `0.88 - 0.93`**.

---

## 5. Empirical Performance Diagnostics

Deterministic evaluation across 20 unseen procedural tracks (`src/v4_diagnostics.py`):

![V4 Performance Profile](assets/v4_diagnostics/v4_performance_profile.png)

```text
=================================================================
                 V4 PRODUCTION SUMMARY REPORT
=================================================================
  * Total Tracks Tested     : 20
  * Mean Reward             : 833.48 +/- 111.04
  * Median Reward           : 885.12
  * Peak Lap Score          : 929.30
  * Minimum Lap Score       : 569.70
  * 900+ Solved Rate        : 40.0% (8/20)
  * 800+ High Performance   : 75.0% (15/20)
  * 500-799 Intermediate    : 25.0% (5/20)
  * Under 500 Failures      : 0.0% (0/20)
  * Mean Steps per Episode  : 927.8 / 1000

[Discrete Action Usage Breakdown]
  * Action 0 (Coast      ):   1115 ( 6.01%)
  * Action 1 (Steer Left ):   1846 ( 9.95%)
  * Action 2 (Steer Right):   3123 (16.83%)
  * Action 3 (Gas        ):  11550 (62.25%)
  * Action 4 (Brake      ):    921 ( 4.96%)
=================================================================
```

---

## 6. System Architecture & MDP Formulation

```text
[Raw Visual Feed: 96x96x3 RGB]
               │
               ▼
     [CropDashboard Wrapper]    ──> Slices bottom 12px instrumentation ──> (84x96x3)
               │
               ▼
     [GrayScaleObservation]     ──> Luminance conversion (ITU-R 601-2) ──> (84x96x1)
               │
               ▼
     [VecFrameStack (k=4)]      ──> 4-frame temporal velocity buffer   ──> (84x96x4)
               │
               ▼
     [VecTransposeImage]        ──> Channel-first PyTorch ordering     ──> (4x84x96)
               │
               ▼
     [Nature CNN Backbone]      ──> 3 Conv2D layers (32@8x8/4, 64@4x4/2, 64@3x3/1)
               │
               ├───> [Categorical Policy Head] ──> Softmax over 5 Discrete Actions
               └───> [Critic Value Head]       ──> Expected Return V(s)
```

### Hardware-Tuned Training Configuration (AMD Ryzen 5 3600 / 16GB RAM)

* **Parallel Environments (`n_envs`):** 6 workers (1-to-1 pinned to physical CPU cores, avoiding context-switching latency)
* **Rollout Horizon (`n_steps`):** 1,024 steps per worker (total rollout batch: 6,144 steps)
* **Mini-Batch Size:** 128 (48 minibatches per epoch, sized to fit CPU L3 cache)
* **Optimization Epochs:** 10
* **Learning Rate Schedule:** Linear annealing from $\alpha = 2.5 \times 10^{-4} \to 5.0 \times 10^{-5}$
* **Entropy Regularization:** $\beta = 0.01$ (annealed to preserve late-stage racing lines)
* **Discount Factor ($\gamma$):** 0.99
* **GAE Coefficient ($\lambda$):** 0.95
* **PPO Clip Range ($\epsilon$):** 0.2
* **Max Gradient Norm:** 0.5

---

## 7. Repository Layout

```text
rl-car-racing/
├── src/
│   ├── wrappers.py              # Visual preprocessing (CropDashboard, Grayscale, FrameSkip)
│   ├── train.py                 # Multi-worker discrete PPO production pipeline with backtracking
│   ├── evaluate.py              # Deterministic evaluator & video recorder
│   ├── diagnostics.py           # V2/V3 continuous performance profiler
│   ├── v4_diagnostics.py        # 20-track production benchmark & action usage analyzer
│   ├── export_metrics.py        # TensorBoard event parser & evaluation metric extractor
│   └── export_comparison.py     # Multi-run progression visualizer
├── assets/
│   ├── v4_videos/
│   │   ├── v4_showcase.gif      # High-speed production demonstration
│   │   └── *.mp4                # Recorded evaluation runs
│   ├── v4_diagnostics/
│   │   ├── v4_performance_profile.png
│   │   ├── overall_training_progression.png
│   │   └── v1_vs_v4_comparison.gif
│   ├── old_videos/              # V1 baseline failure recordings
│   └── videos/                  # V2 intermediate evaluation recordings
├── models/
│   ├── best_model/              # Converged V1 baseline weights
│   ├── v2_best/                 # Converged V2 continuous shaped weights
│   └── v4_prod/
│       ├── best_model.zip       # Peak validation model (833.48 mean, 929.30 peak)
│       └── final_model.zip      # 1,000,000-step converged final weights
├── requirements.txt
└── README.md
```

---

## 8. Quickstart & Reproducibility

### Setup

```powershell
# Clone repository
git clone https://github.com/<your-username>/rl-car-racing.git
cd rl-car-racing

# Create virtual environment (Python 3.11 required)
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Reproduce V4 Production Benchmark

Run the 20-track diagnostic suite against the production model:
```powershell
python src/v4_diagnostics.py --model ./models/v4_prod/best_model.zip --episodes 20
```

Record clean evaluation laps to `assets/v4_videos/`:
```powershell
python src/evaluate.py --model ./models/v4_prod/best_model.zip --episodes 3 --record
```

Generate multi-run comparison plots across all logged experiments:
```powershell
python src/export_comparison.py
```

### Train from Scratch

To retrain the V4 discrete production agent using hardware-optimized worker allocations:
```powershell
python src/train.py --n_envs 6 --timesteps 1000000 --target_rate 0.80
```