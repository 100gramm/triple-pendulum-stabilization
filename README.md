**Project Title**

TriplePendulum — Vision-Guided Stabilization of a Triple Inverted Pendulum

Slogan: Research-grade reinforcement learning and vision pipeline for robust swing-up and stabilization of a three-link inverted pendulum on a cart.

--------------------------------------------------------------------------------

**Summary**

This repository implements a research and engineering pipeline for the swing-up and stabilization of a triple inverted pendulum using model-based simulation (MuJoCo), a vision-based perception stack, and a policy-gradient RL controller (PPO). The project focuses on combining visual state estimation and reward engineering with curriculum learning to obtain robust policies that can perform both swing-up and long-term stabilization in several target configurations.

This README documents: the problem definition, architecture, reward design, curriculum, training process, experimental results, limitations, and steps to reproduce experiments.

--------------------------------------------------------------------------------

**Contents**

- **Introduction & Motivation**
- **Demonstration (placeholders for GIFs / videos / screenshots)**
- **Architecture (Mermaid diagram + descriptions)**
- **Environment & Simulation**
- **Observation / Action Spaces**
- **MDP / Mathematical Formulation**
- **Reward Function (detailed decomposition and formulas)**
- **Curriculum Learning**
- **Training Procedure & Hyperparameters**
- **Hardware & Performance**
- **Results**
- **Limitations & Open Problems**
- **Roadmap**
- **Repository Structure**
- **Installation & Reproduction Instructions**
- **References & Acknowledgements**

--------------------------------------------------------------------------------

**1. Problem Statement**

Goal: design and train an agent that controls a cart-mounted triple inverted pendulum so that the system can be swung up from arbitrary initial conditions and stabilized at a set of predefined target configurations. The agent receives observations (either a compact state vector or, in the vision pipeline, raw images) and must produce a continuous torque command to the cart.

Why is this hard?

- The triple inverted pendulum is highly nonlinear and exhibits chaotic dynamics; small changes in initial conditions can strongly affect outcomes.
- Swing-up requires purposeful energy injection followed by precise damping to transition into a holding region around the upright targets.
- Visual-only control introduces partial observability, perception noise, and delays that complicate control compared to direct state feedback.

Reinforcement Learning rationale:

- RL supports learning of closed-loop control policies from sparse or shaped reward signals without deriving explicit analytic control laws for highly nonlinear dynamics.
- Policy-gradient methods (PPO) provide a stable and scalable approach for continuous control using parallelized environment rollouts.

System overview (high level):

- Simulator: MuJoCo XML model defines the cart and three serial pendulum links.
- Agent: PPO policy (stable-baselines3) trained with vectorized environments and VecNormalize for stable learning.
- Perception: a vision pipeline (dataset generator + ShuffleNet-based regressor) is used to map images to geometric features / joint-angle estimates for alternative training approaches.
- Reward engineering: a multi-term reward shaping that separates swing-up incentives, energy bonuses, hold rewards, and penalties.

What success looks like:

- The agent can reliably swing up and stabilize the three links in a target configuration with a high success rate (> 70% in evaluation under standard conditions).
- The policy does not regularly violate safety constraints (cart limits, extreme velocities) and keeps episodes stable once captured.

--------------------------------------------------------------------------------

**2. Demonstration**
There will be gifs or videos here soon

--------------------------------------------------------------------------------

**3. Architecture**

Mermaid diagram (high-level):

```mermaid
flowchart TD
	 Simulator[MuJoCo Simulator (MJCF XML)] --> Renderer[Offscreen Renderer / Camera]
	 Renderer --> Vision[Vision Pipeline / Dataset Generator]
	 Vision --> PerceptionModel[ShuffleNet-based regressor]
	 Simulator --> Env[Gymnasium Environment: TriplePendulumTeacher]
	 Env --> VecEnv[VecNormalize + SubprocVecEnv]
	 PerceptionModel --> Agent[RL Agent (PPO)]
	 VecEnv --> Agent
	 Agent --> Action[Apply torque to cart]
	 Action --> Simulator
	 Agent --> Checkpoints[Checkpoint & Eval Callbacks]
	 Checkpoints --> Logs[CSV / TensorBoard logs]
```

Architecture components (details):

- Simulator: MuJoCo MJCF XML (`src/triple_pendulum.xml` referenced by code) implements the physical model: cart, three rotational joints (pins), and sensorless rendering camera `agent_cam` used for vision.
- Environment: `TriplePendulumTeacher` (in `src/ppo_agent.py`) is a Gymnasium-compatible wrapper that exposes an observation dict, action space, curriculum resets, reward shaping and termination conditions.
- Perception: `src/dataset_generator.py` creates an HDF5 dataset of rendered grayscale frames, projected 2D pin coordinates and joint angles; `src/vision_model.py` trains a ShuffleNetV2 regressor to map images to 2D coordinates and trig-based angle representations.
- Agent: `src/ppo_agent.py` uses Stable-Baselines3 PPO with vectorized Subproc environments, VecNormalize, and custom callbacks (debug CSV log, evaluation checkpoints).

--------------------------------------------------------------------------------

**4. Environment & Simulation Details**

- Simulator: MuJoCo (mujoco >= 2.3.0; see `requirements.txt`). Model is loaded from an XML string inside `src/ppo_agent.py` or from `triple_pendulum.xml` adjacent to the code.
- Frame-skip: `frame_skip=4` (action applied every 4 simulator steps to reduce control frequency and speed up training).
- Time limit: episodes capped at 1200 steps in wrappers (`TimeLimit`).
- Safety constraints: cart position limit |x| >= 1.2 triggers termination; angular velocity limits (abs(qvel) > 60) also terminate.
- Target configurations: a predefined list of possible target joint angle sets (see `POSSIBLE_TARGETS` in `src/ppo_agent.py`) — training alternates between several upright/downward combinations to encourage generalization.

--------------------------------------------------------------------------------

**5. Observation Space**

Two observation modes are used depending on component:

- Compact state vector (used by RL training): a 15-dimensional vector contained under the `state` key in the gym Dict observation. The components in order are:
  - cart position `x`
  - clipped cart velocity `x_dot`
  - sin(theta1), cos(theta1)
  - sin(theta2), cos(theta2)
  - sin(theta3), cos(theta3)
  - clipped angular velocities theta1_dot, theta2_dot, theta3_dot
  - angular differences to the current target (angle_diff_1..3)
  - previous action (scalar)

- Visual observations (vision pipeline): grayscale rendered frames (original size 448x448 in `dataset_generator.py`) downsampled to 224x224 and repeated into 3-channels for ImageNet normalization before being passed to ShuffleNet.

Why sin/cos? Representing angles as sin and cos avoids discontinuities at ±π and preserves rotational structure, enabling the network to learn continuous functions of angle without wraparound artifacts.

--------------------------------------------------------------------------------

**6. Action Space**

- Continuous one-dimensional action: cart motor torque in range [-1.0, 1.0].
- Actions are clipped and passed through `do_simulation` with the configured `frame_skip`.

Scaling and interpretation: the learned policy outputs a raw action scalar. Stable-Baselines3 ensures compatibility by enforcing the `Box(low=-1.0, high=1.0)` action space; the environment applies the given torque to the cart.

--------------------------------------------------------------------------------

**7. MDP Formalization**

We formalize the task as a continuous-control Markov Decision Process (MDP):

- State space S: continuous vectors representing physical quantities (cart position/velocity, joint orientations and velocities) or images (pixel observations) for the vision pipeline.
- Action space A: continuous scalar torque applied to the cart in [-1, 1].
- Transition P: deterministic/stochastic dynamics defined by MuJoCo physics integrator and random resets introduced by curriculum and target sampling.
- Reward R: shaped reward function (detailed below) combining tracking error penalties, energy terms, progress bonuses and hold rewards.
- Discount γ: implicit in PPO defaults (0.99 by default in SB3 unless modified).

--------------------------------------------------------------------------------

**8. Reward Design — Full Decomposition**

Reward engineering is critical to encourage the two-phase strategy required by swing-up tasks:

1. Inject energy and reduce global tracking error to enter a capture region.
2. Switch to a hold mode that rewards precise stabilization.

### Notation

* Let $\theta = [\theta_1, \theta_2, \theta_3]$ be the joint angles and $\theta^*$ the target configuration.
* Angular error is computed as:

$$
\Delta\theta = \mathrm{wrap}(\theta - \theta^*)
$$

where each component is wrapped into $[-\pi, \pi]$.

* Tracking error:

$$
E_{\text{track}}
================

\sum_{i=1}^{3}
w_i (\Delta\theta_i)^2
$$

where the weights emphasize the upper links (e.g. $w=[0.2, 0.2, 0.6]$).

* Rotational energy proxy:

$$
E_{\text{rot}}
==============

\sum_{i=1}^{3}
\dot{\theta}_i^2
$$

---

### Reward Components

#### 1) Tracking Error

Quadratic penalty encouraging alignment with the current target:

$$
E_{\text{track}}
================

\sum_{i=1}^{3}
w_i (\Delta\theta_i)^2
$$

This term is used both for reward shaping and for detecting capture/stable regions.

---

#### 2) Progress Bonus

Rewards the agent for reducing tracking error relative to the previous timestep:

$$
R_{\text{progress}}
===================

5.0
\exp(-4E_{\text{track}})
\left(
E_{\text{track}}^{\text{prev}}
------------------------------

E_{\text{track}}
\right)
$$

**Purpose:** encourage continuous improvement and prevent dithering near local plateaus.

---

#### 3) Energy Bonus (Swing-Up Incentive)

Provides a small reward for energy injection when far from the target:

$$
R_{\text{energy}}
=================

0.08 \cdot s \cdot
\tanh!\left(
\frac{E_{\text{rot}}}{30}
\right)
$$

where

$$
s
=

\mathrm{clip}
\left(
\frac{E_{\text{track}}}{2.0},
0,
1
\right)
$$

is the swing-phase coefficient.

**Purpose:** encourage active swing-up behavior while gradually fading near the target.

---

#### 4) Cart Penalty

Penalizes excessive cart displacement and velocity:

$$
P_{\text{cart}}
===============

(1-s)
\left(
0.05x^2
+
0.03\dot{x}^2
\right)
$$

The factor $(1-s)$ ensures that cart regulation becomes important primarily after the target region has been reached.

---

#### 5) Hold Bonus

Once the pendulum is captured and stabilized, an additional hold reward is applied:

$$
R_{\text{hold}}
===============

12.0
\cdot
\mathrm{clip}
\left(
\frac{\text{stable_steps}}{80},
0,
1
\right)
$$

**Purpose:** strongly incentivize maintaining the upright configuration instead of repeatedly re-swinging.

---

#### 6) Terminal Penalty

A large negative reward is applied whenever episode constraints are violated:

$$
R_{\text{terminal}}
===================

-25
$$

Examples include:

* cart position outside allowed bounds;
* excessively large angular velocities;
* unstable simulation states.

---

### Phase Logic

#### Swing-Up Phase

If the target has not yet been captured (`capture_counter == 0`):

$$
R
=

e^{-E_{\text{track}}}
+
R_{\text{progress}}
+
R_{\text{energy}}
-----------------

P_{\text{cart}}
$$

The reward prioritizes reducing tracking error while generating sufficient energy for swing-up.

---

#### Stabilization Phase

After capture:

$$
R
=

## 2e^{-E_{\text{track}}}

0.01E_{\text{rot}}
+
R_{\text{hold}}
+
0.5R_{\text{progress}}
$$

The objective shifts from energy generation toward accurate stabilization and long-term balance maintenance.

---

### Stability Criterion

A successful stabilization is declared when

$$
E_{\text{track}} < 0.05
$$

and

$$
\text{stable_steps} > 100
$$

simultaneously hold.

---

### Design Rationale

The reward combines:

* global objective shaping through tracking error;
* local improvement signals via the progress bonus;
* energy-based incentives for swing-up;
* state-dependent cart regulation;
* long-horizon stabilization rewards.

Together these components create an implicit curriculum that smoothly transitions from swing-up to stabilization without requiring an explicit phase detector or external supervisory controller.


**9. Curriculum Learning**

Curriculum is embedded in `reset_model` and `sample_target` and uses four named phases: `bottom`, `easy`, `medium`, `hard/eval`.

- `bottom`: initialize near the hanging (rest) start with small perturbations — helps learn pumping strategies to inject energy.
- `easy`: spawn near target state with small angle/velocity perturbations — assists learning the final stabilization controller.
- `medium`: spawn moderately far from the target (wider angle and velocity ranges).
- `hard` (eval_hard): large angle spreads simulating difficult starts.

Selection probabilities and ranges (from code):

- On random reset, the active target selection favors the newest active target with 70% probability; otherwise a prior active target is sampled.
- Easy phase: angle ranges ≈ [0.10, 0.20, 0.25], velocity scale ≈ 0.03.
- Medium: angle ranges ≈ [0.30, 0.40, 0.50], velocity scale ≈ 0.12.
- Bottom: small angle noise around 0 but higher velocities when configured as a `bottom` restart with perturbations.

Why not start always from bottom? Starting from non-bottom states helps the agent obtain early successes on stabilization and speeds up convergence for the hold-phase policy while still training swing-up strategies via probabilistic sampling.

--------------------------------------------------------------------------------

**10. Training Process & Hyperparameters**

High-level training pipeline (script: `src/ppo_agent.py`):

- Vectorized training: `n_envs = 25` SubprocVecEnv for parallel rollouts.
- Normalization: `VecNormalize` is used. Training env loads precomputed observation statistics from saved vectors when fine-tuning.
- Base algorithm: PPO (Stable-Baselines3) with several tuned hyperparameters.

Key hyperparameters and choices (as configured in code):

- `TOTAL_TIMESTEPS = 15_000_000` for the local `train()` call; pre-trained artifacts in the repository point to `teacher_stage4_430000000_steps.zip` indicating long-run training at ~430M steps for the published model.
- Policy adjustments: learning rate set to 2.5e-5 for fine-tuning; clip_range ≈ 0.10, `ent_coef = 0.005`, `vf_coef = 0.8`, `n_epochs = 5`.
- `policy.log_std` reset to encourage a starting exploration std of 0.20.
- Optimizer defaults are those inside Stable-Baselines3 PPO (Adam-like updates internally managed by SB3).
- Checkpoints: model and VecNormalize snapshots saved periodically via `CheckpointCallback` and evaluation via `EvalCallback`.

Vision training (`src/vision_model.py`):

- Architecture: ShuffleNet V2 backbone (lightweight) with final head projecting to 14 outputs (8 image-projected coordinates + 3 cosines + 3 sines).
- Loss: SmoothL1Loss (Huber-ish) with beta=0.1.
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-2, gradient clipping max_norm=1.0.
- Batch size: 256, epochs: 100.

Reproducibility: ensure seeds, VecNormalize stats, and checkpoint paths are used consistently when evaluating or resuming training.

--------------------------------------------------------------------------------

**11. Hardware & Performance**

Reference hardware used for experiments (project documentation):

- CPU: Intel i7-12700H
- GPU: NVIDIA RTX 4070 (8 GB)
- RAM: 16 GB DDR5 4800 MHz
- CUDA: (set the appropriate CUDA toolkit compatible with the used torch build; see `requirements.txt` and local environment)

Performance observations (approximate):

- Steps-per-second depends strongly on `n_envs`, GPU availability for policy inference and VecNormalize overhead; with `n_envs=25` and RTX4070 expect a few hundred environment steps/sec in parallel on typical setups when using frame-skip=4, but this varies.
- Full training to 430M steps required on the order of multiple days to weeks depending on hardware and parallelism. Fine-tuning (tens of millions of steps) typically takes hours to days.

--------------------------------------------------------------------------------

**12. Results**

Current artifacts in the repository indicate extensive training: saved model `teacher_stage4_430000000_steps.zip` and VecNormalize statistics `teacher_stage4_vecnormalize_430000000_steps.pkl` (paths in `src/ppo_agent.py`). From code comments and checkpoint names we infer the project achieved long-run training in the hundreds of millions of environment steps.

Representative achievements (to document experimentally when adding evaluation metrics):

- Swing-up and stabilization were achieved consistently after extended training (~300M+ steps to reach the initial capability; continued fine-tuning up to ~430M steps further improved robustness). The large training budget was primarily due to the lengthy process of designing and tuning a general reward function capable of producing the desired behaviors. In total, approximately 1.5B training steps were spent on reward engineering experiments, most of which were later discarded. With the current reward function, training a policy for a single target configuration requires roughly 100M steps.
- Success rate (example): >70% across typical targets under evaluation with VecNormalize statistics loaded.

Add explicit figures and TensorBoard dashboards here: training curves, success rate table per target, CSV summaries from `logs/`.

--------------------------------------------------------------------------------

**13. Limitations & Known Issues**

- Memory: dataset generation in `dataset_generator.py` loads and writes very large HDF5 datasets (TOTAL_SAMPLES default 1M) — ensure sufficient disk and memory resources before attempting full dataset creation.
- Sim-to-real: vision-only controllers trained in MuJoCo require domain randomization and camera modeling for sim-to-real transfer; current domain randomization is bounded and primarily lightweight.
- Computational cost: long-run PPO training at the scale suggested by model filenames requires substantial compute resources (time and GPU availability).
- The repo currently references pre-trained artifacts (430M steps). If these files are removed or moved, evaluation scripts will fail — ensure checkpoint paths are present.

--------------------------------------------------------------------------------

**14. Roadmap & Next Steps**

Short-term:

- Add example GIFs/videos to `assets/`.
- Export evaluation metrics (per-target success rates, confusion matrices) and include them in `results/`.

Medium-term:

- Implement stronger domain randomization (textures, lighting, camera intrinsics) to improve sim-to-real transfer.
- Integrate an on-board vision stack for real hardware and build a sim-to-real pipeline.

Long-term:

- Deploy policy to a real triple-pendulum hardware setup.
- Publish a technical report or paper describing reward engineering, curriculum design, and sim-to-real performance.
- Migrate the environment from MuJoCo to Isaac Lab and retrain the agent within the Isaac Lab framework.

--------------------------------------------------------------------------------

**15. Repository Structure**

Top-level tree and purpose of key folders/files:

- `assets/` — media and demonstration files (images, gifs, videos).
- `checkpoints/` — optional directory for lightweight checkpoints.
- `logs/` — CSV debug rollouts and SB3 evaluation logs.
- `scripts/` — helper scripts to launch training or evaluation.
- `src/` — core python modules:
  - `src/ppo_agent.py` — environment, reward, curriculum, and training loop for PPO.
  - `src/dataset_generator.py` — generates HDF5 dataset with rendered frames and ground-truth annotations.
  - `src/vision_model.py` — ShuffleNet-based regressor and training harness for perception.
- `requirements.txt` — Python dependencies.
- `triple_pendulum.xml` (model file adjacent to `src`) — MuJoCo MJCF describing the mechanical system.

--------------------------------------------------------------------------------

**16. Installation**

Minimal reproducible environment steps (Windows / Linux):

1) Clone repository:

```bash
git clone https://github.com/100gramm/triple-pendulum-stabilization.git
cd triple-pendulum-stabilization
```

2) Create Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate     # Windows PowerShell
pip install -U pip
```

3) Install dependencies:

```bash
pip install -r requirements.txt
```

Note: MuJoCo installation requires an appropriate MuJoCo build and license. Follow the official MuJoCo (mujoco-py or mujoco) install instructions for your platform.

4) Verify MuJoCo rendering (optional): run a short script to load `triple_pendulum.xml` and render a single frame using the offscreen renderer.

5) Generate dataset (optional / heavy):

```bash
python src/dataset_generator.py
```

This generates `triple_pendulum_dataset.hdf5` (default TOTAL_SAMPLES is large — adjust `CHUNK_SIZE` and `TOTAL_SAMPLES` in the script for a smaller debug run).

6) Train vision model (optional):

```bash
python src/vision_model.py
```

7) Train / fine-tune PPO agent (recommended to use available checkpoints):

```bash
python src/ppo_agent.py
```

You can edit paths for `BEST_MODEL_PATH` and `BEST_VEC_PATH` inside `src/ppo_agent.py` to load pre-trained artifacts.

--------------------------------------------------------------------------------

**17. Reproducing Experiments**

To reproduce published results exactly, ensure you have:

- matching library versions (see `requirements.txt`)
- same VecNormalize files used during training/evaluation (`models_teacher/*_vecnormalize_*.pkl`)
- identical random seeds and checkpoint loading (use SB3's `load` functions with `env=` argument to reattach normalized environments)

Suggested evaluation workflow:

1) Prepare environment and VecNormalize statistics:

```bash
# create vec env similar to training
python -c "from src.ppo_agent import make_env; from stable_baselines3.common.vec_env import DummyVecEnv; env = make_env()( ); print('env ready')"
```

2) Load model and evaluate deterministic rollouts.

```bash
python - <<'PY'
from stable_baselines3 import PPO
from src.ppo_agent import make_env
env = make_env(random_reset=False)()
model = PPO.load('./models_teacher/teacher_stage4_430000000_steps.zip', env=env, device='cpu')
obs = env.reset()
for _ in range(1000):
	 action, _ = model.predict(obs, deterministic=True)
	 obs, reward, done, info = env.step(action)
	 if done:
		  obs = env.reset()
PY
```

Replace paths and device strings for GPU evaluation.

--------------------------------------------------------------------------------

**18. Credits & Acknowledgements**

This repository synthesizes ideas from classical control, modern RL, and computer vision. Key dependencies include MuJoCo, Stable-Baselines3, PyTorch, OpenCV and Gymnasium.

--------------------------------------------------------------------------------

**19. Contribution & Contact**

Contributions are welcome: file issues for bugs and feature requests, submit PRs for improvements (especially documentation, evaluation scripts, and demo assets). For questions or collaboration proposals, open an issue referencing the topic.

--------------------------------------------------------------------------------