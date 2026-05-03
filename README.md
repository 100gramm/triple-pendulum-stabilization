# Triple Inverted Pendulum Stabilization via Deep RL and Computer Vision

### Project Overview
This project addresses a high-complexity challenge in control theory: the stabilization of a **triple inverted pendulum** on a mobile cart. Moving beyond traditional sensor-based approaches (direct state vectors), this implementation trains an agent to control the system using only **raw visual data** (pixel input) from the simulation environment.

The triple inverted pendulum is a classic chaotic system. Its high sensitivity to initial conditions and complex internal dynamics make it an ideal benchmark for testing the limits of Deep Reinforcement Learning (DRL) and Computer Vision (CV) integration.

---

### Project Goals
1.  **High-Fidelity Physical Modeling:** Develop an MJCF (XML) configuration for the **MuJoCo** simulator that accurately reflects mass, inertia, and kinematic constraints of a three-link system.
2.  **CV Pipeline Synthesis:** Implement a preprocessing pipeline to extract latent state features from raw simulation frames.
3.  **RL Agent Optimization:** Apply state-of-the-art policy gradient algorithms (**PPO/SAC**) to derive an optimal control law.
4.  **Robustness Analysis:** Achieve stable vertical equilibrium across all three links, even in the presence of external perturbations.

---

### Technical Stack
* **Physics Simulation:** MuJoCo (Multi-Joint dynamics with Contact).
* **Environment Interface:** Gymnasium (standard API for RL environments).
* **RL Framework:** Stable Baselines3 (PyTorch-based implementations).
* **Deep Learning:** PyTorch (CNN architectures for visual processing).
* **Monitoring & Logging:** TensorBoard (visualization of rewards, entropy, and loss).
* **Development Language:** Python 3.10+.

---

### Methodology

#### 1. Observation Space
The observation space consists of a stack of $n$ consecutive grayscale frames. This **Frame Stacking** approach allows the convolutional network to implicitly calculate temporal derivatives (angular velocities and accelerations) from static images.

#### 2. Reward Function
The target objective is formulated as a multi-objective reward function designed to minimize angular deviation while penalizing excessive control effort:

$$R = \sum_{i=1}^{3} w_i \cos(\theta_i) - w_a \|a\|^2 - w_x |x|$$

Where:
* $\theta_i$: Angular deviation of the $i$-th link from the vertical axis.
* $a$: Control effort (acceleration applied to the cart).
* $x$: Displacement of the cart from the origin.

#### 3. Neural Network Architecture
The agent utilizes a Deep Convolutional Neural Network (**CNN**) as a feature extractor, feeding into a Multi-Layer Perceptron (**MLP**) to map visual features to continuous action commands.

---

### Expected Results
* Successful vertical stabilization of a three-link chaotic system.
* Convergence logs (TensorBoard) demonstrating efficient policy optimization.
* A robust vision-based control system capable of operating without direct access to internal simulator state variables.

### It is also planned to create a real installation of the project to transfer the agent to real conditions.