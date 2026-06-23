import csv
import multiprocessing
import os
import time
from collections import Counter, deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
from gymnasium.wrappers import ClipAction, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
XML_PATH = str(BASE_DIR / "triple_pendulum.xml")

# Load the MuJoCo XML model once into memory
with open(XML_PATH, "r", encoding="utf-8") as f:
    XML_STRING_DATA = f.read()

TOTAL_TIMESTEPS = 15_000_000

# Defines the possible target states for the triple pendulum joints (in radians).
# For example, [pi, 0, 0] corresponds to specific upright/downward configurations.
POSSIBLE_TARGETS = [
    np.array([np.pi, 0.0, 0.0], dtype=np.float32),
    np.array([np.pi, 0.0, -np.pi], dtype=np.float32),
    np.array([np.pi, -np.pi, np.pi], dtype=np.float32),
    np.array([0.0, np.pi, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, np.pi], dtype=np.float32),
    np.array([0.0, np.pi, -np.pi], dtype=np.float32),
    np.array([np.pi, -np.pi, 0.0], dtype=np.float32),
]

LOG_DIR = "./logs_teacher"
MODEL_DIR = "./models_teacher"
DEBUG_LOG_FILE = Path(LOG_DIR) / "debug_rollouts.csv"


# -----------------------------------------------------------------------------
# Environment Definition
# -----------------------------------------------------------------------------
class TriplePendulumTeacher(MujocoEnv):
    """
    Custom Gymnasium environment for a triple inverted pendulum using MuJoCo.
    It includes a curriculum learning mechanism to progressively increase the 
    difficulty of the stabilization task.
    """
    
    def __init__(self, xml_path=XML_PATH, render_mode=None, random_reset=True):
        self.random_reset = random_reset
        self.current_target = POSSIBLE_TARGETS[0].copy()
        self.current_target_idx = 0
        
        # We start learning primarily with the first two targets
        self.active_targets = [0, 1]
        absolute_xml_path = os.path.abspath(xml_path)
        
        # A dummy observation space required for initial MuJoCo setup
        dummy_space = Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        # Custom initialization to load the model from the predefined string
        def _custom_initialize_simulation():
            model = mujoco.MjModel.from_xml_string(XML_STRING_DATA, assets={})
            data = mujoco.MjData(model)
            return model, data

        self._initialize_simulation = _custom_initialize_simulation
        
        super().__init__(
            model_path=absolute_xml_path,
            frame_skip=4,
            render_mode=render_mode,
            observation_space=dummy_space
        )
        
        # State dimension: 1 (cart pos) + 1 (cart vel) + 6 (sin/cos of 3 angles) 
        # + 3 (angular vels) + 3 (angle diffs) + 1 (previous action) = 15
        state_dim = 15
        self.observation_space = Dict({
            "state": Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
            )
        })
        
        # Continuous action space for the cart's motor torque [-1.0, 1.0]
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.previous_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )

        # Save base physics parameters for resetting purposes
        self.base_mass = self.model.body_mass.copy()
        self.base_inertia = self.model.body_inertia.copy()
        self.base_friction = self.model.dof_frictionloss.copy()

        # Tracking variables for metrics and rewards
        self.episode_step = 0
        self.max_abs_x = 0.0
        self.max_abs_theta_dot = 0.0
        self.reward_components = {}
        self.last_termination_reason = "none"
        self.curriculum_name = "none"
        self.prev_tracking_error = None
        self.prev_energy = None
        self.stable_steps = 0
        self.capture_counter = 0

    def sample_target(self):
        """
        Selects a target state for the pendulum. Focuses heavily on the 
        newest active target (70% chance) to reinforce recent learning.
        """
        if len(self.active_targets) == 1:
            idx = self.active_targets[0]
        else:
            newest_idx = self.active_targets[-1]
            if self.np_random.random() < 0.70:
                idx = newest_idx
            else:
                old_targets = self.active_targets[:-1]
                idx = old_targets[
                    self.np_random.integers(len(old_targets))
                ]
                
        self.current_target = POSSIBLE_TARGETS[idx].copy()
        self.current_target_idx = idx

    def _get_obs(self):
        """
        Constructs the observation vector sent to the neural network.
        Normalizes angles using sin/cos and calculates the difference 
        between the current angles and the target angles.
        """
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]
        thetas_dot = self.data.qvel[1:4]
        x_dot = self.data.qvel[0]

        # Calculate the shortest angular difference to the target [-pi, pi]
        angle_diff = thetas - self.current_target
        angle_diff = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi

        obs = np.array([
            x, np.clip(x_dot, -10.0, 10.0),
            np.sin(thetas[0]), np.cos(thetas[0]),
            np.sin(thetas[1]), np.cos(thetas[1]),
            np.sin(thetas[2]), np.cos(thetas[2]),
            np.clip(thetas_dot[0], -35.0, 35.0), 
            np.clip(thetas_dot[1], -35.0, 35.0), 
            np.clip(thetas_dot[2], -35.0, 35.0),
            angle_diff[0], angle_diff[1], angle_diff[2],
            self.previous_action[0]
        ], dtype=np.float32)
        
        return {"state": obs}

    def step(self, action):
        """
        Advances the simulation by one step, applies the action, 
        and calculates the resulting state, reward, and termination status.
        """
        self.episode_step += 1
        action = np.asarray(action, dtype=np.float32)
        
        # Apply the action and advance the MuJoCo physics engine
        self.do_simulation(action, self.frame_skip)
        
        x = self.data.qpos[0]
        thetas_dot = self.data.qvel[1:4]
        
        # Track maximum deviations for debugging
        self.max_abs_x = max(self.max_abs_x, abs(float(x)))
        self.max_abs_theta_dot = max(
            self.max_abs_theta_dot, float(np.max(np.abs(thetas_dot)))
        )
        
        terminated, truncated = self._check_done()
        reward = self.compute_reward(action, terminated)
        
        self.previous_action = action.copy()
        obs = self._get_obs()
        
        # Clip reward to prevent gradient explosions during RL training
        reward = np.clip(reward, -40.0, 40.0)
        
        info = {
            "max_abs_x": self.max_abs_x,
            "max_abs_theta_dot": self.max_abs_theta_dot,
            "termination_reason": self.last_termination_reason,
            "episode_step": self.episode_step,
            "curriculum": self.curriculum_name,
            "target_idx": self.current_target_idx,
            "tracking_error": self.reward_components["tracking_error"],
            "reward_total": self.reward_components["reward_total"],
            "stable_steps": self.reward_components["stable_steps"],
            "success": self.reward_components["success"],
        }
        
        return obs, reward, terminated, truncated, info

    def compute_reward(self, action, terminated):
        """
        Calculates the shaped reward signal to guide the agent.
        It heavily rewards minimizing the tracking error and maintaining stability,
        while penalizing excessive cart movement and high kinetic energy.
        """
        thetas = self.data.qpos[1:4]
        thetas_dot = self.data.qvel[1:4]
        x = self.data.qpos[0]
        x_dot = self.data.qvel[0]

        angle_diff = thetas - self.current_target
        angle_diff = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi

        # Weighted importance of the joints (top joint is most critical)
        weights = np.array([0.20, 0.20, 0.60], dtype=np.float32)
        tracking_error = np.sum(weights * angle_diff**2)
        rot_energy = np.sum(thetas_dot**2)
        swing_velocity = np.mean(np.abs(thetas_dot))

        # Reward the agent for reducing the tracking error compared to the last step
        if self.prev_tracking_error is None:
            error_improvement = 0.0
        else:
            error_improvement = (self.prev_tracking_error - tracking_error)

        self.prev_tracking_error = tracking_error
        progress_bonus = (
            5.0 * np.exp(-4.0 * tracking_error) * error_improvement
        )

        # Hysteresis counter for transitioning between "swing-up" and "hold" phases
        if tracking_error < 0.18:
            self.capture_counter = 25
        elif self.capture_counter > 0:
            self.capture_counter -= 1

        stable_region = (tracking_error < 0.08 and swing_velocity < 1.0)

        if stable_region:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        # Phase 1: Swing-up (agent is far from the target)
        if self.capture_counter == 0:
            swing_phase = np.clip(tracking_error / 2.0, 0.0, 1.0)
            energy_bonus = (
                0.08 * swing_phase * np.tanh(rot_energy / 30.0)
            )
            cart_penalty = (
                (1.0 - swing_phase) * (0.05 * x**2 + 0.03 * x_dot**2)
            )
            reward = (
                2.0 * np.exp(-2.5 * tracking_error) 
                + progress_bonus 
                + energy_bonus 
                - cart_penalty
            )
            
        # Phase 2: Hold/Stabilization (agent is near the target)
        else:
            hold_bonus = (
                12.0 * np.clip(self.stable_steps / 80.0, 0.0, 1.0)
            )
            reward = (
                3.0 * np.exp(-10.0 * tracking_error) 
                - 0.03 * rot_energy 
                + hold_bonus 
                + 0.25 * progress_bonus 
                - 0.05 * x**2 
                - 0.05 * x_dot**2
            )
            if tracking_error > 0.40:
                reward -= 0.5

        # Successful stabilization criteria
        success = (tracking_error < 0.05 and self.stable_steps > 100)

        # Severe penalty for failing the constraints
        if terminated:
            reward -= 25.0

        if not np.isfinite(reward):
            reward = -25.0

        self.reward_components = {
            "tracking_error": float(tracking_error),
            "rot_energy": float(rot_energy),
            "capture_counter": int(self.capture_counter),
            "stable_steps": int(self.stable_steps),
            "progress_bonus": float(progress_bonus),
            "reward_total": float(reward),
            "success": int(success)
        }

        return float(reward)

    def _check_done(self):
        """
        Checks if the episode should be terminated prematurely due to 
        safety constraints (cart out of bounds or extreme velocities).
        """
        x = self.data.qpos[0]
        thetas_dot = self.data.qvel[1:4]
        terminated = False
        self.last_termination_reason = "none"
        
        # Terminate if the cart moves too far from the center
        if abs(x) >= 1.2:
            terminated = True
            self.last_termination_reason = "cart_limit"
            
        # Terminate if the angular velocity of any joint is excessively high
        if np.any(np.abs(thetas_dot) > 60.0):
            terminated = True
            self.last_termination_reason = "velocity_limit"
            
        return terminated, False

    def reset_model(self):
        """
        Resets the environment at the start of a new episode.
        Applies curriculum learning by spawning the pendulum at different 
        degrees of difficulty (bottom, easy, medium) depending on the phase.
        """
        self.stable_steps = 0
        self.episode_step = 0
        self.max_abs_x = 0.0
        self.max_abs_theta_dot = 0.0
        self.last_termination_reason = "none"
        self.prev_tracking_error = None
        self.capture_counter = 0
        
        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.init_qvel)
        self.prev_energy = None
        
        if self.random_reset:
            self.sample_target()
            curriculum_phase = self.np_random.random()
            
            # Start from the bottom position
            if self.current_target_idx == 0:
                self.curriculum_name = "bottom"
                init_thetas = (
                    np.array([0.0, 0.0, 0.0]) 
                    + self.np_random.uniform(-0.20, 0.20, size=3)
                )
                qvel[1:4] += self.np_random.uniform(-1.0, 1.0, size=3) * 0.30
                qpos[0] = 0.0
                
            else:
                # Spawn near the target state
                if curriculum_phase < 0.60:
                    self.curriculum_name = "easy"
                    angle_ranges = np.array([0.10, 0.20, 0.25], dtype=np.float32)
                    vel_scale = 0.03
                    cart_range = 0.20
                    
                # Spawn moderately far from the target state
                elif curriculum_phase < 0.70:
                    self.curriculum_name = "medium"
                    angle_ranges = np.array([0.30, 0.40, 0.50], dtype=np.float32)
                    vel_scale = 0.12
                    cart_range = 0.50
                    
                # Spawn at the bottom position but with high initial perturbations
                else:
                    self.curriculum_name = "bottom"
                    init_thetas = (
                        np.array([0.0, 0.0, 0.0]) 
                        + self.np_random.uniform(-0.20, 0.20, size=3)
                    )
                    qvel[1:4] += self.np_random.uniform(-1.0, 1.0, size=3) * 1.0
                    qpos[0] = 0.0

                if self.curriculum_name != "bottom":
                    init_thetas = (
                        self.current_target.copy() 
                        + self.np_random.uniform(
                            low=-angle_ranges, high=angle_ranges
                        )
                    )
                    qvel += (
                        self.np_random.standard_normal(self.model.nv) 
                        * vel_scale
                    )
                    qpos[0] = self.np_random.uniform(-cart_range, cart_range)
                    
        # Hard evaluation mode
        elif getattr(self, "eval_hard", False):
            self.current_target = POSSIBLE_TARGETS[1].copy()
            self.current_target_idx = 1
            self.curriculum_name = "eval_hard"
            angle_ranges = np.array([3.14, 3.14, 3.14], dtype=np.float32)
            init_thetas = (
                self.current_target.copy() 
                + self.np_random.uniform(low=-angle_ranges, high=angle_ranges)
            )
            qpos[0] = self.np_random.uniform(-0.30, 0.30)
            qvel += self.np_random.standard_normal(self.model.nv) * 0.12
            
        # Easy evaluation mode
        else:
            self.current_target = POSSIBLE_TARGETS[1].copy()
            self.current_target_idx = 1
            self.curriculum_name = "eval_easy"
            init_thetas = (
                self.current_target.copy() 
                + self.np_random.uniform(low=-0.20, high=0.20, size=3)
            )
            qpos[0] = self.np_random.uniform(-0.02, 0.02)
            qvel += self.np_random.standard_normal(self.model.nv) * 0.005
            
        init_thetas = np.mod(init_thetas + np.pi, 2 * np.pi) - np.pi
        qpos[1:4] = init_thetas
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        
        # Reset mass, inertia, and friction logic back to baseline
        self.model.body_mass[:] = self.base_mass
        self.model.body_inertia[:] = self.base_inertia
        self.model.dof_frictionloss[:] = self.base_friction
        
        mujoco.mj_setConst(self.model, self.data)
        self.set_state(qpos, qvel)
        
        return self._get_obs()


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
class DebugCallback(BaseCallback):
    """
    A Stable Baselines3 callback designed to monitor and log custom 
    environment metrics (like tracking error, velocities, and terminations) 
    into a CSV file and output summaries to the console during training.
    """
    
    def __init__(self, log_file, verbose=1):
        super().__init__(verbose)
        self.log_file = log_file
        self.last_print = time.time()
        self.reward_window = deque(maxlen=200)
        self.length_window = deque(maxlen=200)
        self.max_x_window = deque(maxlen=200)
        self.max_vel_window = deque(maxlen=200)
        self.term_counter = Counter()
        self.curriculum_counter = Counter()
        self.target_counter = Counter()
        self.target_errors = {}
        self.target_success = Counter()
        self.target_total = Counter()

    def _on_training_start(self):
        # Initialize the CSV logging file with headers
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timesteps", "reward", "length", "termination_reason", 
                    "curriculum", "max_abs_x", "max_abs_theta_dot", 
                    "tracking_error", "tracking_reward", "progress_reward",
                    "reverse_progress_penalty", "hold_reward", "energy_reward", 
                    "capture_reward", "pumping_penalty", "oscillation_penalty", 
                    "cart_velocity_penalty", "total_energy_penalty",
                    "reexcitation_penalty", "post_capture_swing_penalty", 
                    "stability_drop_penalty", "velocity_penalty", 
                    "stabilization_velocity_penalty", "edge_penalty", 
                    "stable_counter", "reward_total"
                ])

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for done, info in zip(dones, infos):
            if not done:
                continue
                
            # Aggregate episode statistics upon termination
            ep_reward = info.get("episode", {}).get("r", 0.0)
            ep_length = info.get("episode", {}).get("l", 0)
            target_idx = info.get("target_idx", -1)
            term_reason = info.get("termination_reason", "none")
            curriculum = info.get("curriculum", "unknown")
            err = info.get("tracking_error", 999.0)

            self.target_counter[target_idx] += 1
            self.reward_window.append(ep_reward)
            self.length_window.append(ep_length)
            self.max_x_window.append(info.get("max_abs_x", 0.0))
            self.max_vel_window.append(info.get("max_abs_theta_dot", 0.0))
            self.term_counter[term_reason] += 1
            self.curriculum_counter[curriculum] += 1
            self.target_total[target_idx] += 1

            if info.get("success", 0):
                self.target_success[target_idx] += 1

            if target_idx not in self.target_errors:
                self.target_errors[target_idx] = deque(maxlen=200)

            self.target_errors[target_idx].append(err)
            
            # Log exact episode metrics to the CSV
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps, ep_reward, ep_length, term_reason, curriculum,
                    info.get("max_abs_x", 0.0), info.get("max_abs_theta_dot", 0.0),
                    info.get("tracking_error", 0.0), info.get("tracking_reward", 0.0),
                    info.get("progress_reward", 0.0), info.get("reverse_progress_penalty", 0.0),
                    info.get("hold_reward", 0.0), info.get("energy_reward", 0.0),
                    info.get("capture_reward", 0.0), info.get("pumping_penalty", 0.0),
                    info.get("oscillation_penalty", 0.0), info.get("cart_velocity_penalty", 0.0),
                    info.get("total_energy_penalty", 0.0), info.get("reexcitation_penalty", 0.0),
                    info.get("post_capture_swing_penalty", 0.0), info.get("stability_drop_penalty", 0.0),
                    info.get("velocity_penalty", 0.0), info.get("stabilization_velocity_penalty", 0.0),
                    info.get("edge_penalty", 0.0), info.get("stable_counter", 0.0),
                    info.get("reward_total", 0.0)
                ])
                
        # Periodically output statistics to standard output (console)
        if time.time() - self.last_print > 20:
            self.last_print = time.time()
            if len(self.reward_window) > 0:
                print("\n========== DEBUG ==========")
                print(f"episodes={len(self.reward_window)}")
                print(f"reward_mean={np.mean(self.reward_window):.2f}")
                print(f"reward_std={np.std(self.reward_window):.2f}")
                print(f"length_mean={np.mean(self.length_window):.2f}")
                print(f"max_abs_x_mean={np.mean(self.max_x_window):.3f}")
                print(f"max_abs_theta_dot_mean={np.mean(self.max_vel_window):.2f}")
                print(f"terminations={dict(self.term_counter)}")
                print(f"curriculum={dict(self.curriculum_counter)}")
                print(f"targets={dict(self.target_counter)}")
                for t_idx, errors in self.target_errors.items():
                    print(f"target {t_idx}: mean_error={np.mean(errors):.3f}")
                for idx in sorted(self.target_total):
                    rate = self.target_success[idx] / max(self.target_total[idx], 1)
                    print(f"target {idx}: success={rate:.2%}")
                print("===========================\n")
                
        return True


# -----------------------------------------------------------------------------
# Training and Execution Setup
# -----------------------------------------------------------------------------
def make_env(random_reset=True, eval_hard=False):
    """
    Utility function to build and wrap the MuJoCo environment for SB3.
    Applies Action Clipping and Time Limits before passing to the Monitor.
    """
    def _init():
        env = TriplePendulumTeacher(random_reset=random_reset)
        env.eval_hard = eval_hard
        env = ClipAction(env)
        env = TimeLimit(env, max_episode_steps=1200)
        
        # Explicitly enforce the specific action space limits for compatibility
        fixed_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        env.action_space = fixed_space
        env.unwrapped.action_space = fixed_space
        
        return Monitor(env)
    return _init

def train():
    """
    Main training loop. Configures vectorized environments, loads previous 
    weights (checkpoint 4), updates learning rates and trains the PPO algorithm.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    n_envs = 25
    
    # Initialize environment setups
    train_env_setup = make_env(random_reset=True)
    eval_easy_setup = make_env(random_reset=False)
    eval_hard_setup = make_env(random_reset=False, eval_hard=True)
    
    # Create parallel vectorized environments for efficient data collection
    env = make_vec_env(
        train_env_setup, n_envs=n_envs, vec_env_cls=SubprocVecEnv
    )
    eval_easy_env = make_vec_env(eval_easy_setup, n_envs=1)
    eval_hard_env = make_vec_env(eval_hard_setup, n_envs=1)
    
    BEST_MODEL_PATH = "./models_teacher/teacher_stage4_430000000_steps.zip"
    BEST_VEC_PATH = "./models_teacher/teacher_stage4_vecnormalize_430000000_steps.pkl"
    
    # Load running averages for observation normalization
    env = VecNormalize.load(BEST_VEC_PATH, env)
    env.training = True
    env.norm_reward = True
    env.clip_obs = 10.0
    
    eval_easy_env = VecNormalize.load(BEST_VEC_PATH, eval_easy_env)
    eval_easy_env.training = False
    eval_easy_env.norm_reward = False
    eval_easy_env.clip_obs = 10.0
    
    eval_hard_env = VecNormalize.load(BEST_VEC_PATH, eval_hard_env)
    eval_hard_env.training = False
    eval_hard_env.norm_reward = False
    eval_hard_env.clip_obs = 10.0
    
    # Synchronize observation statistics for evaluation environments
    eval_easy_env.obs_rms = env.obs_rms
    eval_hard_env.obs_rms = env.obs_rms
    
    # Load the pre-trained Proximal Policy Optimization (PPO) model
    model = PPO.load(BEST_MODEL_PATH, env=env, device="cuda")
    torch.cuda.empty_cache()
    
    # Dynamically update hyperparameters for fine-tuning
    model.lr_schedule = lambda _: 2.5e-5
    model.clip_range = lambda _: 0.10
    model.ent_coef = 0.005
    model.target_kl = 0.010
    model.vf_coef = 0.8
    model.n_epochs = 5
    model.stats_window_size = 100
    
    # Reset policy standard deviation to encourage controlled exploration
    model.policy.log_std.data.fill_(np.log(0.20))
    
    # Set up checkpoints and evaluations
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // n_envs, 1),
        save_path=MODEL_DIR,
        name_prefix="teacher_stage4",
        save_vecnormalize=True
    )
    
    eval_callback_easy = EvalCallback(
        eval_easy_env,
        best_model_save_path=f"{MODEL_DIR}/best_easy",
        log_path=f"{LOG_DIR}/easy",
        eval_freq=40000,
        deterministic=True,
        warn=False
    )
    
    eval_callback_hard = EvalCallback(
        eval_hard_env,
        best_model_save_path=f"{MODEL_DIR}/best_hard",
        log_path=f"{LOG_DIR}/hard",
        eval_freq=40000,
        deterministic=True,
        warn=False
    )
    
    debug_callback = DebugCallback(log_file=str(DEBUG_LOG_FILE), verbose=1)
    
    # Execute the PPO learning algorithm
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        reset_num_timesteps=False,
        callback=[
            checkpoint_callback,
            eval_callback_easy,
            eval_callback_hard,
            debug_callback
        ],
        progress_bar=True
    )
    
    # Final save of the trained weights and normalization statistics
    model.save(f"{MODEL_DIR}/teacher_stage4_ppo")
    env.save(f"{MODEL_DIR}/teacher_stage4_vecnormalize.pkl")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    train()