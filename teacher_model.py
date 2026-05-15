import os
import numpy as np
import gymnasium as gym
import multiprocessing
import mujoco

from gymnasium.spaces import Dict, Box
from gymnasium.wrappers import TimeLimit
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env import sync_envs_normalization

XML_PATH = 'triple pendulum.xml'
TOTAL_TIMESTEPS = 50_000_000

possible_targets = [
    np.array([np.pi, np.pi, np.pi]),
    np.array([np.pi, np.pi, 0]),
    np.array([0, np.pi, np.pi]),
    np.array([np.pi, 0, np.pi]),
    np.array([0, 0, np.pi]),
    np.array([0, np.pi, 0]),
    np.array([np.pi, 0, 0]),
]

class TriplePendulumTeacher(MujocoEnv):
    def __init__(self, xml_path=XML_PATH, targets=possible_targets, render_mode=None):
        self.all_targets = targets
        self.active_targets_count = 1
        self.current_target = self.all_targets[0].copy()
        
        absolute_xml_path = os.path.abspath(xml_path)
        with open(absolute_xml_path, "r", encoding="utf-8") as f:
            xml_string = f.read()

        dummy_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        def _custom_initialize_simulation():
            model = mujoco.MjModel.from_xml_string(xml_string)
            data = mujoco.MjData(model)
            return model, data
        
        self._initialize_simulation = _custom_initialize_simulation

        super().__init__(
            model_path=absolute_xml_path, 
            frame_skip=4, 
            render_mode=render_mode, 
            observation_space=dummy_space
        )

        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        
        self.k_reward = 5.0
        self.dr_factor = 0.0
        self.transition_mode = False
        
        self.base_mass = self.model.body_mass.copy()
        self.base_inertia = self.model.body_inertia.copy()
        self.base_friction = self.model.dof_frictionloss.copy()

        state_dim = self.model.nq + self.model.nv

        self.observation_space = Dict({
            'state': Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
            'target': Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        })
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.steps_in_episode = 0
        self.target_change_step = 0
    
    def _get_obs(self) -> dict:
        state = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        return {"state": state, "target": self.current_target.astype(np.float32)}

    def step(self, action) -> tuple:
        self.do_simulation(action, self.frame_skip)
        self.steps_in_episode += 1
        
        if self.transition_mode and self.steps_in_episode == self.target_change_step:
            if self.active_targets_count > 1:
                old_target = self.current_target.copy()
                valid_indices = [i for i in range(self.active_targets_count) if not np.array_equal(self.all_targets[i], old_target)]
                if valid_indices:
                    new_idx = self.np_random.choice(valid_indices)
                    self.current_target = self.all_targets[new_idx].copy()
        
        terminated, truncated = self._check_done()
        obs = self._get_obs()
        reward = self.compute_reward(action, terminated, truncated)

        info = {}
        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, action, terminated: bool, truncated: bool) -> float:
        action_rate_penalty = 0.1 * np.sum(np.square(action - self.previous_action))
        self.previous_action = action.copy()
        
        weights = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]

        alive_bonus = 0.01 if not (terminated or truncated) else 0.0

        main_diff = np.sum(((self.current_target - thetas) ** 2) * weights)
        pos_reward = 1 * np.exp(-self.k_reward * main_diff)

        precision_bonus = 0.2 if main_diff < 0.05 else 0.0
        fall_penalty = -2.0 if terminated else 0.0
        action_penalty = 0.01 * np.square(action[0])
        
        boundary_penalty = 0.0
        boundary_dist = 2.4 - abs(x)
        if boundary_dist < 0.5:
            boundary_penalty = 0.5 * np.exp(-5.0 * boundary_dist)

        reward = (alive_bonus + pos_reward + precision_bonus - action_penalty - 
                  boundary_penalty + fall_penalty - action_rate_penalty)
        
        if not np.isfinite(reward):
            print(f"Критическая ошибка: Reward NaN! main_diff: {main_diff}")
            reward = 0.0
        return float(reward)

    def _check_done(self) -> tuple:
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]
        
        angle_threshold = np.pi / 2 
        angle_fail = bool(np.any(np.abs(thetas) > angle_threshold))
        out_of_bounds = bool(abs(x) > 2.4)
        
        terminated = angle_fail or out_of_bounds
        truncated = False
        
        return terminated, truncated

    def reset_model(self):
        self.steps_in_episode = 0
        self.target_change_step = int(self.np_random.integers(200, 301))
        
        if self.active_targets_count == 1:
            p = [1.0]
        else:
            p_old = 0.5 / (self.active_targets_count - 1)
            p = [p_old] * (self.active_targets_count - 1) + [0.5]
        
        idx = self.np_random.choice(self.active_targets_count, p=p)
        self.current_target = self.all_targets[idx].copy()
        noise_level = 0.05 + (0.1 * self.dr_factor)

        if self.transition_mode:
            init_thetas = self.current_target + self.np_random.uniform(-noise_level, noise_level, size=3)
        else:
            if self.np_random.random() < 0.3:
                init_thetas = self.init_qpos[1:4] + self.np_random.uniform(-noise_level, noise_level, size=3)
            else:
                init_thetas = self.current_target + self.np_random.uniform(-noise_level, noise_level, size=3)
        
        init_thetas = np.clip(init_thetas, -np.pi, np.pi)

        qpos = self.init_qpos.copy()
        qpos[1:4] = init_thetas
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.01

        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)

        has_mass = self.base_mass > 1e-6
        mass_noise = self.np_random.normal(0, 0.05 * self.dr_factor, size=self.base_mass.shape)
        
        new_mass = self.base_mass.copy()
        new_mass[has_mass] = np.maximum(1e-4, self.base_mass[has_mass] + mass_noise[has_mass])
        
        mass_ratio = np.ones_like(self.base_mass)
        mass_ratio[has_mass] = new_mass[has_mass] / self.base_mass[has_mass]
        
        self.model.body_mass[:] = new_mass
        self.model.body_inertia[:] = self.base_inertia * mass_ratio[:, np.newaxis]

        friction_noise = self.np_random.normal(0, 0.1 * self.dr_factor, size=self.base_friction.shape)
        self.model.dof_frictionloss[:] = np.maximum(0, self.base_friction + friction_noise)

        mujoco.mj_setConst(self.model, self.data)

        self.set_state(qpos, qvel)
        return self._get_obs()

    def set_curriculum_params(self, targets_count, k_reward, dr_factor, transition_mode):
        self.active_targets_count = targets_count
        self.k_reward = k_reward
        self.dr_factor = dr_factor
        self.transition_mode = transition_mode


class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env=None, verbose=0, max_reward=150):
        super().__init__(verbose)
        self.targets_count = 1
        self.k_reward = 5.0
        self.dr_factor = 0.0
        self.transition_mode = False
        self.eval_env = eval_env
        self.max_reward = max_reward
    
    def _on_training_start(self) -> None:
        self._update_envs()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.eval_env is not None:
            sync_envs_normalization(self.training_env, self.eval_env)

        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])

            if mean_reward > self.max_reward:
                if self.targets_count < 8:
                    self.k_reward = min(15.0, self.k_reward * 1.1)
                    self.dr_factor = min(1.0, self.dr_factor + 0.1)
                    self.targets_count += 1
                    self.max_reward += 5
                else:
                    self.k_reward = min(15.0, self.k_reward * 1.05)
                    self.transition_mode = True
                    self.dr_factor = min(1.0, self.dr_factor + 0.1)
                    self.max_reward += 10

                self._update_envs()

                if self.verbose > 0:
                    mode = "TRANSITIONS" if self.transition_mode else "STATIC"
                    print(f"\n[Curriculum] Level Up! Mode: {mode}, Targets: {self.targets_count}, "
                          f"k: {self.k_reward:.2f}, New Max Reward Target: {self.max_reward}")

    def _update_envs(self) -> None:
        self.training_env.env_method(
            "set_curriculum_params", 
            self.targets_count, 
            self.k_reward, 
            self.dr_factor, 
            self.transition_mode
        )
        if self.eval_env is not None:
            self.eval_env.env_method(
                "set_curriculum_params", 
                self.targets_count, 
                self.k_reward, 
                self.dr_factor, 
                self.transition_mode
            )


def make_env():
    def _init():
        env = TriplePendulumTeacher()
        env = TimeLimit(env, max_episode_steps=500)
        return Monitor(env)
    return _init

LOG_DIR  = "./logs_teacher/"
MODEL_DIR = "./models_teacher/"

def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    n_envs = 10
    env_setup = make_env()
    env = make_vec_env(env_setup, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200000 // n_envs, 1), 
        save_path=MODEL_DIR, 
        name_prefix='teacher_pend', 
        save_vecnormalize=True
    )
    
    eval_env = make_vec_env(env_setup, n_envs=1)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    curriculum_callback = CurriculumCallback(
        eval_env=eval_env, verbose=1, max_reward=150
    )
    
    eval_freq = 4096
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=MODEL_DIR, 
        log_path=LOG_DIR, 
        eval_freq=eval_freq,
        warn=False
    )

    policy_kwargs = dict(ortho_init=True, net_arch=dict(pi=[256, 256], vf=[256, 256]))
    
    def linear_schedule(initial_value: float):
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    learning_rate = linear_schedule(3e-4)
    
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=4096,
        batch_size=1024,
        n_epochs=10,
        learning_rate=learning_rate,
        tensorboard_log=LOG_DIR,
        ent_coef=0.015,
        target_kl=0.015,
        max_grad_norm=0.3,
        device='cuda'
    )

    print(f"Training Teacher on {model.device}...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=[checkpoint_callback, curriculum_callback, eval_callback],
        progress_bar=True
    )
    model.save(f"{MODEL_DIR}/teacher_final_model")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train()