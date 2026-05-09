import os
import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box
from gymnasium.wrappers import PixelObservationWrapper, ResizeObservation, FrameStackObservation, TransformObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

XML_PATH = 'triple pendulum.xml'
IMG_SIZE = 64
FRAME_STACK = 4
TOTAL_TIMESTEPS = 100_000_000

posible_targets = [
    np.array([0, 0, 0]),
    np.array([0, 0, 1]),
    np.array([0, 1, 0]),
    np.array([0, 1, 1]),
    np.array([1, 0, 0]),
    np.array([1, 0, 1]),
    np.array([1, 1, 0]),
    np.array([1, 1, 1])
]

class TriplePendulum(gym.envs.mujoco.MujocoEnv):
    def __init__(self, xml_path=XML_PATH, posible_targets=posible_targets, k_reward=5.0, render_mode='rgb_array'):
        super().__init__(xml_path, frame_skip=1, render_mode=render_mode)
        data = self.data

        self.all_targets = posible_targets
        self.active_targets = [self.all_targets[0]] 
        self.current_target = self.active_targets[0]
        self.previous_action = np.zeros(self.action_space.shape)
        self.dr_factor = 0.0

        self.base_mass = self.model.body_mass.copy()
        self.base_friction = self.model.dof_frictionloss.copy()

        self.observation_space = Dict({
            'state': Box(low=-np.inf, high=np.inf, shape=(7 + data.nv,), dtype=np.float32),
            'target': Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        })
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.k_reward = k_reward
    
    def _get_obs(self) -> np.ndarray:
        thetas = self.data.qpos[1:4]
        sin_thetas = np.sin(thetas)
        cos_thetas = np.cos(thetas)
        state = np.concatenate([
            [self.data.qpos[0]],
            sin_thetas,
            cos_thetas,
            self.data.qvel.flatten()
        ]).astype(np.float32)
        return {"state": state, "target": self.current_target.astype(np.float32)}

    def step(self, action) -> tuple:
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs, action)
        terminated, truncated = self._check_done(obs)
        info = {}
        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, obs, action) -> float:
        action_rate_penalty = 0.1 * np.sum(np.square(action - self.previous_action))
        self.previous_action = action.copy()
        weights = np.array([0.5, 0.3, 0.2])
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]

        terminated, truncated = self._check_done(obs)
        alive_bonus = 0.1 if not (terminated or truncated) else 0.0

        main_diff = np.sum(((self.current_target - thetas) ** 2) * weights)
        pos_reward = 0.1 * np.exp(-self.k_reward * main_diff)

        precision_bonus = 0.2 if main_diff < 0.05 else 0.0

        all_cross_rewards = []
        for target in self.active_targets:
            if np.array_equal(target, self.current_target):
                continue
            diff = np.sum(weights * ((target - thetas) ** 2))
            res = 0.02 * np.exp(-self.k_reward * diff)
            all_cross_rewards.append(res)
        max_cross_reward = max(all_cross_rewards) if all_cross_rewards else 0.0

        fall_penalty = -2.0 if terminated else 0.0
        action_penalty = 0.01 * np.square(action[0])
        
        boundary_penalty = 0.0
        boundary_dist = 2.4 - abs(x)
        if boundary_dist < 0.5:
            boundary_penalty = 0.5 * np.exp(-5 * boundary_dist)

        reward = (alive_bonus + pos_reward + precision_bonus + max_cross_reward - action_penalty - boundary_penalty + fall_penalty - action_rate_penalty)
        return float(reward)

    def _check_done(self, obs):
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]
        
        angle_threshold = np.pi / 3
        angle_diff = np.abs(self.current_target - thetas)
        
        terminated = bool(np.any(angle_diff > angle_threshold))
        truncated = bool(abs(x) > 2.4)
        
        return terminated, truncated

    def reset_model(self):
        n_active = len(self.active_targets)
        if n_active == 1:
            p = [1.0]
        else:
            p_old = 0.5 / (n_active - 1)
            p = [p_old] * (n_active - 1) + [0.5]
        
        idx = self.np_random.choice(len(self.active_targets), p=p)
        self.current_target = self.active_targets[idx]

        if self.np_random.random() < 0.5:
            init_thetas = self.init_qpos[1:4] + self.np_random.uniform(-0.01, 0.01, size=3)
        else:
            init_thetas = self.current_target + self.np_random.uniform(-0.05, 0.05, size=3)
        
        qpos = self.init_qpos.copy()
        qpos[1:4] = init_thetas
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.01

        self.previous_action = np.zeros(self.action_space.shape)

        mass_noise = self.np_random.normal(0, 0.05 * self.dr_factor, size=self.base_mass.shape)
        self.model.body_mass[:] = self.base_mass + mass_noise
        self.model.dof_frictionloss[:] = np.maximum(0, self.base_friction + self.np_random.normal(0, 0.1, size=self.base_friction.shape))

        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def unlock_next_target(self):
        if len(self.active_targets) < len(self.all_targets):
            new_target = self.all_targets[len(self.active_targets)]
            self.active_targets.append(new_target)
            return True
        return False

    def update_k(self, new_k):
        self.k_reward = new_k
    
    def update_dr(self, new_dr):
        self.dr_factor = new_dr

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0, max_reward=80):
        super().__init__(verbose)
        self.max_reward = max_reward
        self.k_reward = 5.0
        self.dr_factor = 0.0

    def _on_step(self) -> bool:
        if 'rollout/ep_rew_mean' in self.model.logger.name_to_value:
            mean_reward = self.model.logger.name_to_value['rollout/ep_rew_mean']
            
            if mean_reward > self.max_reward:
                self.k_reward *= 2
                self.max_reward += 5
                self.dr_factor = min(1.0, self.dr_factor + 0.15)
                
                self.training_env.env_method('unlock_next_target')
                self.training_env.env_method('update_k', new_k=self.k_reward)
                self.training_env.env_method('update_dr', new_dr=self.dr_factor)

                if self.verbose > 0:
                    print(f"Curriculum Level Up! New k: {self.k_reward:.2f}, DR: {self.dr_factor:.2f}, Target: {self.max_reward}")
                
        return True

class VisionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space['pixels'].shape
        self.observation_space['pixels'] = Box(low=0, high=255, shape=(IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    def observation(self, obs):
        img = cv2.cvtColor(obs['pixels'], cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        obs['pixels'] = img
        return obs

def fix_stack_obs(obs):
    if isinstance(obs['target'], np.ndarray) and obs['target'].ndim == 2:
        obs['target'] = obs['target'][-1]
    return obs

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env():
    def _init():
        env = gym.make('TriplePendulumEnv-v0', render_mode="rgb_array")
        env = Monitor(env)
        env = PixelObservationWrapper(env, observation_key='pixels')
        env = VisionWrapper(env)
        env = FrameStackObservation(env, stack_size=FRAME_STACK)
        env = TransformObservation(env, fix_stack_obs)
        return env
    return _init

gym.register(
    id='TriplePendulumEnv-v0',
    entry_point='__main__:TriplePendulum',
    max_episode_steps=500,
)

LOG_DIR  = "./logs/"
MODEL_DIR = "./models/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    n_envs = 10 
    env = make_vec_env(make_env(), n_envs=n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    checkpoint_callback = CheckpointCallback(save_freq=100000 // n_envs, save_path=MODEL_DIR, name_prefix='triple_pend')
    curriculum_callback = CurriculumCallback(verbose=1, max_reward=50)
    eval_env = make_vec_env(make_env(), n_envs=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR, eval_freq=40960)

    policy_kwargs = dict(ortho_init=True)
    learning_rate=linear_schedule(3e-4)
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        n_steps=8192,
        batch_size=1024,
        n_epochs=10,
        learning_rate=learning_rate,
        tensorboard_log=LOG_DIR,
        ent_coef=0.01,
        device='cuda'
    )

    print(f"Training on {model.device}...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=[checkpoint_callback, eval_callback, curriculum_callback],
        progress_bar=True
    )
    model.save(f"{MODEL_DIR}/final_model")

if __name__ == "__main__":
    train()