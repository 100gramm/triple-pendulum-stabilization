import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper, ResizeObservation, GrayScaleObservation, FrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

XML_PATH = 'triple pendulum.xml'
IMG_SIZE = 64
FRAME_STACK = 4
TOTAL_TIMESTAPS = 100_000_000

posible_targets = [
    np.array([0, 0, 0]),
    np.array([np.pi / 2, 0, 0]),
    np.array([0, np.pi / 2, 0]),
    np.array([0, 0, np.pi / 2]),
    np.array([np.pi / 2, np.pi / 2, 0]),
    np.array([np.pi / 2, 0, np.pi / 2]),
    np.array([0, np.pi / 2, np.pi / 2]),
    np.array([np.pi / 2, np.pi / 2, np.pi / 2])
]

class TriplePendulumEnv(gym.envs.mujoco.MujocoEnv):
    def __init__(self, xml_path=XML_PATH):
        super().__init__(xml_path, frame_skip=1)
        data = self.sim.data
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.nq + data.nv), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.target = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
    
    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.flatten(), self.data.qvel.flatten()]).astype(np.float32)

    def step(self, action) -> tuple:
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs, action)
        terminated, truncated = self._chech_done(obs)
        info = {}
        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, obs, action) -> float:
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]

        terminated, truncated = self._chech_done(obs)
        alive_bonus = 2.0 if not (terminated or truncated) else 0.0

        target_diff = np.sum(((self.target - thetas) ** 2) * np.array([0.5, 0.3, 0.2]))
        pos_reward = 5.0 * np.exp(-10.0 * target_diff)

        action_penalty = 0.001 * np.power(action[0], 2)
        bounds_penalty = 0.5 * (x ** 2)

        return float(alive_bonus + pos_reward - action_penalty - bounds_penalty)

    def _chech_done(self, obs):
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]
        
        angle_threshold = np.pi / 3
        angle_diff = np.abs(self.target - thetas)
        
        terminated = bool(np.any(angle_diff > angle_threshold))
        truncated = bool(abs(x) > 2.4)
        
        return terminated, truncated

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.01
        self.set_state(qpos, qvel)
        return self._get_obs()
    

def make_env():
    def _init():
        env = gym.make('TriplePendulumEnv-v0', render_mode="rgb_array")
        env = Monitor(env)
        env = PixelObservationWrapper(env, pixels_only=True)
        env = ResizeObservation(env, shape=(IMG_SIZE, IMG_SIZE))
        env = GrayScaleObservation(env)
        env = FrameStack(env, num_stack=FRAME_STACK)
        return env
    return _init

gym.register(
    id='TriplePendulumEnv-v0',
    entry_point='__main__:TriplePendulumEnv',
    max_episode_steps=500,
)

LOG_DIR  = "./logs/"
MODEL_DIR = "./models/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

checkpointcallback = CheckpointCallback(save_freq=100000, save_path=MODEL_DIR, name_prefix='triple_pendulum')
evalcallback = EvalCallback(make_vec_env(make_env, n_envs=1), best_model_save_path=MODEL_DIR, log_path=LOG_DIR, eval_env=10000)

def train():
    LOG_DIR = "./logs/"
    MODEL_DIR = "./models/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    n_envs = 10 
    env = make_vec_env(make_env(), n_envs=n_envs)

    checkpoint_callback = CheckpointCallback(save_freq=max(100000 // n_envs, 1), save_path=MODEL_DIR, name_prefix='triple_pend')
    
    model = PPO(
        policy='CnnPolicy',
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=2e-4,
        gamma=0.99,
        tensorboard_log=LOG_DIR,
        device='cuda'
    )

    print(f"Training on {model.device}...")
    model.learn(total_timesteps=TOTAL_TIMESTAPS, callback=checkpoint_callback, progress_bar=True)
    model.save(f"{MODEL_DIR}/final_model")

if __name__ == "__main__":
    train()