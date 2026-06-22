import os
import time
import numpy as np
import mujoco

from pathlib import Path
from gymnasium.spaces import Dict, Box
from gymnasium.wrappers import TimeLimit, ClipAction
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

BASE_DIR = Path(__file__).resolve().parent
XML_PATH = str(
    BASE_DIR / "triple_pendulum.xml")

MODEL_PATH = (
    BASE_DIR
    / "models_teacher"
    / "teacher_stage4_415000000_steps")

VEC_PATH = (
    BASE_DIR
    / "models_teacher"
    / "teacher_stage4_vecnormalize_415000000_steps.pkl")

RENDER_FPS = 60
EPISODE_STEPS = 1000
TARGET = np.array([np.pi,  0.0,    0.0], dtype=np.float32)

with open(XML_PATH, "r", encoding="utf-8") as f:
    XML_STRING_DATA = f.read()

class TriplePendulumTeacher(MujocoEnv):
    def __init__(self, render_mode="human", random_reset=True):
        self.random_reset = random_reset
        self.current_target = TARGET.copy()
        dummy_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32)

        def _custom_initialize_simulation():
            model = mujoco.MjModel.from_xml_string(XML_STRING_DATA, assets={})
            data = mujoco.MjData(model)
            return model, data

        self._initialize_simulation = (_custom_initialize_simulation)

        super().__init__(
        model_path=os.path.abspath(XML_PATH), frame_skip=4, 
        render_mode=render_mode, observation_space=dummy_space)

        self.observation_space = Dict({"state": Box(low=-np.inf, 
            high=np.inf, shape=(15,), dtype=np.float32)})
        
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _wrap_angle(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _get_obs(self):
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]
        thetas_dot = self.data.qvel[1:4]
        x_dot = self.data.qvel[0]
        angle_diff = (thetas - self.current_target)
        angle_diff = self._wrap_angle(angle_diff)
        obs = np.array([x, np.clip(x_dot, -10.0, 10.0),
            np.sin(thetas[0]), np.cos(thetas[0]), np.sin(thetas[1]), np.cos(thetas[1]),
            np.sin(thetas[2]), np.cos(thetas[2]), np.clip(thetas_dot[0], -35.0, 35.0),
            np.clip(thetas_dot[1], -35.0, 35.0), np.clip(thetas_dot[2], -35.0, 35.0), 
            angle_diff[0], angle_diff[1], angle_diff[2],  self.previous_action[0]], dtype=np.float32
            )

        return {"state": obs}

    def _check_done(self):
        x = self.data.qpos[0]
        thetas_dot = self.data.qvel[1:4]
        terminated = False

        if abs(x) >= 1.2:
            terminated = True

        if np.any(np.abs(thetas_dot) > 60.0):
            terminated = True

        truncated = False
        return terminated, truncated

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.do_simulation(action, self.frame_skip)
        self.previous_action = (action.copy())
        obs = self._get_obs()
        terminated, truncated = (self._check_done())
        return (obs, 0.0, terminated, truncated, {})
    
    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.init_qvel)

        if self.random_reset:
            curriculum_phase = self.np_random.random()
            if curriculum_phase < 0.10:
                angle_ranges = np.array([0.10, 0.15, 0.25], dtype=np.float32)
                vel_scale = 0.03
                cart_range = 0.10
            elif curriculum_phase < 0.30:
                angle_ranges = np.array([0.50, 0.70, 0.90], dtype=np.float32)
                vel_scale = 0.08
                cart_range = 0.20
            else:
                angle_ranges = np.array([3.14, 3.14, 3.14], dtype=np.float32)
                vel_scale = 0.12
                cart_range = 0.30
            init_thetas = np.array([0,0,0]) + self.np_random.uniform(low=-0.30, high=0.30)
            qpos[0] = self.np_random.uniform(-cart_range,  cart_range)
            qvel += (self.np_random.standard_normal(self.model.nv) * vel_scale)
        else:
            init_thetas = (np.array([0,0,0]) + self.np_random.uniform(low=-0.30, high=0.30))
            qpos[0] = 0.0
            qvel[:] = 0.0

        init_thetas = np.mod(init_thetas + np.pi, 2 * np.pi) - np.pi
        qpos[1:4] = init_thetas
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.set_state(qpos, qvel)
        return self._get_obs()

def make_env():
    env = TriplePendulumTeacher(render_mode="human")
    env = ClipAction(env)
    env = TimeLimit(env, max_episode_steps=EPISODE_STEPS)
    return env

def main():
    os.environ["MUJOCO_GL"] = "glfw"

    print("\n===================================")
    print("TRIPLE PENDULUM VISUAL TEST")
    print("===================================")
    print("\nLoading environment...")

    base_env = make_env()
    venv = DummyVecEnv([lambda: base_env])

    print("Loading VecNormalize...")

    env = VecNormalize.load(str(VEC_PATH), venv)
    env.training = False
    env.norm_reward = False
    print("Loading PPO model...")
    model = PPO.load(str(MODEL_PATH), env=env, device="cuda")

    print("\n===================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("===================================")

    frame_time = 1.0 / RENDER_FPS
    episode_idx = 0

    try:
        while True:
            episode_idx += 1
            obs = env.reset()
            base = env.envs[0].unwrapped
            sim_dt = (base.model.opt.timestep * base.frame_skip)
            current_angles = np.rad2deg(base.data.qpos[1:4])

            print("\n-----------------------------------")
            print(f"EPISODE {episode_idx}")
            print("Initial angles (deg):", np.round(current_angles, 2))
            print("-----------------------------------")    

            total_steps = 0
            done = False
            step_start = time.time()
            while not done:
                step_start = time.perf_counter()
                action, _ = model.predict(obs,deterministic=True)
                obs, reward, dones, infos = env.step(action)
                done = dones[0]
                env.render()
                elapsed = (time.perf_counter() - step_start)
                remaining = sim_dt - elapsed

                if remaining > 0:
                    time.sleep(remaining)

            final_angles = np.rad2deg(base.data.qpos[1:4])
            print(f"\nEpisode finished after {total_steps} steps")
            print("Final angles (deg):", np.round(final_angles,2))

    except KeyboardInterrupt:
        print("\nStopped by user.")
        env.close()

if __name__ == "__main__":
    main()