import os
import time
from pathlib import Path

import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict
from gymnasium.wrappers import ClipAction, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# -----------------------------------------------------------------------------
# Configuration, Paths, and Environment Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
XML_PATH = str(BASE_DIR / "triple_pendulum.xml")

# Paths to the pre-trained PPO model and its corresponding environment normalizer
MODEL_PATH = str(
    BASE_DIR / "models_teacher" / "teacher_stage4_430000000_steps"
)
VEC_PATH = str(
    BASE_DIR 
    / "models_teacher" 
    / "teacher_stage4_vecnormalize_430000000_steps.pkl"
)

EPISODE_STEPS = 1000
TARGET = np.array([np.pi, 0.0, 0.0], dtype=np.float32)

# Load the MuJoCo XML physical model into memory once
with open(XML_PATH, "r", encoding="utf-8") as f:
    XML_STRING_DATA = f.read()


# -----------------------------------------------------------------------------
# Environment Definition
# -----------------------------------------------------------------------------
class TriplePendulumTeacher(MujocoEnv):
    """
    Custom Gymnasium environment for a triple inverted pendulum using MuJoCo.
    Designed to evaluate a trained agent attempting to stabilize the links.
    """

    def __init__(self, render_mode="human", random_reset=True):
        self.random_reset = random_reset
        self.current_target = TARGET.copy()
        
        # Temporary low-dimensional space needed for underlying MuJoCo init
        dummy_space = Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        def _custom_initialize_simulation():
            model = mujoco.MjModel.from_xml_string(XML_STRING_DATA, assets={})
            data = mujoco.MjData(model)
            return model, data

        self._initialize_simulation = _custom_initialize_simulation

        super().__init__(
            model_path=os.path.abspath(XML_PATH),
            frame_skip=4,
            render_mode=render_mode,
            observation_space=dummy_space
        )

        # Define the actual dictionary-based observation space (15 dimensions)
        self.observation_space = Dict({
            "state": Box(
                low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
            )
        })
        
        # Action space representing the continuous motor torque bounds [-1, 1]
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.previous_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )

    def _wrap_angle(self, x):
        """Wraps angles to the continuous range [-pi, pi]."""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def _get_obs(self):
        """
        Constructs the current state observation for the agent, containing
        positions, clipped velocities, trigonometry of angles, and errors.
        """
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]
        thetas_dot = self.data.qvel[1:4]
        x_dot = self.data.qvel[0]
        
        angle_diff = thetas - self.current_target
        angle_diff = self._wrap_angle(angle_diff)
        
        obs = np.array([
            x, 
            np.clip(x_dot, -10.0, 10.0),
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

    def _check_done(self):
        """Terminates the episode if safety operational constraints are breached."""
        x = self.data.qpos[0]
        thetas_dot = self.data.qvel[1:4]
        terminated = False

        # Check cart position boundaries
        if abs(x) >= 1.2:
            terminated = True

        # Check extreme joint velocity limits
        if np.any(np.abs(thetas_dot) > 60.0):
            terminated = True

        return terminated, False

    def step(self, action):
        """Applies the action, updates physics, and returns standard gym outputs."""
        action = np.asarray(action, dtype=np.float32)
        self.do_simulation(action, self.frame_skip)
        self.previous_action = action.copy()
        
        obs = self._get_obs()
        terminated, truncated = self._check_done()
        
        return obs, 0.0, terminated, truncated, {}
    
    def reset_model(self):
        """Resets the positions and velocities under curriculum distributions."""
        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.init_qvel)

        if self.random_reset:
            curriculum_phase = self.np_random.random()
            if curriculum_phase < 0.10:
                vel_scale = 0.03
                cart_range = 0.10
            elif curriculum_phase < 0.30:
                vel_scale = 0.08
                cart_range = 0.20
            else:
                vel_scale = 0.12
                cart_range = 0.30
                
            init_thetas = (
                np.array([0, 0, 0]) 
                + self.np_random.uniform(low=-0.30, high=0.30)
            )
            qpos[0] = self.np_random.uniform(-cart_range, cart_range)
            qvel += (
                self.np_random.standard_normal(self.model.nv) 
                * vel_scale
            )
        else:
            init_thetas = (
                np.array([0, 0, 0]) 
                + self.np_random.uniform(low=-0.30, high=0.30)
            )
            qpos[0] = 0.0
            qvel[:] = 0.0

        init_thetas = np.mod(init_thetas + np.pi, 2 * np.pi) - np.pi
        qpos[1:4] = init_thetas
        self.previous_action = np.zeros(
            self.action_space.shape, dtype=np.float32
        )
        self.set_state(qpos, qvel)
        
        return self._get_obs()


# -----------------------------------------------------------------------------
# Environment Wrapper Factory
# -----------------------------------------------------------------------------
def make_env():
    """Builds the operational evaluation pipeline with standard wrappers."""
    env = TriplePendulumTeacher(render_mode="human")
    env = ClipAction(env)
    env = TimeLimit(env, max_episode_steps=EPISODE_STEPS)
    return env


# -----------------------------------------------------------------------------
# Main Execution Loop
# -----------------------------------------------------------------------------
def main():
    # Force GLFW for standard hardware-accelerated 3D rendering windows
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

    episode_idx = 0

    try:
        while True:
            episode_idx += 1
            obs = env.reset()
            base = env.envs[0].unwrapped
            
            # Step delta time computed from engine settings
            sim_dt = base.model.opt.timestep * base.frame_skip
            current_angles = np.rad2deg(base.data.qpos[1:4])

            print("\n-----------------------------------")
            print(f"EPISODE {episode_idx}")
            print("Initial angles (deg):", np.round(current_angles, 2))
            print("-----------------------------------")    

            total_steps = 0
            done = False
            
            while not done:
                step_start = time.perf_counter()
                
                # Predict deterministic action based on current observations
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, _ = env.step(action)
                done = dones[0]
                
                env.render()
                
                # Synchronize loop frequency with simulation time steps
                elapsed = time.perf_counter() - step_start
                remaining = sim_dt - elapsed

                if remaining > 0:
                    time.sleep(remaining)

            final_angles = np.rad2deg(base.data.qpos[1:4])
            print(f"\nEpisode finished after {total_steps} steps")
            print("Final angles (deg):", np.round(final_angles, 2))

    except KeyboardInterrupt:
        print("\nStopped by user.")
        env.close()


if __name__ == "__main__":
    main()