import os
import threading
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image, ImageTk
from gymnasium.spaces import Box, Dict
from gymnasium.wrappers import ClipAction, TimeLimit
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import tkinter as tk

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
XML_PATH = BASE_DIR / "triple_pendulum.xml"
MODEL_PATH = BASE_DIR / "models_teacher" / "teacher_stage4_430000000_steps"
VEC_PATH = (
    BASE_DIR / "models_teacher" / "teacher_stage4_vecnormalize_430000000_steps.pkl"
)

EPISODE_STEPS = 1200
RENDER_FPS = 50.0

POSSIBLE_TARGETS = [
    np.array([np.pi, 0.0, 0.0], dtype=np.float32),
    np.array([np.pi, 0.0, -np.pi], dtype=np.float32),
    np.array([np.pi, -np.pi, np.pi], dtype=np.float32),
    np.array([0.0, np.pi, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, np.pi], dtype=np.float32),
    np.array([0.0, np.pi, -np.pi], dtype=np.float32),
    np.array([np.pi, -np.pi, 0.0], dtype=np.float32),
]

TARGET_LABELS = [
    "π · 0 · 0",
    "π · 0 · -π",
    "π · -π · π",
    "0 · π · 0",
    "0 · 0 · π",
    "0 · π · -π",
    "π · -π · 0",
]

with open(XML_PATH, "r", encoding="utf-8") as f:
    XML_STRING_DATA = f.read()


class TriplePendulumTeacher(MujocoEnv):
    """
    Custom MuJoCo environment for the triple pendulum control task.
    Handles rendering, state observation, and step logic.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, render_mode="rgb_array"):
        self.current_target = POSSIBLE_TARGETS[0].copy()
        dummy_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # Helper function to load the MuJoCo XML model
        def _custom_initialize_simulation():
            model = mujoco.MjModel.from_xml_string(XML_STRING_DATA, assets={})
            data = mujoco.MjData(model)
            return model, data

        self._initialize_simulation = _custom_initialize_simulation

        # Initialize the parent Gymnasium/MuJoCo environment
        super().__init__(
            model_path=str(XML_PATH),
            frame_skip=4,
            render_mode=render_mode,
            observation_space=dummy_space,
            camera_name="agent_cam",
            width=640,
            height=480,
        )

        # Define environment spaces for the RL agent
        self.observation_space = Dict(
            {"state": Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)}
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.previous_action = np.zeros(1, dtype=np.float32)

    def set_target(self, idx: int):
        """Update the target angle for the pendulum system."""
        self.current_target = POSSIBLE_TARGETS[idx].copy()

    def _get_obs(self):
        """
        Constructs and returns the observation vector.
        Combines position, velocity, and sine/cosine of angles to help the model learn.
        """
        x = self.data.qpos[0]
        thetas = self.data.qpos[1:4]
        t_dot = self.data.qvel[1:4]
        x_dot = self.data.qvel[0]

        # Calculate angular differences for the state representation
        angle_diff = thetas - self.current_target
        angle_diff = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi

        # Create the observation array with clipped values to keep inputs bounded
        obs = np.array(
            [
                x,
                np.clip(x_dot, -10.0, 10.0),
                np.sin(thetas[0]),
                np.cos(thetas[0]),
                np.sin(thetas[1]),
                np.cos(thetas[1]),
                np.sin(thetas[2]),
                np.cos(thetas[2]),
                np.clip(t_dot[0], -35.0, 35.0),
                np.clip(t_dot[1], -35.0, 35.0),
                np.clip(t_dot[2], -35.0, 35.0),
                angle_diff[0],
                angle_diff[1],
                angle_diff[2],
                self.previous_action[0],
            ],
            dtype=np.float32,
        )
        return {"state": obs}

    def _check_done(self):
        """
        Checks if the simulation should stop (e.g., cart moved too far,
        or velocities went out of safe limits).
        """
        terminated = (
            abs(self.data.qpos[0]) >= 1.2 or np.any(np.abs(self.data.qvel[1:4]) > 60.0)
        )
        return bool(terminated), False

    def step(self, action):
        """
        Performs a single simulation step. Applies action, updates physics,
        then returns new observation and termination status.
        """
        action = np.asarray(action, dtype=np.float32)
        self.do_simulation(action, self.frame_skip)
        self.previous_action = action.copy()
        obs = self._get_obs()
        terminated, truncated = self._check_done()
        return obs, 0.0, terminated, truncated, {}

    def reset_model(self):
        """Resets the simulation to the initial standing-still state."""
        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.init_qvel)

        qpos[1:4] = 0.0
        qpos[0] = 0.0

        self.previous_action[:] = 0.0
        mujoco.mj_setConst(self.model, self.data)
        self.set_state(qpos, qvel)
        return self._get_obs()


class SimThread(threading.Thread):
    """
    Background thread for running the simulation loop without blocking the GUI.
    """

    def __init__(self, frame_cb, telemetry_cb):
        super().__init__(daemon=True)
        self.frame_cb = frame_cb  # Callback function for rendering frames
        self.telemetry_cb = telemetry_cb  # Callback for status updates
        self._target_idx = 0
        self._pending_target = None
        self._pending_reset = True
        self._paused = True
        self._lock = threading.Lock()  # Lock to ensure thread-safe variable access
        self._running = True

    def set_target(self, idx):
        """Updates the target index in a thread-safe way."""
        with self._lock:
            self._pending_target = idx

    def toggle_pause(self):
        """Toggles simulation pause state."""
        with self._lock:
            self._paused = not self._paused
        return self._paused

    def force_reset(self):
        """Sets the reset flag so the simulation restarts on next iteration."""
        with self._lock:
            self._pending_reset = True

    def stop(self):
        """Signals the loop to stop."""
        self._running = False

    def _send_telemetry(self, base_env, ep, steps):
        """Callback to update GUI telemetry data."""
        angles_deg = np.rad2deg(base_env.data.qpos[1:4])
        cart_x = base_env.data.qpos[0]
        self.telemetry_cb(
            {"episode": ep, "step": steps, "cart_x": cart_x, "angles": angles_deg}
        )

    def run(self):
        """Main simulation execution loop."""
        # Initialize environment and load trained PPO model
        inner = TriplePendulumTeacher(render_mode="rgb_array")
        inner = ClipAction(inner)
        inner = TimeLimit(inner, max_episode_steps=EPISODE_STEPS)
        venv = DummyVecEnv([lambda: inner])

        env = VecNormalize.load(str(VEC_PATH), venv)
        env.training = False
        env.norm_reward = False

        model = PPO.load(str(MODEL_PATH), env=env, device="cpu")
        base = env.envs[0].unwrapped
        sim_dt = base.model.opt.timestep * base.frame_skip

        ep = 1
        steps = 0
        obs = env.reset()
        done = False

        render_interval = 1.0 / RENDER_FPS
        last_render_time = time.perf_counter()

        # Initial render frame
        frame = base.render()
        if frame is not None:
            self.frame_cb(frame)
        self._send_telemetry(base, ep, steps)

        # Primary simulation loop
        while self._running:
            t0 = time.perf_counter()

            # Safely check for UI requests (Reset/Pause/Target change)
            with self._lock:
                do_reset = self._pending_reset
                self._pending_reset = False
                new_target = self._pending_target
                self._pending_target = None
                is_paused = self._paused

            # If user selected a new target in GUI, update environment
            if new_target is not None and new_target != self._target_idx:
                self._target_idx = new_target
                base.set_target(self._target_idx)

            # Handle Reset Logic
            if do_reset:
                obs = env.reset()
                done = False
                steps = 0
                frame = base.render()
                if frame is not None:
                    self.frame_cb(frame)
                last_render_time = time.perf_counter()
                self._send_telemetry(base, ep, steps)
                if is_paused:
                    time.sleep(0.01)
                    continue

            # If paused, wait and check again
            if is_paused:
                time.sleep(0.01)
                continue

            # Reset environment automatically if current episode finishes
            if done:
                ep += 1
                obs = env.reset()
                done = False
                steps = 0

            # Run PPO model inference and execute step
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            done = dones[0]
            steps += 1

            # Render logic: check if it's time to generate next visual frame
            current_time = time.perf_counter()
            if current_time - last_render_time >= render_interval:
                frame = base.render()
                if frame is not None:
                    self.frame_cb(frame)
                last_render_time = current_time

            # Update telemetry for the UI
            self._send_telemetry(base, ep, steps)

            # Maintain simulation real-time speed
            elapsed = time.perf_counter() - t0
            if sim_dt - elapsed > 0:
                time.sleep(sim_dt - elapsed)


class App(tk.Tk):
    """
    Tkinter application managing the UI components and interaction with
    the simulation thread.
    """

    def __init__(self):
        super().__init__()
        self.title("Triple Pendulum")
        self.configure(bg="#000000")

        # Setup window size and center on screen
        window_width, window_height = 1280, 720
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int((screen_width / 2) - (window_width / 2))
        center_y = int((screen_height / 2) - (window_height / 2))
        self.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")

        self.bind("<Escape>", lambda e: self._quit())
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self._current_target = 0
        self._frame_image = None
        self._last_telemetry = {}
        self._btn_refs = []

        # Build UI and start simulation thread
        self._build_ui()
        self._sim = SimThread(frame_cb=self._on_frame, telemetry_cb=self._on_telemetry)
        self._sim.start()
        self._poll()

    def _build_ui(self):
        """Constructs the UI layout and buttons."""
        self._canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
        self._canvas.pack(side="top", fill="both", expand=True)

        bottom_bar = tk.Frame(self, bg="#111111", height=60)
        bottom_bar.pack(side="bottom", fill="x")
        bottom_bar.pack_propagate(False)

        self._tele_lbl = tk.Label(
            bottom_bar,
            text="EP: 0 | STEP: 0 | CART: 0.000m | ANGLES: 0.0° 0.0° 0.0°",
            bg="#111111",
            fg="#aaaaaa",
            font=("Consolas", 10),
        )
        self._tele_lbl.pack(side="top", anchor="w", padx=10, pady=(5, 0))

        # Build control buttons
        controls = tk.Frame(bottom_bar, bg="#111111")
        controls.pack(side="top", fill="x", padx=10, pady=(2, 5))

        for i, label in enumerate(TARGET_LABELS):
            btn = tk.Button(
                controls,
                text=f"T{i+1}: {label}",
                bg="#222222",
                fg="#ffffff",
                font=("Consolas", 9),
                relief="flat",
                cursor="hand2",
                command=lambda idx=i: self._select_target(idx),
            )
            btn.pack(side="left", padx=2)
            self._btn_refs.append(btn)

        self._btn_reset = tk.Button(
            controls,
            text="RESET",
            bg="#444444",
            fg="#ffffff",
            font=("Consolas", 9, "bold"),
            relief="flat",
            cursor="hand2",
            width=8,
            command=self._trigger_reset,
        )
        self._btn_reset.pack(side="right", padx=2)

        self._btn_start = tk.Button(
            controls,
            text="START",
            bg="#1f6feb",
            fg="#ffffff",
            font=("Consolas", 9, "bold"),
            relief="flat",
            cursor="hand2",
            width=8,
            command=self._toggle_play,
        )
        self._btn_start.pack(side="right", padx=2)

        self._highlight_buttons(0)

    def _select_target(self, idx):
        self._current_target = idx
        self._sim.set_target(idx)
        self._highlight_buttons(idx)

    def _highlight_buttons(self, active_idx):
        for i, btn in enumerate(self._btn_refs):
            btn.config(bg="#1f6feb" if i == active_idx else "#222222")

    def _toggle_play(self):
        is_paused = self._sim.toggle_pause()
        if is_paused:
            self._btn_start.config(text="START", bg="#1f6feb")
        else:
            self._btn_start.config(text="STOP", bg="#da3633")

    def _trigger_reset(self):
        self._sim.force_reset()

    # Callbacks to receive data from simulation thread
    def _on_frame(self, rgb: np.ndarray):
        self._pending_frame = rgb

    def _on_telemetry(self, data: dict):
        self._last_telemetry = data

    def _poll(self):
        """
        The polling loop for the UI. Checks for new frames/telemetry
        and updates widgets. This runs every 16ms (~60Hz).
        """
        # Render the most recent frame
        frame = getattr(self, "_pending_frame", None)
        if frame is not None:
            self._pending_frame = None
            cw, ch = self._canvas.winfo_width(), self._canvas.winfo_height()
            if cw > 10 and ch > 10:
                frame_h, frame_w = frame.shape[:2]
                img = Image.fromarray(frame)

                # Resize image if canvas size has changed
                if frame_w != cw or frame_h != ch:
                    img = img.resize((cw, ch), Image.Resampling.BILINEAR)

                self._frame_image = ImageTk.PhotoImage(img)
                self._canvas.create_image(0, 0, anchor="nw", image=self._frame_image)

        # Update text labels from telemtry
        t = self._last_telemetry
        if t:
            cart = t.get("cart_x", 0.0)
            ang = t.get("angles", [0, 0, 0])
            self._tele_lbl.config(
                text=f"EP: {t.get('episode', '-')} | "
                f"STEP: {t.get('step', '-')} | "
                f"CART: {cart:+.3f}m | "
                f"ANGLES: {ang[0]:+6.1f}° {ang[1]:+6.1f}° {ang[2]:+6.1f}°"
            )

        # Schedule the next poll
        self.after(16, self._poll)

    def _quit(self):
        self._sim.stop()
        self.destroy()


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "glfw"
    app = App()
    app.mainloop()