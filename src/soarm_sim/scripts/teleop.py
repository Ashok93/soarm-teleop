"""Keyboard teleoperation for SO-ARM101 in Isaac Sim."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from isaaclab.app import AppLauncher
except Exception:  # pragma: no cover - allows --help without Isaac Lab installed
    AppLauncher = None


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"


@dataclass
class GripperState:
    is_closed: bool = False


class JointKeyboard:
    """Keyboard device for joint-space teleop."""

    def __init__(self, joint_names: list[str], rate: float) -> None:
        import carb
        import omni

        self._carb = carb
        self._rate = rate
        self._joint_names = joint_names
        self._delta = np.zeros(len(joint_names), dtype=np.float32)
        self._toggle_gripper = False
        self._reset = False

        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = self._app_window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        self._bindings: Dict[int, Tuple[int, float]] = {}
        self._register_default_bindings()

    def _register_default_bindings(self) -> None:
        c = self._carb.input.KeyboardInput
        mapping = [
            (c.W, 0, +1.0),
            (c.S, 0, -1.0),
            (c.E, 1, +1.0),
            (c.D, 1, -1.0),
            (c.R, 2, +1.0),
            (c.F, 2, -1.0),
            (c.T, 3, +1.0),
            (c.G, 3, -1.0),
            (c.Y, 4, +1.0),
            (c.H, 4, -1.0),
        ]
        for key, joint_index, direction in mapping:
            if joint_index < len(self._joint_names):
                self._bindings[key] = (joint_index, direction)

    def _on_keyboard_event(self, event: "carb.input.KeyboardEvent", *_args) -> bool:
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            if event.input in self._bindings:
                idx, direction = self._bindings[event.input]
                self._delta[idx] += direction * self._rate
            elif event.input == self._carb.input.KeyboardInput.O:
                self._toggle_gripper = True
            elif event.input == self._carb.input.KeyboardInput.P:
                self._reset = True
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in self._bindings:
                idx, direction = self._bindings[event.input]
                self._delta[idx] -= direction * self._rate
        return True

    def advance(self) -> tuple[np.ndarray, bool, bool]:
        delta = self._delta.copy()
        toggle = self._toggle_gripper
        reset = self._reset
        self._toggle_gripper = False
        self._reset = False
        return delta, toggle, reset

    def destroy(self) -> None:
        if self._input and self._sub:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO-ARM101 teleoperation")
    parser.add_argument("--device", choices=["keyboard"], default="keyboard", help="Teleop input device.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--real_time", action="store_true", help="Run in real-time, if possible.")
    parser.add_argument("--joint_rate", type=float, default=0.6, help="Joint velocity rate (rad/s).")
    parser.add_argument("--headless", action="store_true", help="Run headless.")
    parser.add_argument("--sim_device", default="cuda", help="Simulation device (e.g. cuda, cpu).")
    parser.add_argument("--debug", action="store_true", help="Print debug info about joint targets.")
    return parser.parse_args()


def _resolve_indices(names: list[str], target_names: list[str]) -> list[int]:
    indices = []
    for name in target_names:
        if name not in names:
            raise ValueError(f"Name '{name}' not found in {names}")
        indices.append(names.index(name))
    return indices


def _clamp_to_limits(joint_targets, joint_limits, joint_ids):
    lower = joint_limits[joint_ids, 0]
    upper = joint_limits[joint_ids, 1]
    return joint_targets.clamp(min=lower, max=upper)


def main() -> None:
    args_cli = _parse_args()

    if AppLauncher is None:
        raise RuntimeError("Isaac Lab is not available in this environment.")

    app_launcher = AppLauncher(headless=args_cli.headless)
    simulation_app = app_launcher.app

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    from soarm_sim.tasks.teleop.teleop_env_cfg import TeleopSceneCfg

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        device=args_cli.sim_device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 0.0, 1.0], [0.0, 0.0, 0.3])

    scene_cfg = TeleopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot = scene["robot"]
    env_origins = scene.env_origins

    joint_ids = _resolve_indices(robot.data.joint_names, JOINT_NAMES)
    gripper_id = _resolve_indices(robot.data.joint_names, [GRIPPER_JOINT])[0]

    def reset_scene() -> None:
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()

    reset_scene()

    joint_target = robot.data.default_joint_pos.clone()
    joint_limits = robot.data.joint_pos_limits[0]
    gripper_limits = joint_limits[gripper_id]
    gripper_open = gripper_limits[1].item()
    gripper_closed = gripper_limits[0].item()
    gripper_state = GripperState()

    joint_device = JointKeyboard(JOINT_NAMES, rate=args_cli.joint_rate)
    print("[INFO] Joint mode: W/S, E/D, R/F, T/G, Y/H for joints; O toggles gripper; P resets.")

    dt = sim.get_physics_dt()
    last_time = time.time()
    debug_last = 0.0

    while simulation_app.is_running():
        delta, toggle, do_reset = joint_device.advance()
        if do_reset:
            reset_scene()
            joint_target = robot.data.default_joint_pos.clone()
        delta_t = torch.tensor(delta, device=joint_target.device).unsqueeze(0)
        joint_target[:, joint_ids] += delta_t * dt
        if toggle:
            gripper_state.is_closed = not gripper_state.is_closed
        joint_target[:, joint_ids] = _clamp_to_limits(joint_target[:, joint_ids], joint_limits, joint_ids)
        joint_target[:, gripper_id] = gripper_closed if gripper_state.is_closed else gripper_open
        robot.set_joint_position_target(joint_target)
        robot.write_data_to_sim()
        if args_cli.debug and (time.time() - debug_last) > 0.5:
            print(
                f"[DEBUG] joints={joint_target[0, joint_ids].detach().cpu().numpy()} "
                f"gripper={joint_target[0, gripper_id].item():.3f}"
            )
            debug_last = time.time()

        sim.step()
        scene.update(dt)

        if args_cli.real_time:
            elapsed = time.time() - last_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

    joint_device.destroy()

    simulation_app.close()


if __name__ == "__main__":
    main()
