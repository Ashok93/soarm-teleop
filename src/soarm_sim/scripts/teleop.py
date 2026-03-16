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
EE_BODY = "gripper_link"


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


class ClickTarget:
    """Click in viewport to set an IK target on a fixed Z plane."""

    def __init__(self, plane_z: float) -> None:
        import carb
        import omni
        import omni.kit.viewport.utility as viewport_utils

        self._carb = carb
        self._plane_z = plane_z
        self._target = None
        self._has_target = False

        self._viewport_window = viewport_utils.get_active_viewport_window()
        self._viewport_api = self._viewport_window.viewport_api if self._viewport_window else None
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._mouse = self._app_window.get_mouse()
        self._sub = self._input.subscribe_to_mouse_events(self._mouse, self._on_mouse_event)

    def _on_mouse_event(self, event: "carb.input.MouseEvent", *_args) -> bool:
        if event.type != self._carb.input.MouseEventType.LEFT_BUTTON_RELEASE:
            return True
        if self._viewport_api is None:
            return True
        try:
            origin, direction = self._viewport_api.get_camera_ray(event.x, event.y)
        except Exception:
            return True
        if abs(direction[2]) < 1e-6:
            return True
        t = (self._plane_z - origin[2]) / direction[2]
        if t <= 0:
            return True
        hit = origin + direction * t
        self._target = hit
        self._has_target = True
        return True

    def read(self):
        if not self._has_target:
            return None
        return self._target

    def destroy(self) -> None:
        if self._input and self._sub:
            self._input.unsubscribe_from_mouse_events(self._mouse, self._sub)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO-ARM101 teleoperation")
    parser.add_argument(
        "--mode", choices=["joint", "ik", "ik_click", "ik_target"], default="joint", help="Teleop mode."
    )
    parser.add_argument("--device", choices=["keyboard"], default="keyboard", help="Teleop input device.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--real_time", action="store_true", help="Run in real-time, if possible.")
    parser.add_argument("--joint_rate", type=float, default=0.6, help="Joint velocity rate (rad/s).")
    parser.add_argument("--ik_pos_step", type=float, default=0.003, help="IK position step size.")
    parser.add_argument("--ik_rot_step", type=float, default=0.01, help="IK rotation step size (rad).")
    parser.add_argument("--ik_blend", type=float, default=0.15, help="Blend to smooth IK targets (0-1).")
    parser.add_argument("--ik_click_plane_z", type=float, default=0.12, help="Click target plane height.")
    parser.add_argument(
        "--ik_target",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Target position for ik_target mode (meters, robot frame).",
    )
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


def _normalize_se3_command(cmd):
    """Return (delta_pose[6], gripper_cmd) from Se3Keyboard output."""
    import torch

    if isinstance(cmd, tuple):
        if len(cmd) == 2:
            delta, gripper = cmd
            delta = torch.as_tensor(delta)
            gripper = torch.as_tensor(gripper).reshape(-1)
            return delta[:6], gripper[0]
    cmd_t = torch.as_tensor(cmd)
    if cmd_t.numel() >= 7:
        return cmd_t[:6], cmd_t[6].reshape(())
    return cmd_t[:6], torch.tensor(0.0, device=cmd_t.device)


def main() -> None:
    args_cli = _parse_args()

    if AppLauncher is None:
        raise RuntimeError("Isaac Lab is not available in this environment.")

    app_launcher = AppLauncher(headless=args_cli.headless)
    simulation_app = app_launcher.app

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.math import apply_delta_pose, subtract_frame_transforms

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
    num_envs = scene.num_envs

    joint_ids = _resolve_indices(robot.data.joint_names, JOINT_NAMES)
    gripper_id = _resolve_indices(robot.data.joint_names, [GRIPPER_JOINT])[0]
    ee_body_id = _resolve_indices(robot.data.body_names, [EE_BODY])[0]

    def reset_scene() -> None:
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()

    reset_scene()

    # Compute table top height in world frame for safe IK clamping.
    table_top_z = None
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        table_prim = stage.GetPrimAtPath("/World/envs/env_0/Table")
        if table_prim.IsValid():
            bbox_cache = UsdGeom.BBoxCache(UsdGeom.Tokens.default_, ["default", "render"])
            bbox = bbox_cache.ComputeWorldBound(table_prim)
            table_top_z = bbox.GetRange().GetMax()[2]
    except Exception:
        table_top_z = None

    joint_target = robot.data.default_joint_pos.clone()
    joint_limits = robot.data.joint_pos_limits[0]
    gripper_limits = joint_limits[gripper_id]
    gripper_open = gripper_limits[1].item()
    gripper_closed = gripper_limits[0].item()
    gripper_state = GripperState()

    joint_device = None
    if args_cli.mode == "joint":
        joint_device = JointKeyboard(JOINT_NAMES, rate=args_cli.joint_rate)
        print("[INFO] Joint mode: W/S, E/D, R/F, T/G, Y/H for joints; O toggles gripper; P resets.")

    ik_device = None
    click_target = None
    ik_controller = None
    target_pos = None
    target_quat = None
    reset_target = False

    def _set_reset_flag() -> None:
        nonlocal reset_target
        reset_target = True

    if args_cli.mode in ("ik", "ik_click", "ik_target"):
        if args_cli.mode == "ik":
            ik_device = Se3Keyboard(
                Se3KeyboardCfg(
                    pos_sensitivity=args_cli.ik_pos_step,
                    rot_sensitivity=args_cli.ik_rot_step,
                    sim_device=sim.device,
                )
            )
            try:
                ik_device.add_callback("R", _set_reset_flag)
            except Exception:
                import carb

                ik_device.add_callback(carb.input.KeyboardInput.R, _set_reset_flag)
        elif args_cli.mode == "ik_click":
            click_target = ClickTarget(args_cli.ik_click_plane_z)

        ik_controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),
            num_envs=num_envs,
            device=sim.device,
        )
        if args_cli.mode == "ik":
            print("[INFO] IK mode: Se3Keyboard mapping active; R resets target.")
        elif args_cli.mode == "ik_click":
            print("[INFO] IK click mode: left-click in viewport to set target position.")
        else:
            print("[INFO] IK target mode: moving to fixed target position.")

    dt = sim.get_physics_dt()
    last_time = time.time()
    debug_last = 0.0

    while simulation_app.is_running():
        if args_cli.mode == "joint" and joint_device is not None:
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

        if args_cli.mode in ("ik", "ik_click", "ik_target") and ik_controller is not None:
            delta_pose = None
            gripper_cmd = None
            if args_cli.mode == "ik" and ik_device is not None:
                cmd = ik_device.advance()
                delta_pose, gripper_cmd = _normalize_se3_command(cmd)
            elif args_cli.mode == "ik_click" and click_target is not None:
                target_hit = click_target.read()
                if target_hit is not None:
                    delta_pose = "click"
            elif args_cli.mode == "ik_target" and args_cli.ik_target is not None:
                delta_pose = "target"

            if reset_target:
                target_pos = None
                target_quat = None
                reset_target = False

            ee_pos_w = robot.data.body_state_w[:, ee_body_id, 0:3]
            ee_quat_w = robot.data.body_state_w[:, ee_body_id, 3:7]
            root_pos_w = robot.data.root_state_w[:, 0:3]
            root_quat_w = robot.data.root_state_w[:, 3:7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

            if target_pos is None or target_quat is None:
                target_pos = ee_pos_b.clone()
                target_quat = ee_quat_b.clone()
                # Start slightly above the current pose to avoid table contact.
                target_pos[:, 2] += 0.08

            if delta_pose == "click":
                target_pos = torch.tensor(target_hit, device=ee_pos_b.device).unsqueeze(0).repeat(num_envs, 1)
            elif delta_pose == "target":
                target_pos_w = torch.tensor(args_cli.ik_target, device=ee_pos_b.device).unsqueeze(0).repeat(
                    num_envs, 1
                )
                identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=ee_pos_b.device).unsqueeze(0).repeat(
                    num_envs, 1
                )
                target_pos, _ = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, identity_quat)
            elif delta_pose is not None:
                delta_pose_t = delta_pose.to(ee_pos_b.device).unsqueeze(0).repeat(num_envs, 1)
                if torch.linalg.norm(delta_pose_t) > 1e-8:
                    target_pos, target_quat = apply_delta_pose(target_pos, target_quat, delta_pose_t)
            if target_pos is not None:
                # Clamp workspace in base frame, but compute Z from table height in world frame.
                base_z = root_pos_w[:, 2:3]
                if table_top_z is not None:
                    min_z_base = (table_top_z + 0.05) - base_z
                else:
                    min_z_base = torch.tensor([[0.12]], device=target_pos.device).repeat(num_envs, 1)
                max_z_base = min_z_base + 0.50
                target_pos[:, 0] = target_pos[:, 0].clamp(0.10, 0.50)
                target_pos[:, 1] = target_pos[:, 1].clamp(-0.30, 0.30)
                target_pos[:, 2] = torch.max(target_pos[:, 2:3], min_z_base).squeeze(1)
                target_pos[:, 2] = torch.min(target_pos[:, 2:3], max_z_base).squeeze(1)
            # Hold orientation constant to avoid pose flips.
            ik_controller.set_command(torch.cat([target_pos, target_quat], dim=-1))

            jacobians = robot.root_physx_view.get_jacobians()
            ee_jacobian = jacobians[:, ee_body_id, :, joint_ids]
            joint_pos = robot.data.joint_pos[:, joint_ids]

            joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, ee_jacobian, joint_pos)
            # Smooth IK output to avoid violent jumps.
            blended = (1.0 - args_cli.ik_blend) * joint_pos + args_cli.ik_blend * joint_pos_des
            joint_target[:, joint_ids] = _clamp_to_limits(blended, joint_limits, joint_ids)
            if gripper_cmd is not None:
                if gripper_cmd.item() > 0:
                    gripper_state.is_closed = False
                elif gripper_cmd.item() < 0:
                    gripper_state.is_closed = True
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

    if joint_device is not None:
        joint_device.destroy()
    if ik_device is not None:
        ik_device.reset()
    if click_target is not None:
        click_target.destroy()

    simulation_app.close()


if __name__ == "__main__":
    main()
