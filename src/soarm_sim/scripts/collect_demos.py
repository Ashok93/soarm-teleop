"""Collect scripted pick-and-place demonstrations with RGB-D observations."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    from isaaclab.app import AppLauncher
except Exception:  # pragma: no cover - allows --help without Isaac Lab installed
    AppLauncher = None

from soarm_sim.datasets.generators import PickPlaceScriptedExpert
from soarm_sim.datasets.recorders import EpisodeRecorder
from soarm_sim.tasks.pick_place import PickPlaceSceneCfg

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
EE_BODY = "gripper_link"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scripted pick-and-place demos.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to record.")
    parser.add_argument("--steps_per_episode", type=int, default=600, help="Steps per episode.")
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/pick_place"), help="Output dataset dir.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--real_time", action="store_true", help="Run in real-time, if possible.")
    parser.add_argument("--headless", action="store_true", help="Run headless.")
    parser.add_argument("--sim_device", default="cuda", help="Simulation device (e.g. cuda, cpu).")
    return parser.parse_args()


def _resolve_indices(names: list[str], target_names: list[str]) -> list[int]:
    indices = []
    for name in target_names:
        if name not in names:
            raise ValueError(f"Name '{name}' not found in {names}")
        indices.append(names.index(name))
    return indices


def _camera_numpy(camera, key: str) -> np.ndarray:
    if camera is None:
        return np.zeros((1,), dtype=np.float32)
    output = getattr(camera.data, "output", None)
    if output and key in output:
        return output[key][0].detach().cpu().numpy()
    if hasattr(camera.data, key):
        return getattr(camera.data, key)[0].detach().cpu().numpy()
    return np.zeros((1,), dtype=np.float32)


def main() -> None:
    args_cli = _parse_args()

    if AppLauncher is None:
        raise RuntimeError("Isaac Lab is not available in this environment.")

    app_launcher = AppLauncher(headless=args_cli.headless)
    simulation_app = app_launcher.app

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.math import subtract_frame_transforms

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        device=args_cli.sim_device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 0.0, 1.0], [0.0, 0.0, 0.3])

    scene_cfg = PickPlaceSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot = scene["robot"]
    cube = scene["cube"]
    cylinder = scene["cylinder"]
    sphere = scene["sphere"]
    wrist_camera = scene.get("wrist_camera", None)
    overhead_camera = scene.get("overhead_camera", None)

    joint_ids = _resolve_indices(robot.data.joint_names, JOINT_NAMES)
    gripper_id = _resolve_indices(robot.data.joint_names, [GRIPPER_JOINT])[0]
    ee_body_id = _resolve_indices(robot.data.body_names, [EE_BODY])[0]

    joint_limits = robot.data.joint_pos_limits[0]
    gripper_limits = joint_limits[gripper_id]
    gripper_open = gripper_limits[1].item()
    gripper_closed = gripper_limits[0].item()

    ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        ),
        num_envs=scene.num_envs,
        device=sim.device,
    )

    rng = np.random.default_rng(args_cli.seed)
    recorder = EpisodeRecorder(args_cli.output_dir)
    expert = PickPlaceScriptedExpert(dt=sim.get_physics_dt())

    target_zones = {
        "cube": np.array([0.50, -0.10, 0.06], dtype=np.float32),
        "cylinder": np.array([0.50, 0.00, 0.06], dtype=np.float32),
        "sphere": np.array([0.50, 0.10, 0.06], dtype=np.float32),
    }

    def reset_scene() -> None:
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()

    dt = sim.get_physics_dt()
    last_time = time.time()

    for _episode in range(args_cli.episodes):
        reset_scene()
        expert.reset()

        target_name = rng.choice(["cube", "cylinder", "sphere"])
        target_obj = {"cube": cube, "cylinder": cylinder, "sphere": sphere}[target_name]
        target_command = f"pick the {target_name} and place it in the target zone"

        for _step in range(args_cli.steps_per_episode):
            # Get object pose in base frame.
            root_pos_w = robot.data.root_state_w[:, 0:3]
            root_quat_w = robot.data.root_state_w[:, 3:7]
            obj_pos_w = target_obj.data.root_state_w[:, 0:3]
            obj_quat_w = target_obj.data.root_state_w[:, 3:7]
            obj_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, obj_pos_w, obj_quat_w)

            pick_pos = obj_pos_b[0].detach().cpu().numpy()
            place_pos = target_zones[target_name]
            expert.set_pick_place(pick_pos, place_pos)
            phase = expert.step()

            ee_pos_w = robot.data.body_state_w[:, ee_body_id, 0:3]
            ee_quat_w = robot.data.body_state_w[:, ee_body_id, 3:7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

            target_pos = torch.tensor(phase.target_pos, device=ee_pos_b.device).unsqueeze(0)
            target_quat = torch.tensor(phase.target_quat, device=ee_pos_b.device).unsqueeze(0)

            ik_controller.set_command(torch.cat([target_pos, target_quat], dim=-1))
            jacobians = robot.root_physx_view.get_jacobians()
            ee_jacobian = jacobians[:, ee_body_id, :, joint_ids]
            joint_pos = robot.data.joint_pos[:, joint_ids]
            joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, ee_jacobian, joint_pos)
            joint_target = robot.data.joint_pos.clone()
            joint_target[:, joint_ids] = joint_pos_des
            joint_target[:, gripper_id] = gripper_closed if phase.gripper_closed else gripper_open

            # Action is delta pose + gripper command for training.
            delta_pos = phase.target_pos - ee_pos_b[0].detach().cpu().numpy()
            delta_rot = np.zeros(3, dtype=np.float32)
            gripper_cmd = -1.0 if phase.gripper_closed else 1.0
            action = np.concatenate([delta_pos, delta_rot, [gripper_cmd]], axis=0)

            observation = {
                "joint_pos": robot.data.joint_pos[0].detach().cpu().numpy(),
                "joint_vel": robot.data.joint_vel[0].detach().cpu().numpy(),
                "ee_pos": ee_pos_b[0].detach().cpu().numpy(),
                "ee_quat": ee_quat_b[0].detach().cpu().numpy(),
                "cube_pos": cube.data.root_state_w[0, 0:3].detach().cpu().numpy(),
                "cube_quat": cube.data.root_state_w[0, 3:7].detach().cpu().numpy(),
                "cylinder_pos": cylinder.data.root_state_w[0, 0:3].detach().cpu().numpy(),
                "cylinder_quat": cylinder.data.root_state_w[0, 3:7].detach().cpu().numpy(),
                "sphere_pos": sphere.data.root_state_w[0, 0:3].detach().cpu().numpy(),
                "sphere_quat": sphere.data.root_state_w[0, 3:7].detach().cpu().numpy(),
                "wrist_rgb": _camera_numpy(wrist_camera, "rgb"),
                "wrist_depth": _camera_numpy(wrist_camera, "depth"),
                "overhead_rgb": _camera_numpy(overhead_camera, "rgb"),
                "overhead_depth": _camera_numpy(overhead_camera, "depth"),
            }

            recorder.add_step(observation, action, target_command)

            robot.set_joint_position_target(joint_target)
            robot.write_data_to_sim()

            sim.step()
            scene.update(dt)

            if args_cli.real_time:
                elapsed = time.time() - last_time
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()

        recorder.flush()

    simulation_app.close()


if __name__ == "__main__":
    main()
