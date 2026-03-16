# SO-ARM101 Teleop (Isaac Sim + Isaac Lab)

Minimal, self-contained teleoperation stack for SO-ARM101 using Isaac Sim 5.1.0 and Isaac Lab 2.3.0. This repo vendors only the SO-ARM101 URDF + meshes and provides a simple joint-space teleop script.

## Layout

- `src/soarm_sim/robots/so_arm101/urdf` includes the SO-ARM101 URDF and meshes.
- `src/soarm_sim/robots/so_arm101/so_arm101_cfg.py` defines the robot articulation.
- `src/soarm_sim/tasks/teleop/teleop_env_cfg.py` defines a minimal scene (table + robot + light).
- `src/soarm_sim/scripts/teleop.py` provides keyboard teleop in joint space.
- `src/soarm_sim/tasks/pick_place/pick_place_env_cfg.py` defines the pick-and-place scene with standard shapes and cameras.
- `src/soarm_sim/scripts/collect_demos.py` collects scripted pick-and-place demos.

## Docker (Vast.ai / GPU)

We build on top of the working Isaac Sim 5.1.0 image, then install Isaac Lab 2.3.0 + our code using uv.

1. Choose the base image tag that works on your Vast.ai machine:

```bash
export ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
```

2. Build and run:

```bash
docker compose build
docker compose up -d
```

3. Enter the container:

```bash
docker compose exec soarm-sim bash
```

4. Run teleop:

```bash
uv run teleop --device keyboard

## Pick-and-Place Demos (Vision + Language)

Collect scripted demonstrations with multi-view RGB-D observations.

```bash
uv run collect-demos --episodes 5 --output_dir datasets/pick_place
```

Notes:
- Uses Isaac Sim standard primitive assets for cube/cylinder/sphere.
- Scripted expert is deterministic and object-relative; replace with a more advanced policy as needed.
```

### X11 (recommended for you)

Make sure X11 forwarding is allowed on the host. Example:

```bash
xhost +local:root
```

The compose file mounts `/tmp/.X11-unix` and passes `DISPLAY`.

## Teleop Controls

### Joint mode

Keys set joint velocity commands; release stops movement.

- `W/S`: shoulder_pan +/-
- `E/D`: shoulder_lift +/-
- `R/F`: elbow_flex +/-
- `T/G`: wrist_flex +/-
- `Y/H`: wrist_roll +/-
- `O`: toggle gripper open/close

## Notes

- Isaac Lab runs inside Isaac Sim's Python environment. That is why the runtime image must be based on an Isaac Sim image.
- The dev container is intentionally minimal and does not include Isaac Lab or Isaac Sim.
