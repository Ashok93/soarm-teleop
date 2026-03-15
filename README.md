# SO-ARM101 Teleop (Isaac Sim + Isaac Lab)

Minimal, self-contained teleoperation stack for SO-ARM101 using Isaac Sim 5.1.0 and Isaac Lab 2.3.0. This repo vendors only the SO-ARM101 URDF + meshes and provides a simple teleop script with joint-space and IK modes.

## Layout

- `src/soarm_sim/robots/so_arm101/urdf` includes the SO-ARM101 URDF and meshes.
- `src/soarm_sim/robots/so_arm101/so_arm101_cfg.py` defines the robot articulation.
- `src/soarm_sim/tasks/teleop/teleop_env_cfg.py` defines a minimal scene (table + robot + light).
- `src/soarm_sim/scripts/teleop.py` provides keyboard teleop in joint or IK mode.

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
uv run teleop --mode joint --device keyboard
uv run teleop --mode ik --device keyboard
```

### X11 (recommended for you)

Make sure X11 forwarding is allowed on the host. Example:

```bash
xhost +local:root
```

The compose file mounts `/tmp/.X11-unix` and passes `DISPLAY`.

### VNC (optional)

Set `ENABLE_VNC=1` in `docker-compose.yml` or when launching:

```bash
ENABLE_VNC=1 docker compose up -d
```

Then connect to port `5900` on the host. The entrypoint will start `x11vnc` and `Xvfb` when `ENABLE_VNC=1`.

## Teleop Controls

### Joint mode

Keys set joint velocity commands; release stops movement.

- `W/S`: shoulder_pan +/-
- `E/D`: shoulder_lift +/-
- `R/F`: elbow_flex +/-
- `T/G`: wrist_flex +/-
- `Y/H`: wrist_roll +/-
- `O`: toggle gripper open/close

### IK mode

Uses Isaac Lab's `Se3Keyboard` mapping:

- Translate: `W/S` (x), `A/D` (y), `Q/E` (z)
- Rotate: `I/K` (roll), `J/L` (pitch), `U/O` (yaw)
- `O`: toggle gripper
- `R`: reset target pose to current

## Notes

- Isaac Lab runs inside Isaac Sim's Python environment. That is why the runtime image must be based on an Isaac Sim image.
- The dev container is intentionally minimal and does not include Isaac Lab or Isaac Sim.
