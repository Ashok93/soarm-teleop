"""Teleop scene for SO-ARM101."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from soarm_sim.robots.so_arm101 import SO_ARM101_CFG

GROUND_Z = -1.05
TABLE_POS = (0.55, 0.0, 0.0)
TABLE_ROT = (0.70711, 0.0, 0.0, 0.70711)
TABLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
LIGHT_COLOR = (0.75, 0.75, 0.75)
LIGHT_INTENSITY = 2500.0


@configclass
class TeleopSceneCfg(InteractiveSceneCfg):
    """Minimal scene for SO-ARM101 teleoperation."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS, rot=TABLE_ROT),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=LIGHT_COLOR, intensity=LIGHT_INTENSITY),
    )

    robot: ArticulationCfg = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
