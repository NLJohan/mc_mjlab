"""Task id for the ISMPC hybrid task.

The RL policy outputs a parameter consumed by ISMPC_Solver directly (a
CoM-height sine reference), not a joint-space residual, so there is a single
control mode and a single task id -- no position/torque split.
"""

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.ismpc_hybrid.ismpc_hybrid_env_cfg import (
  ismpc_hybrid_env_cfg,
  ismpc_hybrid_ppo_cfg,
)
from mc_mjlab.utils.task_naming import get_task_name

TASK_DIR = Path(__file__).resolve().parent.name

register_mjlab_task(
  task_id=get_task_name(TASK_DIR),
  env_cfg=ismpc_hybrid_env_cfg(),
  play_env_cfg=ismpc_hybrid_env_cfg(play=True),
  rl_cfg=ismpc_hybrid_ppo_cfg(max_iterations=500),
)