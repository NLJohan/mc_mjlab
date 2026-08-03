"""Task id for the ISMPC CoM-height sine demo.

A single, play-only id: this is a scripted demo (fixed constant->sine->
constant schedule, no RL, no reward), not a control-mode split like
residual_balance's position/torque -- so there is exactly one
register_mjlab_task call.
"""

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.ismpc_demo.ismpc_demo_env_cfg import (
  ismpc_demo_env_cfg,
  ismpc_demo_rl_cfg,
)
from mc_mjlab.utils.task_naming import get_task_name

TASK_DIR = Path(__file__).resolve().parent.name

register_mjlab_task(
  task_id=get_task_name(TASK_DIR),
  env_cfg=ismpc_demo_env_cfg(),
  play_env_cfg=ismpc_demo_env_cfg(play=True),
  rl_cfg=ismpc_demo_rl_cfg(),
)
