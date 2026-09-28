"""Manual constant-policy play script, hard-coded physical values.

Run: uv run python scripts/manual_play.py
Edit the MANUAL block below, no CLI args.
"""
from __future__ import annotations

import torch

import mjlab
import mjlab.tasks  # noqa: F401  (populate registry)
import mc_mjlab.tasks  # noqa: F401  (adjust if your package registers tasks differently)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

TASK_ID = "Mc-Mjlab-Ismpc-Hybrid-Ismpc-Walking-Hrp5P"
ACTION_NAME = "ismpc_sine"

# =========================== MANUAL (physical units) ===========================
# Edit these, rerun. All values are what reaches the controller, i.e. the
# OUTPUT of the mapping, not raw policy actions.
WALK = True                 # walk gate
TS = 1.1                    # step timing, s        (action term clamps to [0.4, 2.0])
VX, VY, OMEGA = 0.0, 0.0, 0.0   # twist target, m/s, m/s, rad/s (rate-limited by twist_max_delta)
COM_OFFSET = 0.9            # CoM height sine offset, m   (clamped 0.4..1.05)
COM_FREQ = 1.0              # sine frequency, Hz          (clamped 0.1..8.0)
COM_SIN_AMP = 0.0           # m
COM_COS_AMP = 0.0           # m
# ===============================================================================


def _patch_mappings(term) -> None:
    """Make the action term's raw->physical maps return the constants above.

    Only the mapping is replaced; the slow-cadence update, twist rate limiter,
    interpolation and datastore writes all run exactly as in training.
    """
    from mc_mjlab.actions import ismpc_sine_action as m

    def col(x, ref):
        return torch.full_like(ref[..., 0], float(x))

    def map_to_physical(raw):
        offset = min(max(COM_OFFSET, m.OFFSET_MIN), m.OFFSET_MAX)
        freq = min(max(COM_FREQ, m.FREQUENCY_MIN), m.FREQUENCY_MAX)
        # Same radius clamp as the real mapping so amplitudes stay valid.
        r = (COM_SIN_AMP**2 + COM_COS_AMP**2) ** 0.5
        ratio = offset / max(r, offset)
        return {
            "offset": col(offset, raw),
            "frequency": col(freq, raw),
            "sin_amp": col(COM_SIN_AMP * ratio, raw),
            "cos_amp": col(COM_COS_AMP * ratio, raw),
        }

    def map_walk_gate(raw):
        return torch.full_like(raw, WALK, dtype=torch.bool)

    def map_step_timing(raw):
        return torch.full_like(raw, min(max(TS, m.TS_MIN), m.TS_MAX))

    def map_twist(raw):
        return torch.tensor([VX, VY, OMEGA], device=raw.device, dtype=raw.dtype).expand_as(raw).clone()

    term._map_to_physical = map_to_physical
    term._map_walk_gate = map_walk_gate
    term._map_step_timing = map_step_timing
    term._map_twist = map_twist


class ConstantPolicy:
    """Raw action is irrelevant (mappings are patched); zeros of the right shape."""

    def __init__(self, action_shape, device):
        self._a = torch.zeros(action_shape, device=device)

    def __call__(self, obs):
        del obs
        return self._a


def main():
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(TASK_ID, play=True)
    # Keep only the debug_* reward terms (all have weight 1.0 in play mode, so
    # they show up in the viewer's plot panel). Everything else is dropped.
    env_cfg.rewards = {
        name: term for name, term in env_cfg.rewards.items() if name.startswith("debug_")
    }
    cmd = env_cfg.commands["twist"]
    cmd.ranges.lin_vel_x = (VX, VX)
    cmd.ranges.lin_vel_y = (VY, VY)
    cmd.ranges.ang_vel_z = (OMEGA, OMEGA)
    agent_cfg = load_rl_cfg(TASK_ID)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    _patch_mappings(env.action_manager.get_term(ACTION_NAME))
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    policy = ConstantPolicy(env.unwrapped.action_space.shape, env.unwrapped.device)
    NativeMujocoViewer(env, policy).run()
    env.close()


if __name__ == "__main__":

    main()