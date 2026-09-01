"""Per-physics-refresh memoization for mjlab.entity.data.EntityData's
hottest, redundantly-recomputed properties.

EntityData has no caching of its own: every property access re-reads raw
sim buffers and recomputes from scratch, every time. Several `_b`-frame
properties (projected_gravity_b, root_link_lin_vel_b, root_link_ang_vel_b,
root_com_lin_vel_b, root_com_ang_vel_b, heading_w) each independently call
root_link_quat_w, which recomputes the full root_link_pose_w (indexing +
torch.cat) from scratch every call, with zero sharing even within one
sim.forward() cycle. Profiling showed this cluster costing several
seconds of self+cumulative time in a ~12800-step training run.

Per ManagerBasedRlEnv.step()'s own documented contract: derived quantities
(xpos/xquat/cvel/...) only actually change value when Simulation.forward()
runs. Between two forward() calls, every read of these properties returns
byte-for-byte identical underlying data, no matter how many managers read
it or how many times. This module caches on exactly that boundary: clear
right after forward() runs, so cached values are valid for exactly one
"generation" of derived-quantity freshness, matching mjlab's own staleness
contract precisely (including the deliberate pre-forward/post-forward
staleness split within a single step() call -- see step()'s docstring).
"""

from __future__ import annotations

import functools

from mjlab.entity.data import EntityData
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_CACHED_PROP_NAMES = (
    "root_link_pose_w",
    "root_link_vel_w",
    "root_link_quat_w",
    "root_link_pos_w",
    "root_link_lin_vel_w",
    "root_link_ang_vel_w",
    "projected_gravity_b",
    "heading_w",
    "root_link_lin_vel_b",
    "root_link_ang_vel_b",
    "root_com_lin_vel_b",
    "root_com_ang_vel_b",
)
_CACHE_ATTR = "_step_cache"

_original_properties: dict[str, property] = {}
_props_patched = False
_init_patched = False


def _cached_property(name: str, original: property):
    @functools.wraps(original.fget)
    def wrapper(self):
        cache = self.__dict__.get(_CACHE_ATTR)
        if cache is None:
            cache = {}
            object.__setattr__(self, _CACHE_ATTR, cache)
        if name not in cache:
            cache[name] = original.fget(self)
        return cache[name]

    return property(wrapper)


def _clear(entity_data: EntityData) -> None:
    cache = entity_data.__dict__.get(_CACHE_ATTR)
    if cache is not None:
        cache.clear()


def _patch_entity_data_properties() -> None:
    global _props_patched
    if _props_patched:
        return
    for name in _CACHED_PROP_NAMES:
        original = getattr(EntityData, name)
        assert isinstance(original, property), (
            f"EntityData.{name} is not a plain property; caching wrapper "
            "assumptions may not hold -- inspect before re-enabling."
        )
        _original_properties[name] = original
        setattr(EntityData, name, _cached_property(name, original))
    _props_patched = True


def apply() -> None:
    """Enable per-forward() caching for EntityData's hot properties.

    Call this ONCE, before any ManagerBasedRlEnv is constructed.
    """
    global _init_patched
    if _init_patched:
        return
    _patch_entity_data_properties()

    original_init = ManagerBasedRlEnv.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_forward = self.sim.forward

        @functools.wraps(original_forward)
        def patched_forward(*f_args, **f_kwargs):
            result = original_forward(*f_args, **f_kwargs)
            for entity in self.scene.entities.values():
                _clear(entity.data)
            return result

        self.sim.forward = patched_forward

    ManagerBasedRlEnv.__init__ = patched_init
    _init_patched = True
