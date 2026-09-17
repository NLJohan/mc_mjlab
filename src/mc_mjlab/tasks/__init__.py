"""RL tasks built on the mc_rtc residual action terms."""

from mjlab.utils.lab_api.tasks.importer import import_packages

from mc_mjlab.patches.entity_data_cache import apply as _apply_entity_data_cache_patch

# Must happen before any task sub-package below constructs a
# ManagerBasedRlEnv. This is the single earliest, task-agnostic point in the
# import chain, so every task registered under this package picks up the
# patch uniformly. See mc_mjlab/patches/entity_data_cache.py.
_apply_entity_data_cache_patch()

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)