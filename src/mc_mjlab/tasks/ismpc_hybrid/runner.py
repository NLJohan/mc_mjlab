"""PPO runner for ismpc_hybrid: every checkpoint also writes an ONNX policy.

`save()` first does exactly what MjlabOnPolicyRunner does (model_N.pt), then on
rank 0 writes, next to it:

  model_N.onnx           policy + normalizer, `ismpc_contract` in its metadata
  model_N.contract.json  human-readable copy (C++ never reads it)

Any failure in the export is printed as a warning and swallowed: a broken
export must never cost a training run. A file that fails the onnxruntime
self-check is deleted, so a bad model can never be deployed by accident.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner

from mc_mjlab.tasks.ismpc_hybrid.contract import (
  ContractError,
  build_contract,
  to_json,
)

METADATA_KEY = "ismpc_contract"
SELF_CHECK_ATOL = 1e-4
TAG = "[ismpc-export]"


class IsmpcHybridOnPolicyRunner(MjlabOnPolicyRunner):
  """MjlabOnPolicyRunner + ONNX/contract export on every save."""

  def save(self, path: str, infos: Any = None) -> None:
    super().save(path, infos)
    if int(os.environ.get("RANK", "0")) != 0:
      return
    try:
      self._export_onnx_with_contract(path)
    except ContractError as e:
      print(f"{TAG} WARNING: no ONNX written for {Path(path).name}: {e}")
    except Exception:  # noqa: BLE001
      print(
        f"{TAG} WARNING: ONNX export failed for {Path(path).name}; "
        f"training continues.\n{traceback.format_exc()}"
      )

  # ------------------------------------------------------------------ #

  def _export_onnx_with_contract(self, path: str) -> None:
    ckpt = Path(path)
    out_dir, stem = ckpt.parent, ckpt.stem
    onnx_path = out_dir / f"{stem}.onnx"
    json_path = out_dir / f"{stem}.contract.json"

    env = self.env.unwrapped
    policy = self.alg.get_policy()
    device, was_training = self._policy_state(policy)
    try:
      # Built BEFORE exporting: an unsupported obs term aborts with no file.
      contract = build_contract(
        env, policy, checkpoint_path=path, iteration=self.current_learning_iteration
      )

      self.export_policy_to_onnx(str(out_dir), onnx_path.name)
      self._check_io_shapes(onnx_path, contract)
      attach_metadata_to_onnx(str(onnx_path), {METADATA_KEY: to_json(contract)})
      json_path.write_text(to_json(contract, indent=2))

      try:
        self._self_check(onnx_path, contract)
      except Exception:
        onnx_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise
    finally:
      self._restore_policy_state(policy, device, was_training)

    print(
      f"{TAG} {onnx_path.name}: obs_dim={contract['obs']['dim']} "
      f"action_dim={contract['action']['dim']} latch_ticks={contract['latch_ticks']} "
      f"controller_dt={contract['controller_dt']} (self-check ok)"
    )

  # ------------------------------------------------------------------ #

  @staticmethod
  def _policy_state(policy: Any) -> tuple[torch.device | None, bool | None]:
    params = list(policy.parameters()) if hasattr(policy, "parameters") else []
    device = params[0].device if params else None
    return device, getattr(policy, "training", None)

  @staticmethod
  def _restore_policy_state(
    policy: Any, device: torch.device | None, was_training: bool | None
  ) -> None:
    """The export path calls .to('cpu') / .eval(); if that ever reached the
    live policy (instead of a copy), put it back so training is unaffected."""
    if device is not None and hasattr(policy, "parameters"):
      params = list(policy.parameters())
      if params and params[0].device != device:
        print(f"{TAG} WARNING: export moved the live policy to {params[0].device}; restoring {device}.")
        policy.to(device)
    if was_training is not None and getattr(policy, "training", None) != was_training:
      policy.train(was_training)

  @staticmethod
  def _check_io_shapes(onnx_path: Path, contract: dict) -> None:
    import onnx

    graph = onnx.load(str(onnx_path)).graph

    def dims(v: Any) -> list[int]:
      return [d.dim_value for d in v.type.tensor_type.shape.dim]

    in_dims, out_dims = dims(graph.input[0]), dims(graph.output[0])
    obs_dim, act_dim = contract["obs"]["dim"], contract["action"]["dim"]
    if in_dims != [1, obs_dim]:
      raise RuntimeError(f"ONNX input shape {in_dims} != [1, {obs_dim}] from the obs terms")
    if out_dims != [1, act_dim]:
      raise RuntimeError(f"ONNX output shape {out_dims} != [1, {act_dim}]")

  def _self_check(self, onnx_path: Path, contract: dict) -> None:
    """Exported file (through onnxruntime) vs the torch export module, one
    random observation. Skipped with a warning if onnxruntime is missing."""
    try:
      import onnxruntime as ort
    except ImportError:
      print(f"{TAG} WARNING: onnxruntime not installed, self-check skipped (uv add onnxruntime).")
      return

    ref = self.alg.get_policy().as_onnx(verbose=False)
    ref.to("cpu")
    ref.eval()
    torch.manual_seed(0)
    inputs = tuple(torch.randn_like(t) for t in ref.get_dummy_inputs())
    with torch.no_grad():
      expected = ref(*inputs)
    if isinstance(expected, (tuple, list)):
      expected = expected[0]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {n: t.numpy() for n, t in zip(ref.input_names, inputs)})[0]

    if not np.all(np.isfinite(got)):
      raise RuntimeError("self-check: ONNX output is not finite")
    err = float(np.max(np.abs(got - expected.detach().numpy())))
    if err > SELF_CHECK_ATOL:
      raise RuntimeError(f"self-check: max |onnx - torch| = {err:.2e} > {SELF_CHECK_ATOL:.0e}")
