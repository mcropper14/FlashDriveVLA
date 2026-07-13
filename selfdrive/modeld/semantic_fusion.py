from dataclasses import dataclass

import numpy as np

from .constants import ModelConstants, Plan


NEAR_HORIZON_S = 0.0
MID_HORIZON_S = 0.0
MID_ALPHA_START = 1.0
MID_ALPHA_END = 1.0


@dataclass
class SemanticFusionResult:
  applied: bool = False
  source: int = 0
  alpha: float = 0.0
  confidence: float = 0.0
  consistency: float = 0.0
  age: float = 0.0


def _enum_raw(value, default: int = 0) -> int:
  raw = getattr(value, "raw", None)
  if raw is not None:
    try:
      return int(raw)
    except (TypeError, ValueError):
      return default

  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _safe_float(value, default: float = 0.0) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _weights_for_t_idxs(t_idxs) -> np.ndarray:
  t = np.asarray(t_idxs, dtype=np.float32)
  return np.ones_like(t)[:, None]


def _as_float32_array(values, expected_len: int) -> np.ndarray | None:
  try:
    arr = np.asarray(list(values), dtype=np.float32)
  except (TypeError, ValueError):
    return None
  if arr.shape != (expected_len,):
    return None
  return arr


def _resample_xyzt(data, target_t: np.ndarray) -> np.ndarray | None:
  try:
    expected_len = len(data.t)
  except (AttributeError, TypeError):
    return None

  t = _as_float32_array(data.t, expected_len)
  x = _as_float32_array(data.x, expected_len)
  y = _as_float32_array(data.y, expected_len)
  z = _as_float32_array(data.z, expected_len)
  if t is None or x is None or y is None or z is None or len(t) < 2:
    return None
  if not np.all(np.diff(t) > 0):
    return None

  return np.column_stack([
    np.interp(target_t, t, x),
    np.interp(target_t, t, y),
    np.interp(target_t, t, z),
  ]).astype(np.float32)


def _semantic_plan_to_plan_components(semantic_plan, target_t: np.ndarray, plan_enum) -> dict[slice, np.ndarray] | None:
  components = {
    plan_enum.POSITION: _resample_xyzt(semantic_plan.position, target_t),
    plan_enum.VELOCITY: _resample_xyzt(semantic_plan.velocity, target_t),
    plan_enum.ACCELERATION: _resample_xyzt(semantic_plan.acceleration, target_t),
    plan_enum.T_FROM_CURRENT_EULER: _resample_xyzt(semantic_plan.orientation, target_t),
    plan_enum.ORIENTATION_RATE: _resample_xyzt(semantic_plan.orientationRate, target_t),
  }
  if any(values is None for values in components.values()):
    return None
  return components


def _safe_blend(stock_values: np.ndarray, semantic_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
  sanitized_semantic = np.where(np.isfinite(semantic_values), semantic_values, stock_values)
  return stock_values + weights * (sanitized_semantic - stock_values)


def apply_semantic_fusion_generic(model_output: dict[str, np.ndarray], semantic_plan, model_constants, plan_enum) -> tuple[dict[str, np.ndarray], SemanticFusionResult]:
  result = SemanticFusionResult()

  try:
    plan = model_output.get('plan')
    if not isinstance(plan, np.ndarray) or plan.ndim != 3 or plan.shape[0] < 1 or plan.shape[1] != model_constants.IDX_N:
      return model_output, result

    if _enum_raw(getattr(semantic_plan, "status", 0)) != 1:
      return model_output, result

    source = _enum_raw(getattr(semantic_plan, "source", 0))
    if source <= 1:
      return model_output, result

    target_t = np.asarray(model_constants.T_IDXS, dtype=np.float32)
    semantic_components = _semantic_plan_to_plan_components(semantic_plan, target_t, plan_enum)
    if semantic_components is None:
      return model_output, result

    weights = _weights_for_t_idxs(target_t)
    fused_plan = plan.copy()
    fused_values = fused_plan[0]
    for slc, semantic_values in semantic_components.items():
      fused_values[:, slc] = _safe_blend(fused_values[:, slc], semantic_values, weights)
    fused_plan[0] = fused_values

    fused_output = dict(model_output)
    fused_output['plan'] = fused_plan
    return fused_output, SemanticFusionResult(
      applied=True,
      source=source,
      alpha=1.0,
      confidence=float(np.clip(_safe_float(getattr(semantic_plan, "confidence", 0.0)), 0.0, 1.0)),
      consistency=float(np.clip(_safe_float(getattr(semantic_plan, "consistency", 0.0)), 0.0, 1.0)),
      age=max(_safe_float(getattr(semantic_plan, "age", 0.0)), 0.0),
    )
  except Exception:
    return model_output, result


def apply_semantic_lateral_overrides(model_output: dict[str, np.ndarray], plan_enum, desired_curvature: float | None = None) -> dict[str, np.ndarray]:
  plan = model_output.get('plan')
  if not isinstance(plan, np.ndarray) or plan.ndim != 3 or plan.shape[0] < 1:
    return model_output

  patched_output = dict(model_output)
  plan_values = plan[0]

  lat_planner_solution = patched_output.get('lat_planner_solution')
  if isinstance(lat_planner_solution, np.ndarray) and lat_planner_solution.ndim == 3 and lat_planner_solution.shape[0] >= 1 and lat_planner_solution.shape[2] >= 4:
    lat = lat_planner_solution.copy()
    lat[0, :, 0] = plan_values[:, plan_enum.POSITION.start + 0]
    lat[0, :, 1] = plan_values[:, plan_enum.POSITION.start + 1]
    lat[0, :, 2] = plan_values[:, plan_enum.T_FROM_CURRENT_EULER.start + 2]
    lat[0, :, 3] = plan_values[:, plan_enum.ORIENTATION_RATE.start + 2]
    patched_output['lat_planner_solution'] = lat

  if 'desired_curvature' in patched_output:
    if desired_curvature is None:
      patched_output = dict(patched_output)
      patched_output.pop('desired_curvature', None)
    else:
      curvature = np.array(patched_output['desired_curvature'], copy=True, dtype=np.float32)
      curvature[...] = float(desired_curvature)
      patched_output['desired_curvature'] = curvature

  return patched_output


def apply_semantic_fusion(model_output: dict[str, np.ndarray], semantic_plan) -> tuple[dict[str, np.ndarray], SemanticFusionResult]:
  return apply_semantic_fusion_generic(model_output, semantic_plan, ModelConstants, Plan)
