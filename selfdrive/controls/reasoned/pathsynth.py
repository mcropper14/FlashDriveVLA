from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence

try:
  from openpilot.selfdrive.controls.lib.drive_helpers import MAX_CURVATURE, MAX_LATERAL_ACCEL_NO_ROLL, MIN_SPEED
except Exception:
  MAX_CURVATURE = 0.2
  MAX_LATERAL_ACCEL_NO_ROLL = 3.0
  MIN_SPEED = 1.0

try:
  from openpilot.selfdrive.controls.reasoned.rtp import RtpProgram
except ModuleNotFoundError:
  from selfdrive.controls.reasoned.rtp import RtpProgram


MAX_COMPILED_LAT_BIAS_M = 1.3
MIN_CONFIDENCE_TO_ACT = 0.2


@dataclass(frozen=True)
class BasePlan:
  frame_id: int
  model_log_mono_time_ns: int
  t: tuple[float, ...]
  x: tuple[float, ...]
  y: tuple[float, ...]
  speeds: tuple[float, ...]
  desired_curvature: float
  v_ego: float

  @property
  def current_speed(self) -> float:
    if self.v_ego > 0.01:
      return self.v_ego
    if self.speeds:
      return max(0.0, self.speeds[0])
    return 0.0

  @property
  def desired_speed(self) -> float:
    if self.speeds:
      return max(0.0, self.speeds[0])
    return self.current_speed

  @classmethod
  def from_model(cls, model_msg, model_log_mono_time_ns: int, v_ego: float = 0.0) -> "BasePlan":
    return cls(
      frame_id=int(getattr(model_msg, "frameId", 0)),
      model_log_mono_time_ns=int(model_log_mono_time_ns),
      t=tuple(_float_list(getattr(getattr(model_msg, "position", None), "t", []))),
      x=tuple(_float_list(getattr(getattr(model_msg, "position", None), "x", []))),
      y=tuple(_float_list(getattr(getattr(model_msg, "position", None), "y", []))),
      speeds=tuple(_float_list(getattr(getattr(model_msg, "velocity", None), "x", []))),
      desired_curvature=float(getattr(getattr(model_msg, "action", None), "desiredCurvature", 0.0)),
      v_ego=float(v_ego),
    ).with_defaults()

  def with_defaults(self) -> "BasePlan":
    if self.x and self.y and self.t:
      return self
    horizon = tuple(float(i * 5) for i in range(17))
    return BasePlan(
      frame_id=self.frame_id,
      model_log_mono_time_ns=self.model_log_mono_time_ns,
      t=tuple(float(i) * 0.2 for i in range(len(horizon))),
      x=horizon,
      y=tuple(0.0 for _ in horizon),
      speeds=tuple(max(self.v_ego, 0.0) for _ in horizon),
      desired_curvature=self.desired_curvature,
      v_ego=self.v_ego,
    )


@dataclass(frozen=True)
class CandidatePath:
  name: str
  desired_curvature: float
  lateral_offset_m: float
  speed_cap_mps: float
  stop_s: Optional[float]
  cost: float
  feasible: bool


@dataclass(frozen=True)
class PathSynthResult:
  valid: bool
  frame_id: int
  desired_curvature: float
  selected_candidate: str
  candidates: tuple[CandidatePath, ...] = field(default_factory=tuple)
  speed_cap_mps: float = 0.0
  stop_s: Optional[float] = None
  vlm_changed_path_meters: float = 0.0
  vlm_changed_speed_mps: float = 0.0
  invalid_reason: str = ""


class PathSynth:
  def compile(self, base_plan: BasePlan, program: RtpProgram) -> PathSynthResult:
    if program.confidence < MIN_CONFIDENCE_TO_ACT:
      return _base_result(base_plan, "confidence_below_threshold")

    base_speed = max(0.0, base_plan.desired_speed)
    requested_cap = _resolve_speed_cap(program, base_speed)
    if program.stop_s is not None:
      requested_cap = min(requested_cap, _stop_speed_cap(program.stop_s, base_speed))

    candidates = (
      self._candidate_base(base_plan, program, base_speed),
      self._candidate_bias(base_plan, program, requested_cap),
      self._candidate_slow(base_plan, program, requested_cap),
      self._candidate_branch(base_plan, program, requested_cap),
      self._candidate_creep(base_plan, program, requested_cap),
    )
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
      return PathSynthResult(
        valid=False,
        frame_id=base_plan.frame_id,
        desired_curvature=base_plan.desired_curvature,
        selected_candidate="none",
        candidates=candidates,
        invalid_reason="no_feasible_candidate",
      )

    selected = min(feasible, key=lambda c: c.cost)
    return PathSynthResult(
      valid=True,
      frame_id=base_plan.frame_id,
      desired_curvature=selected.desired_curvature,
      selected_candidate=selected.name,
      candidates=candidates,
      speed_cap_mps=selected.speed_cap_mps,
      stop_s=selected.stop_s,
      vlm_changed_path_meters=abs(selected.lateral_offset_m),
      vlm_changed_speed_mps=max(0.0, base_speed - selected.speed_cap_mps),
    )

  def _candidate_base(self, base_plan: BasePlan, program: RtpProgram, base_speed: float) -> CandidatePath:
    cost = _weight(program, "base", 1.0)
    if program.meta == "REJECT_BASE":
      cost += 100.0 * program.confidence
    if program.avoid:
      cost += 20.0 * _weight(program, "obs", 1.0) * program.confidence
    if abs(program.lat_bias_m) > 1e-3:
      cost += 10.0 * program.confidence
    if _requests_slowing(program):
      cost += 4.0 * program.confidence
    return CandidatePath("C0", base_plan.desired_curvature, 0.0, base_speed, None, cost, True)

  def _candidate_bias(self, base_plan: BasePlan, program: RtpProgram, speed_cap: float) -> CandidatePath:
    lat_bias = _bounded(program.lat_bias_m, -MAX_COMPILED_LAT_BIAS_M, MAX_COMPILED_LAT_BIAS_M)
    if program.meta in ("BIAS_LEFT", "BIAS_LEFT_AND_SLOW") and lat_bias <= 0.0:
      lat_bias = max(abs(lat_bias), 0.2)
    elif program.meta in ("BIAS_RIGHT", "BIAS_RIGHT_AND_SLOW") and lat_bias >= 0.0:
      lat_bias = -max(abs(lat_bias), 0.2)
    curvature = _clip_curvature(base_plan.desired_curvature + _curvature_delta_for_bias(base_plan, lat_bias), base_plan.current_speed)
    feasible = abs(lat_bias) <= MAX_COMPILED_LAT_BIAS_M and abs(curvature) <= MAX_CURVATURE
    cost = 5.0 + _weight(program, "lane", 1.0) * abs(lat_bias)
    if abs(program.lat_bias_m) > 1e-3 or program.meta in ("BIAS_LEFT", "BIAS_RIGHT", "BIAS_LEFT_AND_SLOW", "BIAS_RIGHT_AND_SLOW", "REJECT_BASE"):
      cost -= 8.0 * program.confidence
    if program.avoid:
      cost -= 5.0 * _weight(program, "obs", 1.0) * program.confidence
    return CandidatePath("C1", curvature, lat_bias, speed_cap, program.stop_s, cost, feasible)

  def _candidate_slow(self, base_plan: BasePlan, program: RtpProgram, speed_cap: float) -> CandidatePath:
    cost = 7.0 + _weight(program, "comfort", 1.0)
    if _requests_slowing(program):
      cost -= 8.0 * program.confidence
    return CandidatePath("C2", base_plan.desired_curvature, 0.0, speed_cap, program.stop_s, cost, True)

  def _candidate_branch(self, base_plan: BasePlan, program: RtpProgram, speed_cap: float) -> CandidatePath:
    branch_delta = 0.0
    if program.meta == "TAKE_LEFT_BRANCH":
      branch_delta = -0.003
    elif program.meta == "TAKE_RIGHT_BRANCH":
      branch_delta = 0.003
    elif program.branch in ("C3", "C4"):
      branch_delta = 0.002
    curvature = _clip_curvature(base_plan.desired_curvature + branch_delta, base_plan.current_speed)
    requested = program.meta in ("TAKE_LEFT_BRANCH", "TAKE_RIGHT_BRANCH") or program.branch in ("C3", "C4")
    cost = 8.0 - (7.0 * program.confidence if requested else 0.0)
    return CandidatePath("C3", curvature, math.copysign(0.35, branch_delta) if branch_delta else 0.0, speed_cap, program.stop_s, cost, abs(curvature) <= MAX_CURVATURE)

  def _candidate_creep(self, base_plan: BasePlan, program: RtpProgram, speed_cap: float) -> CandidatePath:
    creep_cap = min(speed_cap, 2.0)
    requested = program.meta in ("YIELD", "STOP") or program.stop_s is not None
    cost = 9.0 - (8.0 * program.confidence if requested else 0.0)
    return CandidatePath("C4", base_plan.desired_curvature, 0.0, creep_cap, program.stop_s, cost, True)


def _base_result(base_plan: BasePlan, reason: str = "") -> PathSynthResult:
  return PathSynthResult(
    valid=True,
    frame_id=base_plan.frame_id,
    desired_curvature=base_plan.desired_curvature,
    selected_candidate="C0",
    candidates=(CandidatePath("C0", base_plan.desired_curvature, 0.0, base_plan.current_speed, None, 0.0, True),),
    speed_cap_mps=base_plan.current_speed,
    invalid_reason=reason,
  )


def _float_list(values: Sequence[float]) -> list[float]:
  try:
    return [float(value) for value in values]
  except TypeError:
    return []


def _weight(program: RtpProgram, key: str, default: float) -> float:
  return float(program.weights.get(key, default))


def _bounded(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


def _curvature_delta_for_bias(base_plan: BasePlan, lat_bias_m: float) -> float:
  lookahead_m = max(12.0, min(40.0, base_plan.current_speed * 2.0 + 8.0))
  return 2.0 * lat_bias_m / (lookahead_m ** 2)


def _clip_curvature(curvature: float, v_ego: float) -> float:
  v = max(v_ego, MIN_SPEED)
  accel_limited = MAX_LATERAL_ACCEL_NO_ROLL / (v ** 2)
  return _bounded(curvature, -min(MAX_CURVATURE, accel_limited), min(MAX_CURVATURE, accel_limited))


def _requests_slowing(program: RtpProgram) -> bool:
  return program.speed_cap_mps is not None or program.speed_scale is not None or program.stop_s is not None or program.meta in (
    "BIAS_LEFT_AND_SLOW",
    "BIAS_RIGHT_AND_SLOW",
    "SLOW",
    "YIELD",
    "STOP",
    "OCCLUSION_CAUTION",
    "EMERGENCY_CAUTION",
  )


def _resolve_speed_cap(program: RtpProgram, desired_speed_mps: float) -> float:
  cap = desired_speed_mps
  if program.speed_scale is not None:
    cap = min(cap, desired_speed_mps * _bounded(program.speed_scale, 0.0, 1.0))
  if program.speed_cap_mps is not None:
    cap = min(cap, program.speed_cap_mps)
  return cap


def _stop_speed_cap(stop_s: float, desired_speed_mps: float) -> float:
  if stop_s <= 8.0:
    return 0.0
  if stop_s <= 20.0:
    return desired_speed_mps * 0.4
  return desired_speed_mps * 0.8
