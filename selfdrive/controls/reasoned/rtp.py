from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional


RTP_VERSION = "RTPv1"
MAX_RTP_CHARS = 2048
DEFAULT_MAX_GENERATED_TOKENS = 48
MAX_LIST_ITEMS = 16
MAX_LAT_BIAS_M = 1.3
MAX_SPEED_CAP_MPS = 45.0
MAX_SPEED_SCALE = 1.0
MAX_STOP_S = 120.0
MAX_WEIGHT = 5.0

VALID_METAS = {
  "BASE",
  "BIAS_LEFT",
  "BIAS_RIGHT",
  "BIAS_LEFT_AND_SLOW",
  "BIAS_RIGHT_AND_SLOW",
  "SLOW",
  "YIELD",
  "STOP",
  "TAKE_LEFT_BRANCH",
  "TAKE_RIGHT_BRANCH",
  "REJECT_BASE",
  "OCCLUSION_CAUTION",
  "EMERGENCY_CAUTION",
}
VALID_BRANCHES = {"base", "C0", "C1", "C2", "C3", "C4"}
VALID_WEIGHT_KEYS = ("obs", "lane", "comfort", "base", "vlm")
FORBIDDEN_PLACEHOLDER_TOKENS = {"lower_snake_case", "token", "<token>"}
REQUIRED_FIELDS = {
  "scene",
  "evidence",
  "meta",
  "branch",
  "lat_bias_m",
  "speed_cap_mps",
  "stop_s",
  "avoid",
  "weights",
  "confidence",
}
TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
SCENE_RE = re.compile(r"^[a-z0-9_]{1,64}$")
WEIGHT_RE = re.compile(r"^(obs|lane|comfort|base|vlm)(-?\d+(?:\.\d+)?)$")


class RtpValidationError(ValueError):
  pass


@dataclass(frozen=True)
class RtpProgram:
  scene: str
  evidence: tuple[str, ...]
  meta: str
  branch: str
  lat_bias_m: float
  speed_cap_mps: Optional[float]
  stop_s: Optional[float]
  avoid: tuple[str, ...]
  weights: dict[str, float] = field(default_factory=dict)
  confidence: float = 0.0
  speed_scale: Optional[float] = None

  def to_wire_text(self) -> str:
    weights = ",".join(f"{k}{self.weights[k]:.3g}" for k in VALID_WEIGHT_KEYS if k in self.weights)
    if self.speed_scale is not None:
      speed_cap = f"{self.speed_scale * 100.0:.3g}%"
    else:
      speed_cap = "none" if self.speed_cap_mps is None else f"{self.speed_cap_mps:.3g}"
    stop_s = "none" if self.stop_s is None else f"{self.stop_s:.3g}"
    return "\n".join((
      RTP_VERSION,
      f"scene={self.scene}",
      f"evidence=[{','.join(self.evidence)}]",
      f"meta={self.meta}",
      f"branch={self.branch}",
      f"lat_bias_m={self.lat_bias_m:.3g}",
      f"speed_cap_mps={speed_cap}",
      f"stop_s={stop_s}",
      f"avoid=[{','.join(self.avoid)}]",
      f"weights=[{weights}]",
      f"confidence={self.confidence:.3g}",
    ))


def estimate_generated_tokens(text: str) -> int:
  return len([part for part in re.split(r"\s+", text.strip()) if part])


def _parse_float(raw: str, field_name: str, lower: float, upper: float) -> float:
  try:
    value = float(raw)
  except ValueError as exc:
    raise RtpValidationError(f"{field_name} is not a float") from exc
  if not lower <= value <= upper:
    raise RtpValidationError(f"{field_name} out of bounds: {value}")
  return value


def _parse_optional_float(raw: str, field_name: str, lower: float, upper: float) -> Optional[float]:
  if raw == "none":
    return None
  return _parse_float(raw, field_name, lower, upper)


def _parse_speed_cap(raw: str) -> tuple[Optional[float], Optional[float]]:
  if raw == "none":
    return None, None
  if raw.endswith("%"):
    scale_pct = _parse_float(raw[:-1], "speed_cap_mps percent", 0.0, MAX_SPEED_SCALE * 100.0)
    return None, scale_pct / 100.0
  if raw.endswith("x"):
    scale = _parse_float(raw[:-1], "speed_cap_mps scale", 0.0, MAX_SPEED_SCALE)
    return None, scale
  return _parse_float(raw, "speed_cap_mps", 0.0, MAX_SPEED_CAP_MPS), None


def _parse_list(raw: str, field_name: str) -> tuple[str, ...]:
  if not raw.startswith("[") or not raw.endswith("]"):
    raise RtpValidationError(f"{field_name} must be a bracketed list")
  inner = raw[1:-1]
  if not inner:
    return ()
  items = tuple(part.strip() for part in inner.split(","))
  if len(items) > MAX_LIST_ITEMS:
    raise RtpValidationError(f"{field_name} has too many entries")
  for item in items:
    if item in FORBIDDEN_PLACEHOLDER_TOKENS:
      raise RtpValidationError(f"{field_name} contains placeholder token: {item}")
    if not TOKEN_RE.match(item):
      raise RtpValidationError(f"{field_name} contains invalid token: {item}")
  return items


def _parse_weights(raw: str) -> dict[str, float]:
  weights: dict[str, float] = {}
  for item in _parse_list(raw, "weights"):
    match = WEIGHT_RE.match(item)
    if match is None:
      raise RtpValidationError(f"invalid weight token: {item}")
    key, value_raw = match.groups()
    if key in weights:
      raise RtpValidationError(f"duplicate weight: {key}")
    weights[key] = _parse_float(value_raw, f"weights.{key}", 0.0, MAX_WEIGHT)
  return weights


def parse_rtp(text: str, max_generated_tokens: int = DEFAULT_MAX_GENERATED_TOKENS) -> RtpProgram:
  if len(text) > MAX_RTP_CHARS:
    raise RtpValidationError("RTP output is too long")
  if estimate_generated_tokens(text) > max_generated_tokens:
    raise RtpValidationError("RTP output exceeds generated token budget")

  lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
  if not lines or lines[0] != RTP_VERSION:
    raise RtpValidationError("RTP output must start with RTPv1")

  fields: dict[str, str] = {}
  for line in lines[1:]:
    if "=" not in line:
      raise RtpValidationError(f"invalid RTP line: {line}")
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key in fields:
      raise RtpValidationError(f"duplicate RTP field: {key}")
    if key not in REQUIRED_FIELDS:
      raise RtpValidationError(f"unknown RTP field: {key}")
    fields[key] = value

  missing = REQUIRED_FIELDS - fields.keys()
  if missing:
    raise RtpValidationError(f"missing RTP fields: {sorted(missing)}")

  scene = fields["scene"]
  if SCENE_RE.match(scene) is None:
    raise RtpValidationError(f"invalid scene token: {scene}")
  if scene in FORBIDDEN_PLACEHOLDER_TOKENS:
    raise RtpValidationError(f"scene contains placeholder token: {scene}")

  evidence = _parse_list(fields["evidence"], "evidence")
  avoid = _parse_list(fields["avoid"], "avoid")

  meta = fields["meta"]
  if meta not in VALID_METAS:
    raise RtpValidationError(f"invalid meta token: {meta}")

  branch = fields["branch"]
  if branch not in VALID_BRANCHES:
    raise RtpValidationError(f"invalid branch token: {branch}")

  speed_cap_mps, speed_scale = _parse_speed_cap(fields["speed_cap_mps"])

  return RtpProgram(
    scene=scene,
    evidence=evidence,
    meta=meta,
    branch=branch,
    lat_bias_m=_parse_float(fields["lat_bias_m"], "lat_bias_m", -MAX_LAT_BIAS_M, MAX_LAT_BIAS_M),
    speed_cap_mps=speed_cap_mps,
    stop_s=_parse_optional_float(fields["stop_s"], "stop_s", 0.0, MAX_STOP_S),
    avoid=avoid,
    weights=_parse_weights(fields["weights"]),
    confidence=_parse_float(fields["confidence"], "confidence", 0.0, 1.0),
    speed_scale=speed_scale,
  )
