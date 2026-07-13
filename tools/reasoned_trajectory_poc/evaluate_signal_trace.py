from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SIGNAL_STOP_SOURCES = ("traffic_light_stop", "stop_line_s")


def _labels(record: dict[str, Any]) -> set[str]:
  return {str(label) for label in record.get("qwen_labels") or ()}


def _sources(record: dict[str, Any]) -> set[str]:
  return {str(source) for source in record.get("durable_speed_plan_sources") or ()}


def _rtp_text(record: dict[str, Any]) -> str:
  return str(record.get("rtp_text") or "")


def _has_signal_stop_source(record: dict[str, Any]) -> bool:
  return any(source == "traffic_light_stop" or source.startswith("stop_line_s") for source in _sources(record))


def _has_non_signal_speed_source(record: dict[str, Any]) -> bool:
  return any(not (source == "traffic_light_stop" or source.startswith("stop_line_s")) for source in _sources(record))


def _float_or_none(value: Any) -> float | None:
  try:
    if value is None:
      return None
    return float(value)
  except (TypeError, ValueError):
    return None


def _red_or_stop_rtp(record: dict[str, Any]) -> bool:
  text = _rtp_text(record)
  return "red_signal_for_path" in text or "stop_sign_for_path" in text


def _green_rtp(record: dict[str, Any]) -> bool:
  return "green_signal_for_path" in _rtp_text(record)


def qwen_signal_class(record: dict[str, Any]) -> str:
  labels = _labels(record)
  if "stop_sign" in labels or "stop_sign_for_path" in _rtp_text(record):
    return "stop_sign"
  if "red_stop_light" in labels or "red_signal_for_path" in _rtp_text(record):
    return "red_stop_light"
  if "green_go_light" in labels or "green_signal_for_path" in _rtp_text(record):
    return "green_go_light"
  return ""


def signal_requirement_from_record(record: dict[str, Any], *, passed_ignore_m: float) -> str:
  remaining_m = _float_or_none(record.get("traffic_light_remaining_to_stop_m"))
  if remaining_m is not None and remaining_m < -passed_ignore_m:
    return "passed_signal"
  labels = _labels(record)
  text = _rtp_text(record)
  if "stop_sign" in labels or "stop_sign_for_path" in text:
    return "stop_sign"
  state = str(record.get("traffic_light_state") or "")
  if state == "red":
    return "red_stop_light"
  if state == "green":
    return "green_go_light"
  return ""


def _target_speed_ok_for_red_or_stop(record: dict[str, Any], *, full_stop_m: float, slow_distance_m: float, stop_speed_mps: float) -> bool:
  desired = _float_or_none(record.get("desired_speed_mps")) or 0.0
  target = _float_or_none(record.get("target_speed_mps"))
  remaining = _float_or_none(record.get("traffic_light_remaining_to_stop_m"))
  if target is None or remaining is None:
    return False
  if remaining <= full_stop_m:
    return target <= stop_speed_mps
  if remaining <= slow_distance_m:
    return target < desired - 0.05
  return _has_signal_stop_source(record) or _red_or_stop_rtp(record)


def _green_release_ok(record: dict[str, Any], *, min_speed_fraction: float) -> bool:
  if _has_signal_stop_source(record):
    return False
  if _has_non_signal_speed_source(record):
    return True
  desired = _float_or_none(record.get("desired_speed_mps")) or 0.0
  target = _float_or_none(record.get("target_speed_mps"))
  if target is None:
    return False
  return target >= desired * min_speed_fraction


def _age_ok(record: dict[str, Any], max_age_frames: int) -> bool:
  age = record.get("control_consumed_age_frames", record.get("rtp_age_frames"))
  if age is None:
    return False
  try:
    return int(age) <= max_age_frames
  except (TypeError, ValueError):
    return False


def _rate(ok: int, total: int) -> float:
  return 1.0 if total == 0 else ok / total


def evaluate_episode(
  episode: dict[str, Any],
  *,
  max_age_frames: int = 8,
  passed_ignore_m: float = 2.0,
  full_stop_m: float = 4.0,
  slow_distance_m: float = 28.0,
  stop_speed_mps: float = 0.15,
  green_min_speed_fraction: float = 0.80,
  min_success_rate: float = 0.95,
) -> dict[str, Any]:
  rows: list[dict[str, Any]] = []
  previous_remaining_m: float | None = None
  counts = {
    "red_required": 0,
    "red_qwen_ok": 0,
    "red_control_ok": 0,
    "green_required": 0,
    "green_qwen_ok": 0,
    "green_control_ok": 0,
    "stop_sign_required": 0,
    "stop_sign_qwen_ok": 0,
    "stop_sign_control_ok": 0,
    "passed_signal_frames": 0,
    "red_pass_violations": 0,
    "age_violations": 0,
    "green_path_mismatch": 0,
  }

  for record in episode.get("records", []):
    requirement = signal_requirement_from_record(record, passed_ignore_m=passed_ignore_m)
    remaining_m = _float_or_none(record.get("traffic_light_remaining_to_stop_m"))
    signal_class = qwen_signal_class(record)
    age_ok = _age_ok(record, max_age_frames)
    if not age_ok and requirement in {"red_stop_light", "green_go_light", "stop_sign"}:
      counts["age_violations"] += 1
    if record.get("green_path_matches_tracked_path") is False:
      counts["green_path_mismatch"] += 1

    red_qwen_ok = signal_class == "red_stop_light"
    green_qwen_ok = signal_class == "green_go_light"
    stop_sign_qwen_ok = signal_class == "stop_sign"
    red_control_ok = _target_speed_ok_for_red_or_stop(record, full_stop_m=full_stop_m, slow_distance_m=slow_distance_m, stop_speed_mps=stop_speed_mps)
    green_control_ok = _green_release_ok(record, min_speed_fraction=green_min_speed_fraction)
    stop_sign_control_ok = _target_speed_ok_for_red_or_stop(record, full_stop_m=full_stop_m, slow_distance_m=slow_distance_m, stop_speed_mps=stop_speed_mps)

    if requirement == "red_stop_light":
      counts["red_required"] += 1
      counts["red_qwen_ok"] += int(red_qwen_ok)
      counts["red_control_ok"] += int(red_control_ok)
    elif requirement == "green_go_light":
      counts["green_required"] += 1
      counts["green_qwen_ok"] += int(green_qwen_ok or _green_rtp(record))
      counts["green_control_ok"] += int(green_control_ok)
    elif requirement == "stop_sign":
      counts["stop_sign_required"] += 1
      counts["stop_sign_qwen_ok"] += int(stop_sign_qwen_ok)
      counts["stop_sign_control_ok"] += int(stop_sign_control_ok)
    elif requirement == "passed_signal":
      counts["passed_signal_frames"] += 1
      state = str(record.get("traffic_light_state") or "")
      if (
        state == "red" and
        previous_remaining_m is not None and
        previous_remaining_m >= -passed_ignore_m and
        remaining_m is not None and
        remaining_m < -passed_ignore_m
      ):
        counts["red_pass_violations"] += 1
    previous_remaining_m = remaining_m

    rows.append({
      "frame_id": record.get("frame_id"),
      "requirement": requirement,
      "traffic_light_state": record.get("traffic_light_state"),
      "remaining_to_stop_m": record.get("traffic_light_remaining_to_stop_m"),
      "target_speed_mps": record.get("target_speed_mps"),
      "speed_mps": record.get("speed_mps"),
      "qwen_signal_class": signal_class,
      "qwen_labels": record.get("qwen_labels") or [],
      "qwen_label_scores": record.get("qwen_label_scores") or {},
      "rtp_age_frames": record.get("rtp_age_frames"),
      "control_consumed_age_frames": record.get("control_consumed_age_frames"),
      "rtp_valid": record.get("reasoned_valid"),
      "rtp_text": record.get("rtp_text") or "",
      "durable_speed_plan_sources": sorted(_sources(record)),
      "visual_signal_label": record.get("visual_signal_label"),
      "visual_signal_guard_enabled": bool(record.get("visual_signal_guard_enabled")),
      "red_control_ok": red_control_ok,
      "green_control_ok": green_control_ok,
      "stop_sign_control_ok": stop_sign_control_ok,
      "age_ok": age_ok,
      "green_path_matches_tracked_path": record.get("green_path_matches_tracked_path"),
    })

  rates = {
    "red_qwen_success_rate": _rate(counts["red_qwen_ok"], counts["red_required"]),
    "red_control_success_rate": _rate(counts["red_control_ok"], counts["red_required"]),
    "green_qwen_success_rate": _rate(counts["green_qwen_ok"], counts["green_required"]),
    "green_control_success_rate": _rate(counts["green_control_ok"], counts["green_required"]),
    "stop_sign_qwen_success_rate": _rate(counts["stop_sign_qwen_ok"], counts["stop_sign_required"]),
    "stop_sign_control_success_rate": _rate(counts["stop_sign_control_ok"], counts["stop_sign_required"]),
  }
  issues: list[str] = []
  rate_denominators = {
    "red_qwen_success_rate": "red_required",
    "red_control_success_rate": "red_required",
    "green_qwen_success_rate": "green_required",
    "green_control_success_rate": "green_required",
    "stop_sign_qwen_success_rate": "stop_sign_required",
    "stop_sign_control_success_rate": "stop_sign_required",
  }
  for key, rate in rates.items():
    denominator = counts.get(rate_denominators[key], 0)
    if denominator > 0 and rate < min_success_rate:
      issues.append(f"{key} {rate:.3f} < {min_success_rate:.3f}")
  if counts["red_pass_violations"] > 0:
    issues.append(f"red_pass_violations={counts['red_pass_violations']}")
  if counts["age_violations"] > 0:
    issues.append(f"age_violations={counts['age_violations']}")
  if counts["green_path_mismatch"] > 0:
    issues.append(f"green_path_mismatch={counts['green_path_mismatch']}")
  if any(row["visual_signal_guard_enabled"] for row in rows):
    issues.append("visual_signal_guard_enabled")

  return {
    "kind": "signal_trace_evaluation",
    "ok": not issues,
    "issues": issues,
    "counts": counts,
    **rates,
    "max_age_frames": max_age_frames,
    "passed_ignore_m": passed_ignore_m,
    "full_stop_m": full_stop_m,
    "slow_distance_m": slow_distance_m,
    "stop_speed_mps": stop_speed_mps,
    "green_min_speed_fraction": green_min_speed_fraction,
    "min_success_rate": min_success_rate,
    "rows": rows,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Evaluate Qwen/RTP traffic-light and stop-sign traces.")
  parser.add_argument("episode", type=Path, help="Path to vlm/episode.json")
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--max-age-frames", type=int, default=8)
  parser.add_argument("--passed-ignore-m", type=float, default=2.0)
  parser.add_argument("--full-stop-m", type=float, default=4.0)
  parser.add_argument("--slow-distance-m", type=float, default=28.0)
  parser.add_argument("--stop-speed-mps", type=float, default=0.15)
  parser.add_argument("--green-min-speed-fraction", type=float, default=0.80)
  parser.add_argument("--min-success-rate", type=float, default=0.95)
  args = parser.parse_args()

  episode = json.loads(args.episode.read_text(encoding="utf-8"))
  result = evaluate_episode(
    episode,
    max_age_frames=args.max_age_frames,
    passed_ignore_m=args.passed_ignore_m,
    full_stop_m=args.full_stop_m,
    slow_distance_m=args.slow_distance_m,
    stop_speed_mps=args.stop_speed_mps,
    green_min_speed_fraction=args.green_min_speed_fraction,
    min_success_rate=args.min_success_rate,
  )
  out = args.out or (args.episode.parent / "signal_trace_evaluation.json")
  out.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
  main()
