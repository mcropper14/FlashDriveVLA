#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from selfdrive.controls.reasoned.lead_semantics import NO_SLOW_LEAD_CLASSES, REQUIRED_LEAD_CLASSES
from tools.reasoned_trajectory_poc.evaluate_lead_trace import evaluate_episode


LEAD_SUITE_CLASSES = (
  "true_moving_lead",
  "slower_lead",
  "braking_lead",
  "stopped_lead",
  "cut_in_vehicle",
  "crossing_vehicle",
  "irrelevant_vehicle",
)


def _episode_path(path: Path, mode: str = "vlm") -> Path:
  if path.is_file():
    return path
  for candidate in (path / mode / "episode.json", path / "vlm" / "episode.json", path / "static" / "episode.json", path / "episode.json"):
    if candidate.exists():
      return candidate
  raise FileNotFoundError(f"no episode.json found under {path}")


def discover_latest_case_paths(root: Path, case_names: tuple[str, ...] = LEAD_SUITE_CLASSES) -> dict[str, Path]:
  paths: dict[str, Path] = {}
  if not root.exists():
    return paths
  directories = [item for item in root.iterdir() if item.is_dir()]
  for case in case_names:
    matches = []
    for directory in directories:
      name = directory.name.lower()
      if case not in name:
        continue
      try:
        _episode_path(directory)
      except FileNotFoundError:
        continue
      matches.append(directory)
    if matches:
      paths[case] = max(matches, key=lambda item: item.stat().st_mtime)
  return paths


def _case_gate(
  case_name: str,
  result: dict[str, Any],
  *,
  min_success_rate: float,
  max_false_slow_rate: float,
  require_control_success: bool,
  require_vehicle_heading_evidence: bool,
) -> tuple[bool, list[str]]:
  issues: list[str] = []
  bucket = result.get("per_class", {}).get(case_name)
  if not bucket or int(bucket.get("frames", 0)) <= 0:
    return False, [f"{case_name}: no physical frames classified as {case_name}"]

  if int(result.get("collision_count", 0)) != 0:
    issues.append(f"{case_name}: collision_count {result.get('collision_count')} != 0")
  if int(result.get("age_violation_count", 0)) != 0:
    issues.append(f"{case_name}: age_violation_count {result.get('age_violation_count')} != 0")
  if require_vehicle_heading_evidence:
    if int(result.get("vehicle_heading_checked_frames", 0)) <= 0:
      issues.append(f"{case_name}: no vehicle heading evidence checked")
    if int(result.get("vehicle_heading_missing_frames", 0)) != 0:
      issues.append(f"{case_name}: vehicle_heading_missing_frames {result.get('vehicle_heading_missing_frames')} != 0")
    if int(result.get("vehicle_heading_violation_count", 0)) != 0:
      issues.append(f"{case_name}: vehicle_heading_violation_count {result.get('vehicle_heading_violation_count')} != 0")
  false_slow_rate = result.get("false_slow_rate")
  if false_slow_rate is not None and float(false_slow_rate) > max_false_slow_rate:
    issues.append(f"{case_name}: episode false_slow_rate {false_slow_rate} > {max_false_slow_rate}")

  if case_name in REQUIRED_LEAD_CLASSES:
    qwen_rate = bucket.get("qwen_success_rate")
    control_rate = bucket.get("control_success_rate")
    required_qwen_rate = result.get("required_qwen_success_rate")
    required_control_rate = result.get("required_control_success_rate")
    if qwen_rate is None or float(qwen_rate) < min_success_rate:
      issues.append(f"{case_name}: qwen_success_rate {qwen_rate} < {min_success_rate}")
    if require_control_success and (control_rate is None or float(control_rate) < min_success_rate):
      issues.append(f"{case_name}: control_success_rate {control_rate} < {min_success_rate}")
    if required_qwen_rate is None or float(required_qwen_rate) < min_success_rate:
      issues.append(f"{case_name}: required_qwen_success_rate {required_qwen_rate} < {min_success_rate}")
    if require_control_success and (required_control_rate is None or float(required_control_rate) < min_success_rate):
      issues.append(f"{case_name}: required_control_success_rate {required_control_rate} < {min_success_rate}")
  elif case_name in NO_SLOW_LEAD_CLASSES:
    qwen_rate = bucket.get("qwen_success_rate")
    if qwen_rate is None or float(qwen_rate) < min_success_rate:
      issues.append(f"{case_name}: qwen_success_rate {qwen_rate} < {min_success_rate}")
    false_slow_rate = bucket.get("false_slow_rate")
    control_rate = bucket.get("control_success_rate")
    if false_slow_rate is None or float(false_slow_rate) > max_false_slow_rate:
      issues.append(f"{case_name}: false_slow_rate {false_slow_rate} > {max_false_slow_rate}")
    if require_control_success and (control_rate is None or float(control_rate) < 1.0 - max_false_slow_rate):
      issues.append(f"{case_name}: control_success_rate {control_rate} < {1.0 - max_false_slow_rate}")
  return not issues, issues


def evaluate_lead_suite(
  cases: dict[str, dict[str, Any]],
  *,
  horizon_m: float = 60.0,
  path_lateral_m: float = 1.35,
  min_vehicle_clearance_m: float = 2.0,
  max_allowed_age_frames: int | None = None,
  min_success_rate: float = 0.95,
  max_false_slow_rate: float = 0.05,
  require_control_success: bool = True,
  require_vehicle_heading_evidence: bool = True,
  required_cases: tuple[str, ...] = LEAD_SUITE_CLASSES,
) -> dict[str, Any]:
  issues: list[str] = []
  case_results: dict[str, Any] = {}
  for case_name in required_cases:
    episode = cases.get(case_name)
    if episode is None:
      issues.append(f"{case_name}: missing episode")
      continue
    result = evaluate_episode(
      episode,
      horizon_m=horizon_m,
      path_lateral_m=path_lateral_m,
      min_vehicle_clearance_m=min_vehicle_clearance_m,
      max_allowed_age_frames=max_allowed_age_frames,
    )
    ok, case_issues = _case_gate(
      case_name,
      result,
      min_success_rate=min_success_rate,
      max_false_slow_rate=max_false_slow_rate,
      require_control_success=require_control_success,
      require_vehicle_heading_evidence=require_vehicle_heading_evidence,
    )
    case_results[case_name] = {key: value for key, value in result.items() if key != "rows"}
    case_results[case_name]["ok"] = ok
    case_results[case_name]["issues"] = case_issues
    issues.extend(case_issues)
  return {
    "ok": not issues,
    "issues": issues,
    "required_cases": list(required_cases),
    "min_success_rate": min_success_rate,
    "max_false_slow_rate": max_false_slow_rate,
    "max_allowed_age_frames": max_allowed_age_frames,
    "min_vehicle_clearance_m": min_vehicle_clearance_m,
    "require_vehicle_heading_evidence": require_vehicle_heading_evidence,
    "cases": case_results,
  }


def _parse_case_spec(specs: list[str]) -> dict[str, Path]:
  result: dict[str, Path] = {}
  for spec in specs:
    if "=" not in spec:
      raise ValueError(f"case spec must be name=path: {spec}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if name not in LEAD_SUITE_CLASSES:
      raise ValueError(f"unknown lead suite case: {name}")
    result[name] = Path(raw_path.strip())
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description="Evaluate the lead/cut-in/crossing suite from MetaDrive RTP episode traces.")
  parser.add_argument("--artifacts-root", type=Path, default=REPO_ROOT / "artifacts" / "reasoned_trajectory_poc")
  parser.add_argument("--case", action="append", default=[], help="Explicit case mapping, for example slower_lead=path/to/run. Repeatable.")
  parser.add_argument("--mode", default="vlm")
  parser.add_argument("--horizon-m", type=float, default=60.0)
  parser.add_argument("--path-lateral-m", type=float, default=1.35)
  parser.add_argument("--min-vehicle-clearance-m", type=float, default=2.0)
  parser.add_argument("--max-allowed-age-frames", type=int, default=8)
  parser.add_argument("--min-success-rate", type=float, default=0.95)
  parser.add_argument("--max-false-slow-rate", type=float, default=0.05)
  parser.add_argument("--require-control-success", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--require-vehicle-heading-evidence", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  case_paths = discover_latest_case_paths(args.artifacts_root)
  case_paths.update(_parse_case_spec(args.case))
  cases: dict[str, dict[str, Any]] = {}
  resolved_paths: dict[str, str] = {}
  for case_name, path in sorted(case_paths.items()):
    episode_path = _episode_path(path, args.mode)
    cases[case_name] = json.loads(episode_path.read_text(encoding="utf-8"))
    resolved_paths[case_name] = str(episode_path)

  result = evaluate_lead_suite(
    cases,
    horizon_m=args.horizon_m,
    path_lateral_m=args.path_lateral_m,
    min_vehicle_clearance_m=args.min_vehicle_clearance_m,
    max_allowed_age_frames=args.max_allowed_age_frames,
    min_success_rate=args.min_success_rate,
    max_false_slow_rate=args.max_false_slow_rate,
    require_control_success=args.require_control_success,
    require_vehicle_heading_evidence=args.require_vehicle_heading_evidence,
  )
  result["episode_paths"] = resolved_paths
  out_path = args.out or (args.artifacts_root / "lead_suite_evaluation.json")
  out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
  print(json.dumps(result, indent=2))
  print(f"out={out_path}")
  if not result["ok"]:
    sys.exit(2)


if __name__ == "__main__":
  main()
