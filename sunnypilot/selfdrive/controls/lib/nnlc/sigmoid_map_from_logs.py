#!/usr/bin/env python3
"""Generate NNLC sigmoid map solutions from recorded rlog/qlog files."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cereal import car, custom  # noqa: E402
from opendbc.car.car_helpers import interfaces  # noqa: E402
from opendbc.car.vehicle_model import VehicleModel  # noqa: E402
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY  # noqa: E402
from openpilot.selfdrive.controls.lib.latcontrol_torque import LOW_SPEED_X, LOW_SPEED_Y  # noqa: E402
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.sigmoid_map_tuner import SigmoidMapTuner  # noqa: E402
from tools.lib.logreader import LogReader  # noqa: E402

LOG_PATTERNS = ("*.rlog", "*.rlog.*", "*.qlog", "*.qlog.*")


class OfflineLatControlAdapter:
  """Minimal LatControlTorque-like object for SigmoidMapTuner."""

  def __init__(self, CI, CP):
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.steer_max = float(getattr(CP, "steerLimitTorque", 1.0)) or 1.0

  def update_live_torque_params(self, latAccelFactor: float, latAccelOffset: float, friction: float) -> None:
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction


class OfflineSigmoidMapTuner(SigmoidMapTuner):
  """Sigmoid tuner that writes solutions to a user-provided directory."""

  def __init__(self, lac_adapter: OfflineLatControlAdapter, CI, CP, output_dir: Path):
    super().__init__(lac_adapter,
                     lac_adapter.torque_params,
                     CI.torque_from_lateral_accel_in_torque_space(),
                     CP.carFingerprint)
    self._storage_dirs = [str(output_dir)]
    self._file_name = f"{CP.carFingerprint}_nnlc.json"

  def _update_torque_params(self) -> None:  # offline mode never tweaks live params
    return


def _find_logs(path: Path) -> list[Path]:
  paths: list[Path] = []
  for pattern in LOG_PATTERNS:
    paths.extend(path.rglob(pattern))
  return sorted({p for p in paths if p.is_file()})


def _extract_car_params(log_paths: list[Path]) -> tuple[car.CarParams, custom.CarParamsSP]:
  cp_msg = None
  cp_sp_msg = None
  for msg in LogReader([str(p) for p in log_paths]):
    if msg.which() == "carParams" and cp_msg is None:
      cp_msg = msg.carParams
    elif msg.which() == "carParamsSP" and cp_sp_msg is None:
      cp_sp_msg = msg.carParamsSP
    if cp_msg and cp_sp_msg:
      break
  if cp_msg is None:
    raise RuntimeError("No carParams message found in provided logs")
  if cp_sp_msg is None:
    cp_sp_msg = custom.CarParamsSP.new_message()
  return cp_msg, cp_sp_msg


def _process_logs(log_paths: list[Path], output_dir: Path) -> Path:
  cp_msg, cp_sp_msg = _extract_car_params(log_paths)
  CI = interfaces[cp_msg.carFingerprint](cp_msg, cp_sp_msg)
  adapter = OfflineLatControlAdapter(CI, cp_msg)
  tuner = OfflineSigmoidMapTuner(adapter, CI, cp_msg, output_dir)
  vm = VehicleModel(cp_msg)

  lp_angle_offset = 0.0
  lp_roll = 0.0
  lp_sr = cp_msg.steerRatio
  lp_stiffness = cp_msg.tireStiffnessFactor
  last_car_state = None

  for msg in LogReader([str(p) for p in log_paths]):
    which = msg.which()
    if which == "carState":
      last_car_state = msg.carState
    elif which == "liveParameters":
      lp = msg.liveParameters
      if lp.steerRatio > 0.0 and lp.stiffnessFactor > 0.0:
        lp_sr = lp.steerRatio
        lp_stiffness = lp.stiffnessFactor
        vm.update_params(lp_stiffness, lp_sr)
      lp_angle_offset = lp.angleOffsetDeg
      lp_roll = lp.roll
    elif which == "controlsState":
      if last_car_state is None:
        continue
      if msg.controlsState.lateralControlState.which() != "torqueState":
        continue
      torque_state = msg.controlsState.lateralControlState.torqueState
      if not torque_state.active:
        continue
      desired_curvature = msg.controlsState.desiredCurvature
      steer_angle = math.radians(last_car_state.steeringAngleDeg - lp_angle_offset)
      actual_curvature = -vm.calc_curvature(steer_angle, last_car_state.vEgo, lp_roll)
      low_speed_factor = float(np.interp(last_car_state.vEgo, LOW_SPEED_X, LOW_SPEED_Y)) ** 2
      setpoint = torque_state.desiredLateralAccel + low_speed_factor * desired_curvature
      measurement = torque_state.actualLateralAccel + low_speed_factor * actual_curvature
      roll_comp = lp_roll * ACCELERATION_DUE_TO_GRAVITY
      output_torque = -float(torque_state.output)
      tuner.observe(True,
                    last_car_state,
                    setpoint,
                    measurement,
                    torque_state.desiredLateralAccel,
                    output_torque,
                    torque_state.saturated,
                    roll_comp)

  if tuner._solution is None:
    solution = tuner._solve()
    if solution is None:
      raise RuntimeError("Insufficient coverage to solve for a sigmoid map")
    serialized = solution.serialize()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / tuner._file_name
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(serialized, f, indent=2)
    return output_path

  return Path(tuner._file_path)


def main(argv: Sequence[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Generate NNLC sigmoid maps from recorded logs")
  parser.add_argument("log_dir", type=Path, help="Directory containing qlog/rlog files")
  parser.add_argument("--output", type=Path, default=Path("sigmoid_maps"), help="Directory to write the solution JSON")
  args = parser.parse_args(argv)

  log_paths = _find_logs(args.log_dir)
  if not log_paths:
    raise SystemExit(f"No qlog/rlog files found under {args.log_dir}")

  output_path = _process_logs(log_paths, args.output)
  print(f"Wrote sigmoid map to {output_path}")


if __name__ == "__main__":
  main()
