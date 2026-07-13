"""
Copyright (c) 2021-, JAGOFF NOT SUNNY

This file  is licensed under the MIT License.

"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from opendbc.sunnypilot.car.interfaces import LatControlInputs
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.hw import Paths


MPH_TO_MS = 0.44704
MAX_TUNING_SPEED = 70.0 * MPH_TO_MS
MIN_TUNING_SPEED = 5.0 * MPH_TO_MS  # 5 mph
EMA_ALPHA = 0.2
DEFAULT_MIN_BIN_SAMPLES = 20
MIN_READY_SPEED_SLICES = 5
HIGH_LAT_ACCEL_THRESHOLD = 2.0  # m/s^2
LOW_LAT_ACCEL_THRESHOLD = 0.4
MAX_SLICE_MSE = 0.35
LOGISTIC_EPS = 1e-4
FRICTION_SAMPLE_THRESHOLD = 0.3
LAT_ACCEL_SEARCH_RANGE = 6.0


@dataclass
class BinValue:
  lat_accel: float = 0.0
  torque: float = 0.0
  torque_space: float = 0.0
  count: int = 0


@dataclass
class SigmoidSlice:
  speed: float
  slope: float
  intercept: float


class SigmoidMapSolution:
  def __init__(self, slices: List[SigmoidSlice]):
    self.slices = sorted(slices, key=lambda s: s.speed)

  def serialize(self) -> Dict[str, List[Dict[str, float]]]:
    return {
      "slices": [
        {"speed": s.speed, "slope": s.slope, "intercept": s.intercept}
        for s in self.slices
      ],
    }

  @staticmethod
  def deserialize(data: Dict[str, List[Dict[str, float]]]) -> "SigmoidMapSolution":
    slices = [SigmoidSlice(speed=s["speed"], slope=s["slope"], intercept=s["intercept"]) for s in data.get("slices", [])]
    return SigmoidMapSolution(slices)

  def params_at_speed(self, speed: float) -> Optional[Tuple[float, float]]:
    if not self.slices:
      return None

    if speed <= self.slices[0].speed:
      return self.slices[0].slope, self.slices[0].intercept
    if speed >= self.slices[-1].speed:
      return self.slices[-1].slope, self.slices[-1].intercept

    for low, high in zip(self.slices[:-1], self.slices[1:]):
      if low.speed <= speed <= high.speed:
        ratio = 0.0 if high.speed == low.speed else (speed - low.speed) / (high.speed - low.speed)
        slope = low.slope + ratio * (high.slope - low.slope)
        intercept = low.intercept + ratio * (high.intercept - low.intercept)
        return slope, intercept

    return None


class SigmoidMapTuner:
  def __init__(self,
               lac_torque,
               torque_params,
               torque_from_lateral_accel_in_torque_space: Callable,
               car_fingerprint: str,
               freeze_when_solution_loaded: bool = True,
               min_bin_samples: int = DEFAULT_MIN_BIN_SAMPLES):
    self.lac_torque = lac_torque
    self.torque_params = torque_params
    self._torque_from_lateral_accel_in_torque_space = torque_from_lateral_accel_in_torque_space

    self._base_torque_from_lataccel = lac_torque.torque_from_lateral_accel
    self._base_lataccel_from_torque = lac_torque.lateral_accel_from_torque
    self._freeze_when_solution_loaded = freeze_when_solution_loaded
    self._min_bin_samples = max(1, min_bin_samples)

    self._storage_dirs = self._candidate_storage_dirs()
    self._file_name = f"{car_fingerprint}_nnlc.json"
    self._file_path = os.path.join(self._storage_dirs[0], self._file_name)
    self._current_speed = 0.0
    self._solution: Optional[SigmoidMapSolution] = None
    self._current_roll_compensation = 0.0
    self._current_longitudinal_accel = 0.0
    self._frozen_from_disk = False

    mph_edges = np.arange(5.0, 75.0, 5.0) * MPH_TO_MS
    self._speed_edges = mph_edges
    self._speed_centers = 0.5 * (self._speed_edges[1:] + self._speed_edges[:-1])
    self._lat_edges = np.linspace(-4.0, 4.0, 33)  # 0.25 m/s^2 bins

    self._bins: Dict[Tuple[int, int], BinValue] = {}
    self._install_wrappers()
    self._load_solution()

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------
  def observe(self,
              enabled: bool,
              CS,
              setpoint: float,
              measurement: float,
              desired_lat_accel: float,
              output_torque: float,
              steer_limited: bool,
              roll_compensation: float) -> None:
    self._current_speed = CS.vEgo
    self._current_longitudinal_accel = CS.aEgo
    self._current_roll_compensation = roll_compensation

    if self._frozen_from_disk:
      return

    if not enabled:
      return

    if steer_limited or CS.steeringPressed:
      return

    if CS.vEgo < MIN_TUNING_SPEED or CS.vEgo > MAX_TUNING_SPEED:
      return

    lat_error = abs(setpoint - measurement)
    if lat_error > 0.5:
      return

    if abs(desired_lat_accel) < LOW_LAT_ACCEL_THRESHOLD:
      return

    idx_speed = self._speed_index(CS.vEgo)
    idx_lat = self._lat_index(desired_lat_accel)
    if idx_speed is None or idx_lat is None:
      return

    lat_inputs = LatControlInputs(desired_lat_accel, roll_compensation, CS.vEgo, CS.aEgo)
    torque_space = float(self._torque_from_lateral_accel_in_torque_space(lat_inputs, self.torque_params, gravity_adjusted=True))

    key = (idx_speed, idx_lat)
    bin_value = self._bins.setdefault(key, BinValue())
    bin_value.count += 1
    bin_value.lat_accel = self._ema(bin_value.lat_accel, desired_lat_accel)
    bin_value.torque = self._ema(bin_value.torque, float(np.clip(output_torque, -self.lac_torque.steer_max, self.lac_torque.steer_max)))
    bin_value.torque_space = self._ema(bin_value.torque_space, torque_space)

    if self._ready_to_solve():
      solution = self._solve()
      if solution is not None:
        self._apply_solution(solution)

  # ---------------------------------------------------------------------------
  # Internal helpers
  # ---------------------------------------------------------------------------
  def _install_wrappers(self) -> None:
    def torque_from_lataccel(lat_accel, torque_params):
      params = self._solution.params_at_speed(self._current_speed) if self._solution is not None else None
      if params is None or self._current_speed < MIN_TUNING_SPEED:
        return self._base_torque_from_lataccel(lat_accel, torque_params)

      slope, intercept = params
      lat_inputs = LatControlInputs(lat_accel,
                                    self._current_roll_compensation,
                                    self._current_speed,
                                    self._current_longitudinal_accel)
      torque_space = float(self._torque_from_lateral_accel_in_torque_space(lat_inputs, torque_params, gravity_adjusted=True))
      torque = self._predict_torque_from_space(torque_space, slope, intercept)
      return float(np.clip(torque, -self.lac_torque.steer_max, self.lac_torque.steer_max))

    def lataccel_from_torque(torque, torque_params):
      params = self._solution.params_at_speed(self._current_speed) if self._solution is not None else None
      if params is None or self._current_speed < MIN_TUNING_SPEED:
        return self._base_lataccel_from_torque(torque, torque_params)

      slope, intercept = params
      torque_space = self._torque_space_from_torque(torque, slope, intercept)
      if torque_space is None:
        return self._base_lataccel_from_torque(torque, torque_params)
      return float(self._latacc_from_torque_space(torque_space))

    self.lac_torque.torque_from_lateral_accel = torque_from_lataccel
    self.lac_torque.lateral_accel_from_torque = lataccel_from_torque

  def _ema(self, prev: float, new: float) -> float:
    if math.isclose(prev, 0.0, abs_tol=1e-6):
      return new
    return EMA_ALPHA * new + (1.0 - EMA_ALPHA) * prev

  def _speed_index(self, speed: float) -> Optional[int]:
    idx = np.digitize(speed, self._speed_edges) - 1
    if 0 <= idx < len(self._speed_centers):
      return idx
    return None

  def _lat_index(self, lat_accel: float) -> Optional[int]:
    idx = np.digitize(lat_accel, self._lat_edges) - 1
    if 0 <= idx < len(self._lat_edges) - 1:
      return idx
    return None

  def _ready_to_solve(self) -> bool:
    ready_slices = 0
    for speed_idx in range(len(self._speed_centers)):
      if self._slice_has_coverage(speed_idx):
        ready_slices += 1
    if not self._frozen_from_disk:
      cloudlog.event("sigmoid_map_tuner_ready",
                     ready_slices=ready_slices,
                     required_slices=MIN_READY_SPEED_SLICES,
                     bin_count=len(self._bins))
    return ready_slices >= MIN_READY_SPEED_SLICES

  def _slice_has_coverage(self, speed_idx: int) -> bool:
    pos_high = False
    neg_high = False
    mid = False
    for (idx_speed, idx_lat), value in self._bins.items():
      if idx_speed != speed_idx or value.count < self._min_bin_samples:
        continue
      lat = value.lat_accel
      if abs(lat) >= HIGH_LAT_ACCEL_THRESHOLD:
        if lat > 0:
          pos_high = True
        else:
          neg_high = True
      if abs(lat) <= HIGH_LAT_ACCEL_THRESHOLD and abs(lat) >= LOW_LAT_ACCEL_THRESHOLD:
        mid = True
    return pos_high and neg_high and mid

  def _solve(self) -> Optional[SigmoidMapSolution]:
    slices: List[SigmoidSlice] = []
    for speed_idx, speed_center in enumerate(self._speed_centers):
      slice_bins = [value for (idx_speed, _), value in self._bins.items() if idx_speed == speed_idx and value.count >= self._min_bin_samples]
      if len(slice_bins) < 3:
        continue

      lats = np.array([b.lat_accel for b in slice_bins])
      torque_space = np.array([b.torque_space for b in slice_bins])
      torques = np.array([b.torque for b in slice_bins])
      if np.all(lats <= 0.0) or np.all(lats >= 0.0):
        continue

      slope, intercept = self._fit_logistic(torque_space, torques)
      if slope is None:
        continue

      slices.append(SigmoidSlice(speed=speed_center, slope=slope, intercept=intercept))

    if len(slices) < MIN_READY_SPEED_SLICES:
      return None

    return SigmoidMapSolution(slices)

  def _fit_logistic(self, torque_space: np.ndarray, torques: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    steer_max = self.lac_torque.steer_max
    torque_norm = np.clip(0.5 * (torques / steer_max + 1.0), LOGISTIC_EPS, 1.0 - LOGISTIC_EPS)
    logits = np.log(torque_norm / (1.0 - torque_norm))
    A = np.vstack([torque_space, np.ones_like(torque_space)]).T
    try:
      slope, intercept = np.linalg.lstsq(A, logits, rcond=None)[0]
    except np.linalg.LinAlgError:
      return None, None

    pred = self._predict_torque_from_space(torque_space, slope, intercept)
    mse = float(np.mean((pred - torques) ** 2))
    if mse > MAX_SLICE_MSE:
      return None, None

    return slope, intercept

  def _predict_torque_from_space(self, torque_space, slope: float, intercept: float):
    logits = slope * torque_space + intercept
    torque_norm = 1.0 / (1.0 + np.exp(-logits))
    return (2.0 * torque_norm - 1.0) * self.lac_torque.steer_max

  def _torque_space_from_torque(self, torque, slope: float, intercept: float) -> Optional[float]:
    steer_max = self.lac_torque.steer_max
    torque_norm = 0.5 * (np.clip(torque, -steer_max, steer_max) / steer_max + 1.0)
    torque_norm = np.clip(torque_norm, LOGISTIC_EPS, 1.0 - LOGISTIC_EPS)
    logits = np.log(torque_norm / (1.0 - torque_norm))
    if abs(slope) < 1e-4:
      return None
    return (logits - intercept) / slope

  def _latacc_from_torque_space(self, torque_space: float) -> float:
    low = -LAT_ACCEL_SEARCH_RANGE
    high = LAT_ACCEL_SEARCH_RANGE
    for _ in range(16):
      mid = 0.5 * (low + high)
      lat_inputs = LatControlInputs(mid,
                                    self._current_roll_compensation,
                                    self._current_speed,
                                    self._current_longitudinal_accel)
      mid_torque = float(self._torque_from_lateral_accel_in_torque_space(lat_inputs, self.torque_params, gravity_adjusted=True))
      if torque_space >= mid_torque:
        low = mid
      else:
        high = mid
    return 0.5 * (low + high)

  def _apply_solution(self, solution: SigmoidMapSolution) -> None:
    self._solution = solution
    self._write_solution()
    self._update_torque_params()

  def _update_torque_params(self) -> None:
    if self._solution is None or not self._solution.slices:
      return

    lat_accel_factor = self._estimate_latacc_factor()
    offset_samples = [value.lat_accel for value in self._bins.values()
                      if value.count >= self._min_bin_samples and abs(value.torque) < 0.1]
    friction_samples = [abs(value.torque) for value in self._bins.values()
                        if value.count >= self._min_bin_samples and abs(value.lat_accel) < FRICTION_SAMPLE_THRESHOLD]

    lat_accel_offset = float(np.clip(np.median(offset_samples), -1.0, 1.0)) if offset_samples else self.torque_params.latAccelOffset
    friction = float(np.clip(np.median(friction_samples), 0.05, 3.0)) if friction_samples else self.torque_params.friction

    if lat_accel_factor is None:
      lat_accel_factor = self.torque_params.latAccelFactor

    self.lac_torque.update_live_torque_params(lat_accel_factor, lat_accel_offset, friction)

  def _estimate_latacc_factor(self) -> Optional[float]:
    slopes = []
    for value in self._bins.values():
      if value.count < self._min_bin_samples:
        continue
      if abs(value.torque) < 1e-3:
        continue
      slopes.append(value.lat_accel / value.torque)
    if not slopes:
      return None
    return float(np.clip(np.median(slopes), 0.01, 10.0))

  def _write_solution(self) -> None:
    if self._solution is None or self._frozen_from_disk:
      return

    serialized = self._solution.serialize()
    for directory in self._storage_dirs:
      path = os.path.join(directory, self._file_name)
      try:
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
          json.dump(serialized, f)
        self._file_path = path
        return
      except OSError as e:
        cloudlog.event("sigmoid_map_tuner_write_failed", path=path, error=str(e))

  def _load_solution(self) -> None:
    for directory in self._storage_dirs:
      path = os.path.join(directory, self._file_name)
      if not os.path.exists(path):
        continue
      try:
        with open(path, "r", encoding="utf-8") as f:
          data = json.load(f)
        solution = SigmoidMapSolution.deserialize(data)
        if solution.slices:
          self._solution = solution
          self._file_path = path
          if self._freeze_when_solution_loaded:
            self._frozen_from_disk = True
          self._update_torque_params()
        return
      except (OSError, json.JSONDecodeError):
        continue

  def _candidate_storage_dirs(self) -> List[str]:
    return ["/data/params/d_tmp/sigmoid_maps"]

