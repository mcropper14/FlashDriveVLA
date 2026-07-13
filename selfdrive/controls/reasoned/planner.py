from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

try:
  from openpilot.selfdrive.controls.reasoned.pathsynth import BasePlan, PathSynth, PathSynthResult
  from openpilot.selfdrive.controls.reasoned.rtp import DEFAULT_MAX_GENERATED_TOKENS, RtpProgram, RtpValidationError, parse_rtp
  from openpilot.selfdrive.controls.reasoned.scene_board import SceneBoard, SceneBoardRenderer
  from openpilot.selfdrive.controls.reasoned.vlm import RtpEngine, RtpEngineResult, VlmError, build_rtp_engine
except ModuleNotFoundError:
  from selfdrive.controls.reasoned.pathsynth import BasePlan, PathSynth, PathSynthResult
  from selfdrive.controls.reasoned.rtp import DEFAULT_MAX_GENERATED_TOKENS, RtpProgram, RtpValidationError, parse_rtp
  from selfdrive.controls.reasoned.scene_board import SceneBoard, SceneBoardRenderer
  from selfdrive.controls.reasoned.vlm import RtpEngine, RtpEngineResult, VlmError, build_rtp_engine


@dataclass(frozen=True)
class ReasonedPlannerConfig:
  deadline_ms: float = 50.0
  max_generated_tokens: int = DEFAULT_MAX_GENERATED_TOKENS
  allow_async_rtp: bool = False
  max_async_age_frames: int = 6


@dataclass
class ReasonedTimings:
  camera_to_scene_board_ms: float = 0.0
  scene_board_to_vlm_prefill_ms: float = 0.0
  vlm_decode_ms: float = 0.0
  rtp_parse_ms: float = 0.0
  path_synth_ms: float = 0.0
  publish_age_ms: float = 0.0

  @property
  def total_ms(self) -> float:
    return (
      self.camera_to_scene_board_ms +
      self.scene_board_to_vlm_prefill_ms +
      self.vlm_decode_ms +
      self.rtp_parse_ms +
      self.path_synth_ms
    )


@dataclass
class ReasonedStepResult:
  frame_id: int
  model_log_mono_time_ns: int
  valid: bool
  should_publish: bool
  same_frame: bool
  deadline_met: bool
  rtp_text: str
  generated_token_count: int
  vlm_backend: str
  rtp_source_frame_id: int | None
  rtp_age_frames: int | None
  labels: tuple[str, ...]
  label_scores: dict[str, float]
  raw_labels: tuple[str, ...]
  raw_label_scores: dict[str, float]
  labels_scored_this_request: tuple[str, ...]
  score_group_index: int | None
  label_state_debug: dict | None
  choice: dict | None
  timings: ReasonedTimings
  program: Optional[RtpProgram]
  synth: Optional[PathSynthResult]
  board: Optional[SceneBoard]
  invalid_reason: str = ""


class ReasonedPlanner:
  def __init__(
    self,
    config: Optional[ReasonedPlannerConfig] = None,
    renderer: Optional[SceneBoardRenderer] = None,
    engine: Optional[RtpEngine] = None,
    path_synth: Optional[PathSynth] = None,
  ):
    self.config = config or ReasonedPlannerConfig()
    self.renderer = renderer or SceneBoardRenderer()
    self.engine = engine or build_rtp_engine()
    self.path_synth = path_synth or PathSynth()
    self.deadline_miss_count = 0
    self.invalid_rtp_count = 0

  def step(self, base_plan: BasePlan, vehicle_state: Optional[dict[str, float]] = None) -> ReasonedStepResult:
    start = time.perf_counter()
    timings = ReasonedTimings()
    state = vehicle_state or {}

    board_start = time.perf_counter()
    board = self.renderer.render(base_plan, state)
    timings.camera_to_scene_board_ms = _elapsed_ms(board_start)

    rtp_result: Optional[RtpEngineResult] = None
    program: Optional[RtpProgram] = None
    synth: Optional[PathSynthResult] = None
    invalid_reason = ""

    try:
      remaining_ms = max(1.0, self.config.deadline_ms - _elapsed_ms(start))
      rtp_result = self.engine.generate(base_plan.frame_id, board, state, remaining_ms)
      timings.scene_board_to_vlm_prefill_ms = rtp_result.prefill_ms
      timings.vlm_decode_ms = rtp_result.decode_ms

      parse_start = time.perf_counter()
      program = parse_rtp(rtp_result.text, self.config.max_generated_tokens)
      timings.rtp_parse_ms = _elapsed_ms(parse_start)

      synth_start = time.perf_counter()
      synth = self.path_synth.compile(base_plan, program)
      timings.path_synth_ms = _elapsed_ms(synth_start)
    except RtpValidationError as exc:
      self.invalid_rtp_count += 1
      invalid_reason = str(exc)
    except VlmError as exc:
      self.invalid_rtp_count += 1
      invalid_reason = str(exc)

    total_ms = _elapsed_ms(start)
    timings.publish_age_ms = total_ms
    deadline_met = total_ms <= self.config.deadline_ms
    if not deadline_met:
      self.deadline_miss_count += 1

    valid = program is not None and synth is not None and synth.valid
    rtp_source_frame_id = rtp_result.source_frame_id if rtp_result is not None else None
    rtp_age_frames = None if rtp_source_frame_id is None else base_plan.frame_id - rtp_source_frame_id
    same_frame_rtp = rtp_age_frames == 0
    async_fresh = (
      self.config.allow_async_rtp and
      rtp_age_frames is not None and
      0 <= rtp_age_frames <= self.config.max_async_age_frames
    )
    rtp_fresh_enough = same_frame_rtp or async_fresh
    should_publish = valid and deadline_met and synth.frame_id == base_plan.frame_id and rtp_fresh_enough
    if valid and not should_publish and not invalid_reason:
      if not deadline_met:
        invalid_reason = "deadline_miss"
      elif not rtp_fresh_enough:
        invalid_reason = f"rtp_stale_or_not_same_frame age_frames={rtp_age_frames}"
      else:
        invalid_reason = "frame_mismatch"
    if synth is not None and not synth.valid and not invalid_reason:
      invalid_reason = synth.invalid_reason

    return ReasonedStepResult(
      frame_id=base_plan.frame_id,
      model_log_mono_time_ns=base_plan.model_log_mono_time_ns,
      valid=valid,
      should_publish=should_publish,
      same_frame=(synth is not None and synth.frame_id == base_plan.frame_id and same_frame_rtp),
      deadline_met=deadline_met,
      rtp_text=rtp_result.text if rtp_result is not None else "",
      generated_token_count=rtp_result.generated_token_count if rtp_result is not None else 0,
      vlm_backend=rtp_result.backend if rtp_result is not None else getattr(self.engine, "backend", "unknown"),
      rtp_source_frame_id=rtp_source_frame_id,
      rtp_age_frames=rtp_age_frames,
      labels=rtp_result.labels if rtp_result is not None else (),
      label_scores=rtp_result.label_scores if rtp_result is not None and rtp_result.label_scores is not None else {},
      raw_labels=rtp_result.raw_labels if rtp_result is not None else (),
      raw_label_scores=rtp_result.raw_label_scores if rtp_result is not None and rtp_result.raw_label_scores is not None else {},
      labels_scored_this_request=rtp_result.labels_scored_this_request if rtp_result is not None else (),
      score_group_index=rtp_result.score_group_index if rtp_result is not None else None,
      label_state_debug=rtp_result.label_state_debug if rtp_result is not None else None,
      choice=rtp_result.choice if rtp_result is not None else None,
      timings=timings,
      program=program,
      synth=synth,
      board=board,
      invalid_reason=invalid_reason,
    )


def _elapsed_ms(start: float) -> float:
  return (time.perf_counter() - start) * 1000.0
