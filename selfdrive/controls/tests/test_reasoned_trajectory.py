import base64
import json
import unittest
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from selfdrive.controls.reasoned.pathsynth import BasePlan, PathSynth
from selfdrive.controls.reasoned.planner import ReasonedPlanner, ReasonedPlannerConfig
from selfdrive.controls.reasoned.rtp import RtpValidationError, parse_rtp
from selfdrive.controls.reasoned.scene_board import SceneBoard
from selfdrive.controls.reasoned import side_semantics
from selfdrive.controls.reasoned import lead_semantics
from selfdrive.controls.reasoned.ui_scene_board import OverlayGeometry, UiSceneBoardRenderer
from selfdrive.controls.reasoned.vlm import AsyncRtpEngine, PersistentRtpEngine, RtpEngine, RtpEngineResult
from selfdrive.controls.reasoned import vlm as vlm_module
from argparse import Namespace

from tools.reasoned_trajectory_poc.qwen_label_rtp_worker import (
  CONSTRUCTION_ACTION_BOOTSTRAP_MAX_TRACKED_OFFSET_M,
  CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M,
  CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_MARGIN,
  CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_SCORE,
  CONSTRUCTION_ACTION_IMMEDIATE_MARGIN,
  CONSTRUCTION_ACTION_IMMEDIATE_SCORE,
  CONSTRUCTION_COMMITTED_CONFLICT_REQUIRES_ACTION,
  CONSTRUCTION_CLEAR_RTP_CONFIDENCE,
  CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES,
  CONSTRUCTION_DIRECT_EDGE_MARGIN,
  CONSTRUCTION_DIRECT_EDGE_SCORE,
  CONSTRUCTION_DIRECT_SEMANTIC_MARGIN,
  CONSTRUCTION_DIRECT_SEMANTIC_SCORE,
  CONSTRUCTION_EDGE_BOOTSTRAP_MARGIN,
  CONSTRUCTION_EDGE_BOOTSTRAP_OPPOSITE_MAX,
  CONSTRUCTION_EDGE_BOOTSTRAP_SCORE,
  CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN,
  CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE,
  CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN,
  CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE,
  CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE,
  CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_MARGIN,
  CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE,
  CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN,
  CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE,
  CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_MARGIN,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_AGE_FRAMES,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_TRACKED_OFFSET_M,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_OPPOSITE_MAX,
  CONSTRUCTION_SIDE_EARLY_REVERSAL_SCORE,
  CONSTRUCTION_SIDE_TOWARD_PATH_OVERRIDE_SCORE,
  CONSTRUCTION_STATE_MACHINE_VERSION,
  DEFAULT_SCORE_LABEL_GROUPS,
  RotatingScoreState,
  SCORE_PROMPT,
  SCORE_QUESTIONS,
  _labels_to_rtp,
  _with_visual_fallbacks,
)
from tools.reasoned_trajectory_poc.qwen_trt_label_engine import (
  CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN,
  _apply_construction_mirror_consistency,
  _apply_construction_mirror_fusion,
  _apply_candidate_relative_choice,
  _apply_lead_state_consistency,
  _args_for_score_group,
  _construction_text_precision,
  _choice_group_spec,
  _choice_scores_from_word_scores,
  _choice_text,
  _choice_token_id_variants,
  _generic_text_engine_path,
  _image_from_payload_for_labels,
  _torch_attn_implementation,
  _score_prompt_mode,
  _score_text,
  _mirror_construction_labels_to_original,
  _runtime_contract,
  _score_calibrated_construction_confidence,
  _resolve_candidate_obstruction_scores,
  _select_text_output,
  _set_qwen_attention_implementation,
  _text_output_tensor_name,
  _text_position_dtype,
  _text_precision,
  _vehicle_state_for_labels,
  _with_rtp_confidence,
)
from tools.reasoned_trajectory_poc.evaluate_construction_trace import (
  consumed_construction_side,
  construction_requirement_from_record,
  evaluate_episode,
  lateral_command_side,
  planned_lateral_command_side,
  qwen_construction_side,
)
from tools.reasoned_trajectory_poc.evaluate_lead_trace import (
  evaluate_episode as evaluate_lead_episode,
  lead_requirement_from_record,
  qwen_lead_class,
)
from tools.reasoned_trajectory_poc.evaluate_pedestrian_trace import (
  evaluate_episode as evaluate_pedestrian_episode,
  pedestrian_requirement_from_record,
  qwen_pedestrian_class,
)
from tools.reasoned_trajectory_poc.evaluate_signal_trace import (
  evaluate_episode as evaluate_signal_episode,
  signal_requirement_from_record,
)
from tools.reasoned_trajectory_poc.evaluate_lead_suite import evaluate_lead_suite
from tools.reasoned_trajectory_poc.run_metadrive_overlay_demo import (
  CONSTRUCTION_SCENES,
  DurableAvoidance,
  DurableLateralOverrideState,
  DurableSpeedPlan,
  ROUTE_VEHICLE_MODEL_CLASS,
  ROUTE_VEHICLE_MODEL_CLASSES,
  ROUTE_VEHICLE_USE_SPECIAL_COLOR,
  ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD,
  VEHICLE_SCENES,
  _apply_current_lead_state_guard,
  _apply_current_visual_signal_guard,
  _convert_camera_frame_color,
  _adjust_signal_speed_plan,
  _expected_lead_class_from_spawned,
  _physical_lead_clear_reason,
  _should_apply_visual_signal_guard,
  _visual_traffic_signal_label_from_frame,
  compose_lateral_offset,
  compose_lateral_offset_after_publish,
  construction_avoidance_side_valid,
  construction_hazard_side_from_token,
  construction_scene_side,
  _merge_durable_speed_plan,
  durable_avoidance_sign_valid,
  durable_avoidance_sign_valid_for_args,
  durable_avoidance_from_program,
  lateral_side_metadrive,
  lateral_side_openpilot,
  durable_speed_plan_from_program,
  metadrive_to_openpilot_lateral_m,
  nearest_route_vehicle_state,
  openpilot_to_route_lateral_m_from_args,
  openpilot_to_metadrive_lateral_m,
  route_lateral_for_side_from_args,
  route_to_openpilot_lateral_m_from_args,
  route_vehicle_class_from_name,
  route_vehicle_heading_alignment_cos,
  route_vehicle_heading_error_rad,
  route_vehicle_heading_same_direction,
  route_vehicle_visual_heading,
  scene_board_renderer_for_args,
  spawned_route_clearance_m,
  spawned_route_proximity,
  selected_lateral_offset_m,
  update_durable_lateral_plans,
  update_durable_speed_plans,
)


SAMPLE_RTP = """RTPv1
scene=construction_merge
evidence=[cones_right_s22_45,lead_s18_braking,lane_left_open]
meta=BIAS_LEFT_AND_SLOW
branch=base
lat_bias_m=1.25
speed_cap_mps=2.5
stop_s=none
avoid=[right_edge_s8_48_margin1.25]
weights=[obs2.2,lane1.4,comfort1.0,base0.7,vlm1.0]
confidence=0.72"""


def _choice_word_scores(**overrides):
  words = (
    "red",
    "green",
    "go",
    "top",
    "bottom",
    "clear",
    "stop",
    "left",
    "right",
    "blocked",
    "present",
    "absent",
    "entering",
    "moving",
    "slower",
    "braking",
    "stopped",
    "merge",
    "crossing",
    "irrelevant",
    "blue",
    "purple",
    "orange",
    "cyan",
    "pink",
    "A",
    "B",
    "C",
    "none",
  )
  scores = {word: -1.0 for word in words}
  scores.update(overrides)
  return scores


class _FakeChoiceTokenizer:
  def __call__(self, text, add_special_tokens=False):
    token_map = {
      "left": [101],
      " left": [201],
      "\nleft": [301],
      "right": [102],
      " right": [202],
      "\nright": [302],
    }
    return Namespace(input_ids=token_map.get(text, [999, 1000]))


class _FakeChoiceProcessor:
  tokenizer = _FakeChoiceTokenizer()


class _FakeStdin:
  def __init__(self):
    self.writes = []

  def write(self, value):
    self.writes.append(value)

  def flush(self):
    pass


class _FakeStdout:
  def __init__(self, lines):
    self.lines = list(lines)

  def readline(self):
    if not self.lines:
      return ""
    return self.lines.pop(0)


class _FakeProc:
  def __init__(self, lines):
    self.stdin = _FakeStdin()
    self.stdout = _FakeStdout(lines)
    self.returncode = None

  def poll(self):
    return None


class TestRtpParser(unittest.TestCase):
  def test_accepts_bounded_rtp(self):
    program = parse_rtp(SAMPLE_RTP)
    self.assertEqual(program.scene, "construction_merge")
    self.assertEqual(program.meta, "BIAS_LEFT_AND_SLOW")
    self.assertEqual(program.branch, "base")
    self.assertAlmostEqual(program.lat_bias_m, 1.25)
    self.assertAlmostEqual(program.speed_cap_mps, 2.5)
    self.assertIn("cones_right_s22_45", program.evidence)

  def test_rejects_prose(self):
    with self.assertRaises(RtpValidationError):
      parse_rtp("I think the car should move left because cones are visible.")

  def test_rejects_out_of_bounds_bias(self):
    with self.assertRaises(RtpValidationError):
      parse_rtp(SAMPLE_RTP.replace("lat_bias_m=1.25", "lat_bias_m=4.0"))

  def test_accepts_percent_speed_cap(self):
    program = parse_rtp(SAMPLE_RTP.replace("speed_cap_mps=2.5", "speed_cap_mps=25%"))
    self.assertIsNone(program.speed_cap_mps)
    self.assertAlmostEqual(program.speed_scale, 0.25)
    self.assertIn("speed_cap_mps=25%", program.to_wire_text())


class TestPersistentRtpEngine(unittest.TestCase):
  def test_ready_jsonl_command_waits_before_first_generate(self):
    response = {
      "source_frame_id": 8,
      "rtp_text": SAMPLE_RTP,
      "backend": "fake_qwen",
    }
    fake_proc = _FakeProc((
      json.dumps({"ready": True}) + "\n",
      json.dumps(response) + "\n",
    ))
    original_popen = vlm_module.subprocess.Popen
    try:
      vlm_module.subprocess.Popen = lambda *args, **kwargs: fake_proc
      engine = PersistentRtpEngine("fake --ready-jsonl serve")
    finally:
      vlm_module.subprocess.Popen = original_popen

    board = SceneBoard(1, 1, bytearray((0, 0, 0)), "lead present no")
    result = engine.generate(8, board, {}, 50.0)

    self.assertEqual(result.source_frame_id, 8)
    self.assertEqual(result.text, SAMPLE_RTP)
    self.assertEqual(fake_proc.stdout.lines, [])

  def test_ready_marker_does_not_shift_frame_response(self):
    response = {
      "frame_id": 999,
      "source_frame_id": 7,
      "rtp_text": SAMPLE_RTP,
      "generated_token_count": 12,
      "prefill_ms": 1.5,
      "decode_ms": 2.5,
      "backend": "fake_qwen",
      "labels": ["construction_right"],
      "label_scores": {"construction_right": 1.25},
    }
    engine = PersistentRtpEngine.__new__(PersistentRtpEngine)
    engine.proc = _FakeProc((
      json.dumps({"ready": True}) + "\n",
      json.dumps(response) + "\n",
    ))
    board = SceneBoard(1, 1, bytearray((0, 0, 0)), "lead present no")

    result = engine.generate(7, board, {"v_ego": 2.5}, 50.0)

    self.assertEqual(result.text, SAMPLE_RTP)
    self.assertEqual(result.source_frame_id, 7)
    self.assertEqual(result.backend, "fake_qwen")
    self.assertEqual(result.labels, ("construction_right",))
    self.assertEqual(result.label_scores["construction_right"], 1.25)
    payload = json.loads(engine.proc.stdin.writes[0])
    self.assertEqual(payload["frame_id"], 7)

  def test_rotating_cached_scores_are_logged_with_published_labels(self):
    response = {
      "frame_id": 12,
      "source_frame_id": 10,
      "rtp_text": SAMPLE_RTP,
      "backend": "fake_qwen_rotating",
      "labels": ["cones", "construction_purple_edge"],
      "label_scores": {"construction_purple_edge": 1.5},
      "label_scores_cached": {"cones": 3.8, "construction_purple_edge": 1.5, "construction_blue_edge": -0.4},
      "labels_current_group": ["construction_purple_edge"],
      "labels_scored_this_request": ["construction_blue_edge", "construction_purple_edge"],
      "score_group_index": 4,
      "label_state_debug": {
        "construction_locked_side": "right",
        "construction_pending_side": None,
      },
    }
    engine = PersistentRtpEngine.__new__(PersistentRtpEngine)
    engine.proc = _FakeProc((json.dumps(response) + "\n",))
    board = SceneBoard(1, 1, bytearray((0, 0, 0)), "lead present no")

    result = engine.generate(12, board, {"v_ego": 2.5}, 50.0)

    self.assertEqual(result.labels, ("cones", "construction_purple_edge"))
    self.assertEqual(result.label_scores["cones"], 3.8)
    self.assertEqual(result.label_scores["construction_purple_edge"], 1.5)
    self.assertEqual(result.label_scores["construction_blue_edge"], -0.4)
    self.assertEqual(result.raw_labels, ("construction_purple_edge",))
    self.assertEqual(result.raw_label_scores["construction_purple_edge"], 1.5)
    self.assertEqual(result.labels_scored_this_request, ("construction_blue_edge", "construction_purple_edge"))
    self.assertEqual(result.score_group_index, 4)
    self.assertEqual(result.label_state_debug["construction_locked_side"], "right")

  def test_aux_candidate_images_are_sent_to_persistent_worker(self):
    response = {
      "frame_id": 3,
      "source_frame_id": 3,
      "rtp_text": SAMPLE_RTP,
      "backend": "fake_qwen",
    }
    engine = PersistentRtpEngine.__new__(PersistentRtpEngine)
    engine.proc = _FakeProc((json.dumps(response) + "\n",))
    board = SceneBoard(
      1,
      1,
      bytearray((0, 0, 0)),
      "lead present no",
      aux_pngs={"candidate_left": b"left-png", "candidate_right": b"right-png"},
    )

    engine.generate(3, board, {"v_ego": 2.5}, 50.0)

    payload = json.loads(engine.proc.stdin.writes[0])
    self.assertEqual(base64.b64decode(payload["scene_board_aux_images_b64"]["candidate_left"]), b"left-png")
    self.assertEqual(base64.b64decode(payload["scene_board_aux_images_b64"]["candidate_right"]), b"right-png")

  def test_reset_runtime_state_sends_control_message_to_persistent_worker(self):
    engine = PersistentRtpEngine.__new__(PersistentRtpEngine)
    engine.proc = _FakeProc((json.dumps({"ok": True, "control": "reset_runtime_state"}) + "\n",))

    engine.reset_runtime_state()

    payload = json.loads(engine.proc.stdin.writes[0])
    self.assertEqual(payload, {"control": "reset_runtime_state"})


class TestPathSynth(unittest.TestCase):
  def test_bias_changes_path_and_speed_only_shrinks(self):
    program = parse_rtp(SAMPLE_RTP)
    base = BasePlan(
      frame_id=42,
      model_log_mono_time_ns=1_000_000,
      t=tuple(i * 0.2 for i in range(17)),
      x=tuple(i * 5.0 for i in range(17)),
      y=tuple(0.0 for _ in range(17)),
      speeds=tuple(15.0 for _ in range(17)),
      desired_curvature=0.0,
      v_ego=15.0,
    )
    result = PathSynth().compile(base, program)
    self.assertTrue(result.valid)
    self.assertEqual(result.frame_id, 42)
    self.assertEqual(result.selected_candidate, "C1")
    selected = next(candidate for candidate in result.candidates if candidate.name == result.selected_candidate)
    self.assertAlmostEqual(selected.lateral_offset_m, 1.25)
    self.assertGreater(result.vlm_changed_path_meters, 0.0)
    self.assertGreater(result.vlm_changed_speed_mps, 0.0)
    self.assertLessEqual(result.speed_cap_mps, base.current_speed)

  def test_high_speed_cap_does_not_expand_speed(self):
    program = parse_rtp(SAMPLE_RTP.replace("speed_cap_mps=11.0", "speed_cap_mps=40.0"))
    base = BasePlan(
      frame_id=1,
      model_log_mono_time_ns=1_000_000,
      t=(0.0, 0.2),
      x=(0.0, 5.0),
      y=(0.0, 0.0),
      speeds=(12.0, 12.0),
      desired_curvature=0.0,
      v_ego=12.0,
    )
    result = PathSynth().compile(base, program)
    self.assertTrue(result.valid)
    self.assertLessEqual(result.speed_cap_mps, 12.0)

  def test_percent_speed_cap_scales_desired_speed(self):
    program = parse_rtp(SAMPLE_RTP.replace("speed_cap_mps=2.5", "speed_cap_mps=25%"))
    base = BasePlan(
      frame_id=2,
      model_log_mono_time_ns=1_000_000,
      t=(0.0, 0.2),
      x=(0.0, 5.0),
      y=(0.0, 0.0),
      speeds=(20.0, 20.0),
      desired_curvature=0.0,
      v_ego=6.0,
    )
    result = PathSynth().compile(base, program)
    self.assertTrue(result.valid)
    self.assertAlmostEqual(result.speed_cap_mps, 5.0)


class TestConstructionSideCompiler(unittest.TestCase):
  def test_left_construction_compiles_to_right_bias(self):
    program = parse_rtp(_labels_to_rtp(("construction_left",)))
    self.assertEqual(program.meta, "BIAS_RIGHT")
    self.assertLess(program.lat_bias_m, 0.0)
    self.assertIsNone(program.speed_scale)
    self.assertIn("left_edge_s8_48_margin1.25", program.avoid)

  def test_right_construction_compiles_to_left_bias(self):
    program = parse_rtp(_labels_to_rtp(("construction_right",)))
    self.assertEqual(program.meta, "BIAS_LEFT")
    self.assertGreater(program.lat_bias_m, 0.0)
    self.assertIsNone(program.speed_scale)
    self.assertIn("right_edge_s8_48_margin1.25", program.avoid)

  def test_generic_construction_without_side_does_not_laterally_guess(self):
    program = parse_rtp(_labels_to_rtp(("cones",)))
    self.assertEqual(program.meta, "BASE")
    self.assertAlmostEqual(program.lat_bias_m, 0.0)
    self.assertIsNone(program.speed_scale)
    self.assertIsNone(program.speed_cap_mps)
    self.assertEqual(program.avoid, ())

  def test_construction_shift_left_compiles_to_right_hazard_left_bias(self):
    program = parse_rtp(_labels_to_rtp(("construction_shift_left",)))
    self.assertEqual(program.scene, "construction_right")
    self.assertEqual(program.meta, "BIAS_LEFT")
    self.assertGreater(program.lat_bias_m, 0.0)
    self.assertIn("right_edge_s8_48_margin1.25", program.avoid)

  def test_construction_drive_left_compiles_to_right_hazard_left_bias(self):
    program = parse_rtp(_labels_to_rtp(("construction_drive_left",)))
    self.assertEqual(program.scene, "construction_right")
    self.assertEqual(program.meta, "BIAS_LEFT")
    self.assertGreater(program.lat_bias_m, 0.0)
    self.assertIn("right_edge_s8_48_margin1.25", program.avoid)

  def test_construction_shift_right_compiles_to_left_hazard_right_bias(self):
    program = parse_rtp(_labels_to_rtp(("construction_shift_right",)))
    self.assertEqual(program.scene, "construction_left")
    self.assertEqual(program.meta, "BIAS_RIGHT")
    self.assertLess(program.lat_bias_m, 0.0)
    self.assertIn("left_edge_s8_48_margin1.25", program.avoid)

  def test_construction_drive_right_compiles_to_left_hazard_right_bias(self):
    program = parse_rtp(_labels_to_rtp(("construction_drive_right",)))
    self.assertEqual(program.scene, "construction_left")
    self.assertEqual(program.meta, "BIAS_RIGHT")
    self.assertLess(program.lat_bias_m, 0.0)
    self.assertIn("left_edge_s8_48_margin1.25", program.avoid)

  def test_construction_shift_confidence_uses_qwen_score_margin(self):
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_shift_right",),
        {"construction_shift_right": 1.64, "construction_shift_left": 0.50},
      ),
      0.84,
    )
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_shift_right",),
        {"construction_shift_right": 0.91, "construction_shift_left": 0.09},
      ),
      0.72,
    )

  def test_construction_drive_confidence_uses_qwen_score_margin(self):
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_drive_left",),
        {
          "construction_drive_left": CONSTRUCTION_ACTION_IMMEDIATE_SCORE + 0.60,
          "construction_drive_right": CONSTRUCTION_ACTION_IMMEDIATE_SCORE + 0.60 - CONSTRUCTION_ACTION_IMMEDIATE_MARGIN - 0.10,
        },
      ),
      0.96,
    )
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_drive_left",),
        {"construction_drive_left": 2.40, "construction_drive_right": 1.40},
      ),
      0.84,
    )
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_drive_left",),
        {"construction_drive_left": 1.64, "construction_drive_right": 0.50},
      ),
      0.84,
    )
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_drive_left",),
        {"construction_drive_left": 0.91, "construction_drive_right": 0.09},
      ),
      0.72,
    )

  def test_construction_side_confidence_can_activate_durable_lateral_plan(self):
    confidence = _score_calibrated_construction_confidence(
      ("cones", "construction_right"),
      {"cones": 3.55, "construction_right": 0.375, "construction_left": -0.375},
    )
    self.assertEqual(confidence, 0.80)
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("cones", "construction_right")), confidence))
    self.assertEqual(program.confidence, 0.80)

  def test_rtp_confidence_rewrite_preserves_program_and_updates_confidence(self):
    text = _labels_to_rtp(("construction_shift_right",))
    updated = _with_rtp_confidence(text, 0.84)
    program = parse_rtp(updated)
    self.assertEqual(program.scene, "construction_left")
    self.assertAlmostEqual(program.confidence, 0.84)

  def test_trt_construction_side_choice_left_maps_to_left_hazard(self):
    word_scores = _choice_word_scores(left=8.0, right=4.0, none=0.0, clear=-1.0)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("construction_left", "construction_right"))
    self.assertEqual(labels, ("construction_left",))
    self.assertEqual(choice["answer"], "left")
    self.assertIsNone(choice["neutral"])
    self.assertGreater(scores["construction_left"], scores["construction_right"])

  def test_trt_construction_side_choice_rejects_weak_tie(self):
    word_scores = _choice_word_scores(left=4.0, right=3.75, none=0.0, clear=-1.0)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("construction_left", "construction_right"))
    self.assertEqual(labels, ("none",))
    self.assertEqual(choice["answer"], "left")
    self.assertIsNone(choice["neutral"])
    self.assertTrue(choice["rejected_by_margin"])

  def test_trt_choice_score_thresholds_can_override_choice_margin(self):
    word_scores = _choice_word_scores(left=4.0, right=3.625, none=0.0, clear=-1.0)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("construction_left", "construction_right"))
    self.assertEqual(labels, ("none",))
    self.assertEqual(choice["min_margin"], 0.5)

    labels, scores, choice = _choice_scores_from_word_scores(
      word_scores,
      ("construction_left", "construction_right"),
      {"construction_left": 0.3},
    )
    self.assertEqual(labels, ("construction_left",))
    self.assertEqual(choice["min_margin"], 0.3)
    self.assertEqual(choice["spec_min_margin"], 0.5)

  def test_traffic_signal_choice_uses_neutral_letters_not_color_words(self):
    word_scores = _choice_word_scores(red=9.5, green=9.5, go=9.0, clear=9.0, none=9.5, absent=9.0, A=1.0, B=1.0, C=5.0)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("red_stop_light", "green_go_light"))
    self.assertEqual(labels, ("green_go_light",))
    self.assertEqual(choice["answer"], "C")
    self.assertIn("A", choice["allowed"])
    self.assertIn("B", choice["allowed"])
    self.assertIn("C", choice["allowed"])
    self.assertNotIn("green", choice["allowed"])
    self.assertNotIn("go", choice["allowed"])
    self.assertNotIn("absent", choice["allowed"])
    self.assertNotIn("none", choice["allowed"])
    self.assertNotIn("clear", choice["allowed"])
    self.assertEqual(choice["neutral"], "A")
    self.assertEqual(scores["green_go_light"], 4.0)
    self.assertIn("traffic_signal_calibration", choice)

  def test_traffic_signal_choice_requires_signal_letter_to_beat_neutral_and_opposite(self):
    word_scores = _choice_word_scores(A=2.0, B=1.2, C=1.0, red=12.05, green=12.03, go=10.48, stop=12.81, absent=9.64, none=15.25)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("red_stop_light", "green_go_light"))
    self.assertEqual(labels, ("none",))
    self.assertLess(scores["green_go_light"], 0.5)
    self.assertLess(scores["red_stop_light"], 0.5)
    self.assertEqual(choice["answer"], "A")

  def test_traffic_signal_choice_accepts_red_when_b_beats_a_and_c(self):
    word_scores = _choice_word_scores(A=0.0, B=3.0, C=1.0, red=12.47, green=12.28, go=10.11, stop=11.59, absent=8.55, none=13.25)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("red_stop_light", "green_go_light"))
    self.assertEqual(labels, ("red_stop_light",))
    self.assertGreaterEqual(scores["red_stop_light"], 0.5)
    self.assertEqual(choice["answer"], "B")

  def test_traffic_signal_choice_accepts_green_when_c_beats_a_and_b(self):
    word_scores = _choice_word_scores(A=0.0, B=1.0, C=3.0, red=14.35, green=14.96, go=12.67, stop=13.86, absent=9.88, none=16.08)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("red_stop_light", "green_go_light"))
    self.assertEqual(labels, ("green_go_light",))
    self.assertGreaterEqual(scores["green_go_light"], 0.5)
    self.assertEqual(choice["answer"], "C")

  def test_stop_or_signal_choice_uses_go_not_green_color_word_or_clear(self):
    word_scores = _choice_word_scores(stop=2.0, green=9.5, go=5.0, clear=9.0, none=9.5, absent=1.0)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("stop_sign", "green_go_light"))
    self.assertEqual(labels, ("green_go_light",))
    self.assertEqual(choice["answer"], "go")
    self.assertIn("go", choice["allowed"])
    self.assertNotIn("green", choice["allowed"])
    self.assertIn("absent", choice["allowed"])
    self.assertNotIn("none", choice["allowed"])
    self.assertNotIn("clear", choice["allowed"])
    self.assertEqual(choice["neutral"], "absent")
    self.assertEqual(scores["green_go_light"], 4.0)

  def test_traffic_signal_choice_prompt_requires_lit_lamps(self):
    prompt = _choice_text(("red_stop_light", "green_go_light"), "speed=2.5 mps")
    self.assertGreater(len(prompt), 900)
    self.assertIn("Only the bright illuminated lamp matters", prompt)
    self.assertIn("Dark unlit lenses", prompt)
    self.assertIn("top lamp position is red", prompt)
    self.assertIn("green planned-path overlay is not a traffic light", prompt)
    self.assertIn("Answer C", prompt)
    self.assertIn("Allowed answer words: A, B, C", prompt)

  def test_stop_or_signal_choice_prompt_requires_real_sign_or_lit_green(self):
    prompt = _choice_text(("stop_sign", "green_go_light"), "speed=2.5 mps")
    self.assertGreater(len(prompt), 900)
    self.assertIn("real STOP sign", prompt)
    self.assertIn("Only the bright illuminated traffic-signal lamp matters", prompt)
    self.assertIn("Answer absent", prompt)
    self.assertIn("Allowed answer words: stop, go, absent", prompt)
    self.assertIn("green planned-path overlay is not a traffic signal", prompt)

  def test_trt_choice_score_thresholds_can_calibrate_construction_presence_margin(self):
    word_scores = {
      "present": 10.4,
      "absent": 7.0,
    }
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("cones", "barrier"))
    self.assertEqual(labels, ("none",))
    self.assertEqual(choice["min_margin"], 4.0)

    labels, scores, choice = _choice_scores_from_word_scores(
      word_scores,
      ("cones", "barrier"),
      {"cones": 3.3},
    )
    self.assertEqual(labels, ("cones",))
    self.assertEqual(choice["min_margin"], 3.3)
    self.assertEqual(choice["spec_min_margin"], 4.0)

  def test_trt_shift_choice_left_maps_directly_to_shift_left_label(self):
    word_scores = _choice_word_scores(A=8.0, B=4.0, none=0.0, clear=-1.0)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("construction_shift_left", "construction_shift_right"))
    self.assertEqual(labels, ("construction_shift_left",))
    self.assertEqual(choice["answer"], "A")
    self.assertGreater(scores["construction_shift_left"], scores["construction_shift_right"])
    self.assertGreaterEqual(choice["competitor_margin"], choice["competitor_min_margin"])

  def test_trt_drive_choice_left_maps_directly_to_drive_left_label(self):
    word_scores = _choice_word_scores(left=8.0, right=4.0, clear=-1.0)
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, ("construction_drive_left", "construction_drive_right"))
    self.assertEqual(labels, ("construction_drive_left",))
    self.assertEqual(choice["answer"], "left")
    self.assertGreater(scores["construction_drive_left"], scores["construction_drive_right"])
    self.assertGreaterEqual(choice["competitor_margin"], choice["competitor_min_margin"])

  def test_trt_shift_choice_rejects_low_margin_between_candidates(self):
    labels, scores, choice = _choice_scores_from_word_scores(
      _choice_word_scores(A=7.1484375, B=7.1640625, clear=0.0),
      ("construction_shift_left", "construction_shift_right"),
    )
    self.assertEqual(labels, ("none",))
    self.assertTrue(choice["rejected_by_margin"])
    self.assertLess(choice["competitor_margin"], choice["competitor_min_margin"])
    self.assertGreater(scores["construction_shift_right"], 0.5)

  def test_trt_construction_shift_prompt_anchors_colored_corridor_edges(self):
    prompt = _choice_text(("construction_shift_left", "construction_shift_right"), "speed=2.5 mps")
    self.assertIn("cyan PATH A line is a candidate path shifted left", prompt)
    self.assertIn("pink PATH B line is a candidate path shifted right", prompt)
    self.assertIn("Choose the candidate path ID that avoids visible construction", prompt)
    self.assertIn("blue edge line marks the ego-left side", prompt)
    self.assertIn("purple edge line marks the ego-right side", prompt)
    self.assertIn("Allowed answer words: A, B, clear", prompt)
    self.assertIn("A = use PATH A", prompt)
    self.assertIn("B = use PATH B", prompt)

  def test_trt_construction_drive_prompt_uses_direct_driver_direction(self):
    prompt = _choice_text(("construction_drive_left", "construction_drive_right"), "speed=2.5 mps")
    self.assertIn("SAFE EGO DRIVING DIRECTION", prompt)
    self.assertIn("image left is ego/driver left", prompt)
    self.assertIn("Answer left when the safe bounded path should move toward image-left", prompt)
    self.assertIn("Do not answer the side where the cone or barrier is located", prompt)
    self.assertIn("Allowed answer words: left, right, clear", prompt)

  def test_trt_construction_presence_uses_present_absent_with_high_margin(self):
    spec = _choice_group_spec(("cones", "barrier"))
    self.assertEqual(spec["allowed"], ("present", "absent"))
    self.assertEqual(spec["word_to_label"], {"present": "cones"})
    self.assertGreaterEqual(spec["min_margin"], 4.0)
    labels, scores, choice = _choice_scores_from_word_scores(
      _choice_word_scores(present=11.0, absent=8.0, none=0.0, clear=0.0),
      ("cones", "barrier"),
    )
    self.assertEqual(labels, ("none",))
    self.assertTrue(choice["rejected_by_margin"])
    self.assertEqual(scores["cones"], 3.0)

  def test_trt_construction_side_prompt_scores_after_natural_image_prefix(self):
    prompt = _choice_text(("construction_left", "construction_right"), "speed=2.5 mps")
    self.assertIn("blue edge line marks the ego-left edge", prompt)
    self.assertIn("purple edge line marks the ego-right edge", prompt)
    self.assertIn("nearest colored corridor edge", prompt)
    self.assertIn("nearest the blue edge, answer left", prompt)
    self.assertIn("nearest the purple edge, answer right", prompt)
    self.assertIn("Do not flip side merely because a cone is slightly right of the blue line", prompt)
    self.assertIn("judge side by the colored corridor edge", prompt)
    self.assertIn("just inside the blue ego-left edge line", prompt)
    self.assertIn("just inside the purple ego-right edge line", prompt)
    self.assertIn("Allowed answer words: left, right", prompt)
    self.assertTrue(prompt.endswith("Answer:"))

  def test_trt_construction_edge_color_prompt_avoids_left_right_answer_words(self):
    spec = _choice_group_spec(("construction_blue_edge", "construction_purple_edge"))
    self.assertEqual(spec["allowed"], ("blue", "purple"))
    self.assertEqual(spec["word_to_label"], {"blue": "construction_blue_edge", "purple": "construction_purple_edge"})
    prompt = _choice_text(("construction_blue_edge", "construction_purple_edge"), "speed=2.5 mps")
    self.assertIn("blue line is the ego-left edge", prompt)
    self.assertIn("purple line is the ego-right edge", prompt)
    self.assertIn("Do not answer a steering direction", prompt)
    self.assertIn("Do not answer image-left or image-right", prompt)
    self.assertIn("choose the colored edge that is closest to the row", prompt)
    self.assertIn("Allowed answer words: blue, purple", prompt)
    self.assertNotIn("Allowed answer words: left, right", prompt)

  def test_trt_construction_candidate_choice_scores_same_frame_pair(self):
    spec = _choice_group_spec(("construction_blocks_left_candidate", "construction_blocks_right_candidate"))
    self.assertEqual(spec["allowed"], ("cyan", "pink", "none"))
    self.assertEqual(spec["word_to_label"], {
      "cyan": "construction_blocks_left_candidate",
      "pink": "construction_blocks_right_candidate",
    })
    prompt = _choice_text(("construction_blocks_left_candidate", "construction_blocks_right_candidate"), "speed=2.5 mps")
    self.assertIn("SAME-FRAME candidate trajectory corridor", prompt)
    self.assertIn("cyan / blue-green corridor is the bounded candidate path shifted left", prompt)
    self.assertIn("magenta / pink corridor is the bounded candidate path shifted right", prompt)
    self.assertIn("Do not answer the safe path", prompt)
    self.assertIn("Allowed answer words: cyan, pink, none", prompt)

    labels, scores, choice = _choice_scores_from_word_scores(
      _choice_word_scores(cyan=7.0, pink=4.0, none=1.0),
      ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
    )
    self.assertEqual(labels, ("construction_blocks_left_candidate",))
    self.assertEqual(choice["answer"], "cyan")
    self.assertGreater(scores["construction_blocks_left_candidate"], scores["construction_blocks_right_candidate"])

  def test_trt_construction_edge_color_compact_binary_prompt_fits_contract(self):
    prompt = _score_text("construction_blue_edge", "speed=2.5 mps", score_prompt_mode="construction_compact")
    self.assertIn("Scored label: construction_blue_edge", prompt)
    self.assertIn("blue line is the ego-left edge", prompt)
    self.assertIn("Question: Is the relevant construction hazard touching", prompt)
    self.assertIn("blue ego-left corridor edge", prompt)
    self.assertIn("Answer exactly yes or no", prompt)
    self.assertLess(len(prompt), 900)

  def test_trt_construction_presence_compact_binary_prompt_fits_contract(self):
    prompt = _score_text("cones", "speed=2.5 mps", score_prompt_mode="construction_compact")
    self.assertIn("Scored label: cones", prompt)
    self.assertIn("actual tracked path and vehicle-width corridor", prompt)
    self.assertIn("Edge-local inset panels", prompt)
    self.assertIn("affecting the green planned corridor", prompt)
    self.assertIn("Answer exactly yes or no", prompt)
    self.assertLess(len(prompt), 900)

  def test_trt_construction_candidate_binary_prompt_fits_contract(self):
    prompt = _score_text("construction_blocks_left_candidate", "speed=2.5 mps", score_prompt_mode="construction_compact")
    self.assertIn("Scored label: construction_blocks_left_candidate", prompt)
    self.assertIn("candidate-comparison board", prompt)
    self.assertIn("left-shifted candidate path", prompt)
    self.assertIn("cyan / blue-green corridor", prompt)
    self.assertIn("this candidate corridor is obstructed", prompt)
    self.assertIn("Answer exactly yes or no", prompt)
    self.assertLess(len(prompt), 1200)

  def test_construction_edge_color_labels_compile_to_away_biases(self):
    blue = parse_rtp(_labels_to_rtp(("construction_blue_edge",)))
    purple = parse_rtp(_labels_to_rtp(("construction_purple_edge",)))
    self.assertEqual(blue.scene, "construction_left")
    self.assertLess(blue.lat_bias_m, 0.0)
    self.assertEqual(purple.scene, "construction_right")
    self.assertGreater(purple.lat_bias_m, 0.0)

  def test_construction_candidate_blocked_labels_compile_to_opposite_biases(self):
    left_blocked = parse_rtp(_labels_to_rtp(("cones", "construction_blocks_left_candidate")))
    right_blocked = parse_rtp(_labels_to_rtp(("cones", "construction_blocks_right_candidate")))
    both_blocked = parse_rtp(_labels_to_rtp(("cones", "construction_blocks_left_candidate", "construction_blocks_right_candidate")))
    self.assertEqual(left_blocked.scene, "construction_left")
    self.assertEqual(left_blocked.meta, "BIAS_RIGHT")
    self.assertLess(left_blocked.lat_bias_m, 0.0)
    self.assertEqual(right_blocked.scene, "construction_right")
    self.assertEqual(right_blocked.meta, "BIAS_LEFT")
    self.assertGreater(right_blocked.lat_bias_m, 0.0)
    self.assertEqual(both_blocked.scene, "construction_presence_unknown")
    self.assertEqual(both_blocked.lat_bias_m, 0.0)

  def test_candidate_obstruction_score_resolver_keeps_more_blocked_candidate(self):
    labels = ("cones", "construction_blocks_left_candidate", "construction_blocks_right_candidate")
    right_blocked = _resolve_candidate_obstruction_scores(
      labels,
      {"construction_blocks_left_candidate": 1.2, "construction_blocks_right_candidate": 2.0},
      margin=0.4,
    )
    ambiguous = _resolve_candidate_obstruction_scores(
      labels,
      {"construction_blocks_left_candidate": 1.8, "construction_blocks_right_candidate": 2.0},
      margin=0.4,
    )
    self.assertEqual(right_blocked, ("cones", "construction_blocks_right_candidate"))
    self.assertEqual(ambiguous, ("cones",))

  def test_trt_construction_edge_color_confidence_calibration(self):
    confidence = _score_calibrated_construction_confidence(
      ("construction_purple_edge",),
      {"construction_blue_edge": -1.2, "construction_purple_edge": 1.2},
    )
    self.assertEqual(confidence, 0.84)

  def test_compact_edge_confidence_can_immediately_activate_strong_side(self):
    default_confidence = _score_calibrated_construction_confidence(
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.67},
    )
    compact_confidence = _score_calibrated_construction_confidence(
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.67},
      strong_edge_immediate=True,
    )
    ambiguous_confidence = _score_calibrated_construction_confidence(
      ("construction_purple_edge",),
      {"construction_blue_edge": 1.1, "construction_purple_edge": 2.67},
      strong_edge_immediate=True,
    )
    near_bootstrap_confidence = _score_calibrated_construction_confidence(
      ("construction_blue_edge",),
      {
        "construction_blue_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE + 0.05,
        "construction_purple_edge": 0.0,
      },
      strong_edge_immediate=True,
    )
    just_below_bootstrap_confidence = _score_calibrated_construction_confidence(
      ("construction_blue_edge",),
      {
        "construction_blue_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE - 0.05,
        "construction_purple_edge": 0.0,
      },
      strong_edge_immediate=True,
    )
    self.assertEqual(default_confidence, 0.84)
    self.assertEqual(compact_confidence, 0.96)
    self.assertEqual(ambiguous_confidence, 0.96)
    self.assertEqual(near_bootstrap_confidence, 0.96)
    self.assertEqual(just_below_bootstrap_confidence, 0.84)

  def test_candidate_choice_confidence_can_immediately_activate_strong_side(self):
    strong_left_candidate = _score_calibrated_construction_confidence(
      ("construction_blocks_left_candidate",),
      {"construction_blocks_left_candidate": 10.06, "construction_blocks_right_candidate": 0.0},
    )
    weak_left_candidate = _score_calibrated_construction_confidence(
      ("construction_blocks_left_candidate",),
      {"construction_blocks_left_candidate": 1.45, "construction_blocks_right_candidate": 0.0},
    )
    ambiguous_left_candidate = _score_calibrated_construction_confidence(
      ("construction_blocks_left_candidate",),
      {"construction_blocks_left_candidate": 10.06, "construction_blocks_right_candidate": 9.0},
    )

    self.assertEqual(strong_left_candidate, 0.96)
    self.assertEqual(weak_left_candidate, 0.84)
    self.assertEqual(ambiguous_left_candidate, 0.84)

  def test_choice_token_variants_include_no_space_and_space_forms(self):
    variants = _choice_token_id_variants(_FakeChoiceProcessor(), words=("left", "right"))
    self.assertEqual(variants["left"], (101, 201, 301))
    self.assertEqual(variants["right"], (102, 202, 302))

  def test_text_output_selection_falls_back_to_engine_tensor(self):
    self.assertEqual(_select_text_output("logits", {"inputs_embeds", "selected_logits"}), ("logits", "selected_logits"))
    self.assertEqual(_select_text_output("logits", {"inputs_embeds", "selected_hidden"}), ("hidden", "selected_hidden"))
    self.assertEqual(_select_text_output("logits", {"inputs_embeds", "full_hidden"}), ("full_hidden", "full_hidden"))
    self.assertEqual(_select_text_output("hidden", {"inputs_embeds", "selected_logits"}), ("logits", "selected_logits"))
    self.assertEqual(_select_text_output("full_hidden", {"inputs_embeds", "selected_hidden"}), ("hidden", "selected_hidden"))
    self.assertEqual(_select_text_output("fixed_logits", {"inputs_embeds", "selected_logits"}), ("fixed_logits", "selected_logits"))
    self.assertEqual(_text_output_tensor_name("fixed_logits"), "selected_logits")
    self.assertEqual(_text_output_tensor_name("full_hidden"), "full_hidden")

  def test_text_precision_accepts_fp8_as_additive_candidate(self):
    self.assertEqual(_text_precision(Namespace(text_precision="fp8")), "fp8")
    with self.assertRaises(ValueError):
      _text_precision(Namespace(text_precision="fp12"))

  def test_construction_edge_binary_overrides_only_edge_groups(self):
    args = Namespace(
      construction_edge_binary=True,
      label_decision_mode="choice",
      score_prompt_mode="full",
      construction_text_engine=None,
      construction_text_precision="same",
    )
    edge_args = _args_for_score_group(args, ("construction_blue_edge",))
    presence_args = _args_for_score_group(args, ("cones", "barrier"))
    self.assertEqual(edge_args.label_decision_mode, "binary")
    self.assertEqual(edge_args.score_prompt_mode, "construction_compact")
    self.assertEqual(presence_args.label_decision_mode, "choice")

  def test_strongly_typed_text_engine_uses_distinct_path(self):
    base = Namespace(
      artifact_dir=__import__("pathlib").Path("F:/qwen_trt_export"),
      text_precision="fp8",
      text_seq_len=768,
      text_output="logits",
      label_decision_mode="binary",
      score_prompt_mode="full",
      text_strongly_typed=False,
    )
    strong = Namespace(**{**vars(base), "text_strongly_typed": True})
    self.assertTrue(str(_generic_text_engine_path(base)).endswith("qwen_text_36layer_fp8_seq768_trt.engine"))
    self.assertTrue(str(_generic_text_engine_path(strong)).endswith("qwen_text_36layer_fp8_seq768_strong_trt.engine"))

  def test_torch_attention_implementation_uses_distinct_text_engine_path(self):
    base = Namespace(
      artifact_dir=__import__("pathlib").Path("F:/qwen_trt_export"),
      text_precision="fp16",
      text_seq_len=576,
      text_output="hidden",
      label_decision_mode="choice",
      score_prompt_mode="full",
      text_strongly_typed=False,
      torch_attn_implementation="sdpa",
    )
    eager = Namespace(**{**vars(base), "torch_attn_implementation": "eager"})
    self.assertEqual(_torch_attn_implementation(base), "sdpa")
    self.assertTrue(str(_generic_text_engine_path(base)).endswith("qwen_text_36layer_fp16_seq576_hidden_choice_trt.engine"))
    self.assertTrue(str(_generic_text_engine_path(eager)).endswith("qwen_text_36layer_fp16_seq576_hidden_choice_attn_eager_trt.engine"))

  def test_int32_text_position_dtype_uses_distinct_text_engine_path(self):
    base = Namespace(
      artifact_dir=__import__("pathlib").Path("F:/qwen_trt_export"),
      text_precision="fp16",
      text_seq_len=576,
      text_output="fixed_logits",
      label_decision_mode="choice",
      score_prompt_mode="full",
      text_strongly_typed=False,
      torch_attn_implementation="sdpa",
      text_position_dtype="int64",
    )
    int32 = Namespace(**{**vars(base), "text_position_dtype": "int32"})
    self.assertEqual(_text_position_dtype(base), "int64")
    self.assertEqual(_text_position_dtype(int32), "int32")
    self.assertTrue(str(_generic_text_engine_path(base)).endswith("qwen_text_36layer_fp16_seq576_fixed_logits_choice_trt.engine"))
    self.assertTrue(str(_generic_text_engine_path(int32)).endswith("qwen_text_36layer_fp16_seq576_fixed_logits_choice_pos_int32_trt.engine"))

  def test_construction_group_can_use_additive_text_engine_override(self):
    engine_path = __import__("pathlib").Path("F:/qwen_trt_export/fp8_trt/qwen_text_36layer_fp8_seq576_hidden_choice_trt.engine")
    args = Namespace(
      text_engine=None,
      text_precision="nvfp4",
      construction_text_engine=engine_path,
      construction_text_precision="fp8",
    )

    construction_args = _args_for_score_group(args, ("construction_left", "construction_right"))
    non_construction_args = _args_for_score_group(args, ("red_stop_light", "green_go_light"))

    self.assertEqual(_construction_text_precision(args), "fp8")
    self.assertIs(non_construction_args, args)
    self.assertIsNot(construction_args, args)
    self.assertEqual(construction_args.text_engine, engine_path)
    self.assertEqual(construction_args.text_precision, "fp8")

  def test_construction_candidate_binary_overrides_only_candidate_groups(self):
    args = Namespace(
      label_decision_mode="choice",
      score_prompt_mode="full",
      construction_candidate_binary=True,
      construction_candidate_choice=False,
      construction_edge_binary=False,
      construction_text_engine=None,
      construction_text_precision="same",
    )

    candidate_args = _args_for_score_group(args, ("construction_blocks_left_candidate",))
    edge_args = _args_for_score_group(args, ("construction_blue_edge",))

    self.assertIsNot(candidate_args, args)
    self.assertEqual(candidate_args.label_decision_mode, "binary")
    self.assertEqual(candidate_args.score_prompt_mode, "construction_compact")
    self.assertIs(edge_args, args)

  def test_construction_candidate_choice_uses_override_engine_and_seq_len(self):
    engine_path = __import__("pathlib").Path("F:/qwen_trt_export/nvfp4_trt/qwen_text_36layer_nvfp4_seq576_choice_trt.engine")
    args = Namespace(
      label_decision_mode="binary",
      score_prompt_mode="full",
      construction_candidate_binary=False,
      construction_candidate_choice=True,
      construction_edge_binary=False,
      construction_text_engine=engine_path,
      construction_text_precision="nvfp4",
      construction_text_seq_len=576,
    )

    candidate_args = _args_for_score_group(args, ("construction_blocks_left_candidate", "construction_blocks_right_candidate"))
    edge_args = _args_for_score_group(args, ("construction_blue_edge",))

    self.assertIsNot(candidate_args, args)
    self.assertEqual(candidate_args.label_decision_mode, "choice")
    self.assertEqual(candidate_args.score_prompt_mode, "construction_compact")
    self.assertEqual(candidate_args.text_engine, engine_path)
    self.assertEqual(candidate_args.text_precision, "nvfp4")
    self.assertEqual(candidate_args.text_seq_len, 576)
    self.assertIs(edge_args, args)

  def test_construction_side_and_edge_choice_use_override_engine_and_seq_len(self):
    engine_path = __import__("pathlib").Path("F:/qwen_trt_export/nvfp4_trt/qwen_text_36layer_nvfp4_seq576_choice_trt.engine")
    args = Namespace(
      label_decision_mode="binary",
      score_prompt_mode="full",
      construction_side_choice=True,
      construction_edge_choice=True,
      construction_candidate_choice=False,
      construction_edge_binary=True,
      construction_text_engine=engine_path,
      construction_text_precision="nvfp4",
      construction_text_seq_len=576,
    )

    side_args = _args_for_score_group(args, ("construction_left", "construction_right"))
    edge_args = _args_for_score_group(args, ("construction_blue_edge", "construction_purple_edge"))
    candidate_args = _args_for_score_group(args, ("construction_blocks_left_candidate", "construction_blocks_right_candidate"))

    for group_args in (side_args, edge_args):
      self.assertIsNot(group_args, args)
      self.assertEqual(group_args.label_decision_mode, "choice")
      self.assertEqual(group_args.score_prompt_mode, "construction_compact")
      self.assertEqual(group_args.text_engine, engine_path)
      self.assertEqual(group_args.text_precision, "nvfp4")
      self.assertEqual(group_args.text_seq_len, 576)
    self.assertEqual(candidate_args.label_decision_mode, "binary")

  def test_candidate_relative_choice_keeps_none_when_neutral_is_dominant(self):
    choice = {
      "answer": "none",
      "word_scores": {"cyan": 6.88, "pink": 4.74, "none": 12.24},
    }
    labels, scores, adjusted = _apply_candidate_relative_choice(
      ("none",),
      {"construction_blocks_left_candidate": -5.36, "construction_blocks_right_candidate": -7.50},
      choice,
      ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
      enabled=True,
      min_margin=1.5,
      neutral_margin=CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN,
    )

    self.assertEqual(labels, ("none",))
    self.assertEqual(scores["construction_blocks_left_candidate"], -5.36)
    self.assertEqual(adjusted["answer"], "none")
    self.assertFalse(adjusted["candidate_relative_choice"]["enabled"])
    self.assertEqual(adjusted["candidate_relative_choice"]["reason"], "neutral_dominant")

  def test_candidate_relative_choice_selects_obstructed_candidate_when_neutral_is_close(self):
    choice = {
      "answer": "none",
      "word_scores": {"cyan": 6.88, "pink": 4.74, "none": 7.20},
    }
    labels, scores, adjusted = _apply_candidate_relative_choice(
      ("none",),
      {"construction_blocks_left_candidate": -0.32, "construction_blocks_right_candidate": -2.46},
      choice,
      ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
      enabled=True,
      min_margin=1.5,
      neutral_margin=CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN,
    )

    self.assertEqual(labels, ("construction_blocks_left_candidate",))
    self.assertGreater(scores["construction_blocks_left_candidate"], scores["construction_blocks_right_candidate"])
    self.assertEqual(adjusted["answer"], "cyan")
    self.assertTrue(adjusted["candidate_relative_choice"]["enabled"])

  def test_candidate_relative_choice_keeps_none_when_margin_is_weak(self):
    labels, scores, adjusted = _apply_candidate_relative_choice(
      ("none",),
      {"construction_blocks_left_candidate": -1.0, "construction_blocks_right_candidate": -1.2},
      {"answer": "none", "word_scores": {"cyan": 5.1, "pink": 4.8, "none": 6.0}},
      ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
      enabled=True,
      min_margin=1.5,
      neutral_margin=CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN,
    )

    self.assertEqual(labels, ("none",))
    self.assertEqual(scores["construction_blocks_left_candidate"], -1.0)
    self.assertEqual(adjusted["answer"], "none")

  def test_candidate_label_uses_auxiliary_candidate_image(self):
    def png(color):
      buf = BytesIO()
      Image.new("RGB", (2, 2), color).save(buf, format="PNG")
      return base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
      "scene_board_image_b64": png((1, 2, 3)),
      "scene_board_aux_images_b64": {
        "candidate_left": png((10, 20, 30)),
        "candidate_right": png((40, 50, 60)),
        "candidate_pair": png((70, 80, 90)),
      },
    }

    left = _image_from_payload_for_labels(payload, ("construction_blocks_left_candidate",))
    right = _image_from_payload_for_labels(payload, ("construction_blocks_right_candidate",))
    pair = _image_from_payload_for_labels(payload, ("construction_blocks_left_candidate", "construction_blocks_right_candidate"))
    main = _image_from_payload_for_labels(payload, ("cones",))

    self.assertEqual(left.getpixel((0, 0)), (10, 20, 30))
    self.assertEqual(right.getpixel((0, 0)), (40, 50, 60))
    self.assertEqual(pair.getpixel((0, 0)), (70, 80, 90))
    self.assertEqual(main.getpixel((0, 0)), (1, 2, 3))

  def test_set_qwen_attention_implementation_updates_top_and_language_model_configs(self):
    class Config:
      _attn_implementation = "sdpa"

    class LanguageModel:
      config = Config()

    class InnerModel:
      language_model = LanguageModel()

    class Model:
      config = Config()
      model = InnerModel()

    model = Model()
    self.assertEqual(_set_qwen_attention_implementation(model, "eager"), "eager")
    self.assertEqual(model.config._attn_implementation, "eager")
    self.assertEqual(model.model.language_model.config._attn_implementation, "eager")

  def test_worker_full_prompt_mode_uses_distinct_text_engine_path(self):
    args = Namespace(
      artifact_dir=__import__("pathlib").Path("F:/qwen_trt_export"),
      text_precision="nvfp4",
      text_seq_len=768,
      text_output="logits",
      label_decision_mode="binary",
      score_prompt_mode="worker-full",
      text_strongly_typed=True,
    )
    self.assertEqual(_score_prompt_mode(args), "worker_full")
    self.assertTrue(str(_generic_text_engine_path(args)).endswith("qwen_text_36layer_nvfp4_seq768_worker_full_strong_trt.engine"))

  def test_worker_full_prompt_matches_qwen_label_worker_contract(self):
    vehicle_state = "speed=5.0 mps"
    expected = (
      f"{SCORE_PROMPT}\nVehicle state: {vehicle_state}\n"
      f"Question: {SCORE_QUESTIONS['construction_left']}"
    )
    self.assertEqual(_score_text("construction_left", vehicle_state, "worker_full"), expected)

  def test_runtime_contract_tracks_text_strongly_typed(self):
    args = Namespace(
      model_dir=__import__("pathlib").Path("."),
      runtime_mode="score",
      image_mode="full",
      image_size=168,
      vision_precision="fp32",
      text_seq_len=768,
      text_precision="fp8",
      text_output="logits",
      label_decision_mode="binary",
      score_prompt_mode="full",
      text_position_mode="qwen",
      torch_attn_implementation="eager",
      text_strongly_typed=True,
      score_rotate_groups=False,
      score_rotate_shared_engine=False,
      vehicle_state="speed=5.0 mps",
      use_payload_vehicle_state=True,
      payload_vehicle_state_scope="auto",
      score_threshold=0.0,
      score_thresholds_map={},
      score_cache_ttl_frames=3,
      score_durable_labels="",
      score_negative_clear_threshold=2.0,
      construction_mirror_consistency=False,
      construction_mirror_fusion=True,
      construction_edge_binary=False,
      construction_candidate_binary=False,
      construction_candidate_score_resolve=False,
      construction_candidate_diff_margin=0.4,
      construction_text_precision="fp8",
      construction_text_engine=__import__("pathlib").Path("F:/qwen_trt_export/fp8_trt/qwen_text_36layer_fp8_seq576_hidden_choice_trt.engine"),
      enable_signal_head=False,
      signal_min_probability=0.7,
      signal_min_margin=0.75,
    )
    contract = _runtime_contract(args, (("construction_left", "construction_right"),))["contract"]["runtime"]
    prompt_contract = _runtime_contract(args, (("construction_left", "construction_right"),))["contract"]["prompt"]
    self.assertTrue(contract["text_strongly_typed"])
    self.assertEqual(contract["torch_attn_implementation"], "eager")
    self.assertEqual(contract["text_position_dtype"], "int64")
    self.assertTrue(contract["construction_mirror_fusion"])
    self.assertEqual(contract["construction_text_precision"], "fp8")
    self.assertTrue(contract["use_payload_vehicle_state"])
    self.assertEqual(contract["payload_vehicle_state_scope"], "auto")
    self.assertFalse(contract["construction_side_choice"])
    self.assertFalse(contract["construction_edge_choice"])
    self.assertFalse(contract["construction_edge_binary"])
    self.assertEqual(contract["construction_clear_rtp_confidence"], CONSTRUCTION_CLEAR_RTP_CONFIDENCE)
    self.assertEqual(contract["construction_reactivate_min_presence_score"], CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE)
    self.assertEqual(contract["construction_edge_bootstrap_score"], CONSTRUCTION_EDGE_BOOTSTRAP_SCORE)
    self.assertEqual(contract["construction_edge_bootstrap_margin"], CONSTRUCTION_EDGE_BOOTSTRAP_MARGIN)
    self.assertEqual(contract["construction_edge_bootstrap_opposite_max"], CONSTRUCTION_EDGE_BOOTSTRAP_OPPOSITE_MAX)
    self.assertEqual(contract["construction_edge_neutral_bootstrap_score"], CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE)
    self.assertEqual(contract["construction_edge_neutral_bootstrap_margin"], CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN)
    self.assertEqual(contract["construction_edge_toward_path_override_score"], CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE)
    self.assertEqual(contract["construction_edge_toward_path_override_margin"], CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN)
    self.assertEqual(contract["construction_direct_consensus_max_age_frames"], CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES)
    self.assertEqual(contract["construction_direct_semantic_score"], CONSTRUCTION_DIRECT_SEMANTIC_SCORE)
    self.assertEqual(contract["construction_direct_semantic_margin"], CONSTRUCTION_DIRECT_SEMANTIC_MARGIN)
    self.assertEqual(contract["construction_direct_edge_score"], CONSTRUCTION_DIRECT_EDGE_SCORE)
    self.assertEqual(contract["construction_direct_edge_margin"], CONSTRUCTION_DIRECT_EDGE_MARGIN)
    self.assertEqual(contract["construction_semantic_neutral_bootstrap_score"], CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE)
    self.assertEqual(contract["construction_semantic_neutral_bootstrap_margin"], CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_MARGIN)
    self.assertEqual(contract["construction_semantic_toward_path_override_score"], CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE)
    self.assertEqual(contract["construction_semantic_toward_path_override_margin"], CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN)
    self.assertEqual(contract["construction_side_committed_away_min_tracked_offset_m"], CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M)
    self.assertEqual(contract["construction_side_toward_path_override_score"], CONSTRUCTION_SIDE_TOWARD_PATH_OVERRIDE_SCORE)
    self.assertEqual(contract["construction_side_early_reversal_score"], CONSTRUCTION_SIDE_EARLY_REVERSAL_SCORE)
    self.assertEqual(contract["construction_side_early_reversal_margin"], CONSTRUCTION_SIDE_EARLY_REVERSAL_MARGIN)
    self.assertEqual(contract["construction_side_early_reversal_opposite_max"], CONSTRUCTION_SIDE_EARLY_REVERSAL_OPPOSITE_MAX)
    self.assertEqual(contract["construction_side_early_reversal_max_tracked_offset_m"], CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_TRACKED_OFFSET_M)
    self.assertEqual(contract["construction_side_early_reversal_max_age_frames"], CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_AGE_FRAMES)
    self.assertEqual(contract["construction_action_bootstrap_max_tracked_offset_m"], CONSTRUCTION_ACTION_BOOTSTRAP_MAX_TRACKED_OFFSET_M)
    self.assertEqual(contract["construction_action_continue_min_tracked_offset_m"], CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M)
    self.assertEqual(contract["construction_action_immediate_score"], CONSTRUCTION_ACTION_IMMEDIATE_SCORE)
    self.assertEqual(contract["construction_action_immediate_margin"], CONSTRUCTION_ACTION_IMMEDIATE_MARGIN)
    self.assertEqual(contract["construction_action_contradictory_override_score"], CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_SCORE)
    self.assertEqual(contract["construction_action_contradictory_override_margin"], CONSTRUCTION_ACTION_CONTRADICTORY_OVERRIDE_MARGIN)
    self.assertEqual(contract["construction_committed_conflict_requires_action"], CONSTRUCTION_COMMITTED_CONFLICT_REQUIRES_ACTION)
    self.assertEqual(contract["construction_candidate_relative_neutral_margin"], CONSTRUCTION_CANDIDATE_RELATIVE_NEUTRAL_MARGIN)
    self.assertEqual(contract["construction_state_machine_version"], CONSTRUCTION_STATE_MACHINE_VERSION)
    self.assertIn("rendered_score_prompts_sha256", prompt_contract)

  def test_runtime_contract_tracks_construction_edge_binary(self):
    base = Namespace(
      model_dir=__import__("pathlib").Path("."),
      runtime_mode="score",
      image_mode="full",
      image_size=168,
      vision_precision="fp32",
      text_seq_len=768,
      text_precision="fp8",
      text_output="logits",
      label_decision_mode="binary",
      score_prompt_mode="full",
      text_position_mode="qwen",
      torch_attn_implementation="sdpa",
      text_strongly_typed=False,
      score_rotate_groups=True,
      score_rotate_shared_engine=True,
      vehicle_state="speed=5.0 mps",
      use_payload_vehicle_state=True,
      payload_vehicle_state_scope="auto",
      score_threshold=0.0,
      score_thresholds_map={},
      score_cache_ttl_frames=3,
      score_durable_labels="",
      score_negative_clear_threshold=2.0,
      construction_mirror_consistency=False,
      construction_mirror_fusion=True,
      construction_edge_binary=False,
      construction_candidate_binary=False,
      construction_candidate_score_resolve=False,
      construction_candidate_diff_margin=0.4,
      construction_text_precision="same",
      construction_text_engine=None,
      enable_signal_head=False,
      signal_min_probability=0.7,
      signal_min_margin=0.75,
    )
    groups = (("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge"))
    default_contract = _runtime_contract(base, groups)
    edge_binary = Namespace(**{**vars(base), "construction_edge_binary": True})
    edge_contract = _runtime_contract(edge_binary, groups)
    self.assertFalse(default_contract["contract"]["runtime"]["construction_edge_binary"])
    self.assertTrue(edge_contract["contract"]["runtime"]["construction_edge_binary"])
    self.assertNotEqual(default_contract["contract_sha256"], edge_contract["contract_sha256"])

  def test_runtime_contract_tracks_rendered_scoring_prompt_text(self):
    base = Namespace(
      model_dir=__import__("pathlib").Path("."),
      runtime_mode="score",
      image_mode="full",
      image_size=168,
      vision_precision="fp32",
      text_seq_len=768,
      text_precision="fp8",
      text_output="logits",
      label_decision_mode="binary",
      score_prompt_mode="construction-compact",
      text_position_mode="qwen",
      torch_attn_implementation="sdpa",
      text_strongly_typed=False,
      score_rotate_groups=True,
      score_rotate_shared_engine=True,
      vehicle_state="speed=5.0 mps",
      use_payload_vehicle_state=True,
      payload_vehicle_state_scope="auto",
      score_threshold=0.0,
      score_thresholds_map={},
      score_cache_ttl_frames=3,
      score_durable_labels="",
      score_negative_clear_threshold=2.0,
      construction_mirror_consistency=False,
      construction_mirror_fusion=True,
      construction_edge_binary=True,
      construction_candidate_binary=False,
      construction_candidate_score_resolve=False,
      construction_candidate_diff_margin=0.4,
      construction_text_precision="same",
      construction_text_engine=None,
      enable_signal_head=False,
      signal_min_probability=0.7,
      signal_min_margin=0.75,
    )
    prompt_contract = _runtime_contract(base, (("construction_blue_edge", "construction_purple_edge"),))["contract"]["prompt"]
    self.assertIn("rendered_score_prompts_sha256", prompt_contract)
    self.assertNotEqual(prompt_contract["score_questions_sha256"], prompt_contract["rendered_score_prompts_sha256"])

  def test_payload_vehicle_state_auto_scope_feeds_physical_and_construction_state_groups(self):
    args = Namespace(
      vehicle_state="speed=2.5 mps",
      use_payload_vehicle_state=True,
      payload_vehicle_state_scope="auto",
    )
    payload_state = "frame=1 v_ego=2.0mps lead present yes distance 12.0 m relative speed -0.7 m/s"
    self.assertEqual(
      _vehicle_state_for_labels(args, ("construction_left", "construction_right"), payload_state),
      payload_state,
    )
    self.assertEqual(
      _vehicle_state_for_labels(args, ("construction_blue_edge", "construction_purple_edge"), payload_state),
      payload_state,
    )
    self.assertEqual(
      _vehicle_state_for_labels(args, ("slower_lead", "true_moving_lead"), payload_state),
      payload_state,
    )

  def test_mirrored_construction_label_maps_back_to_original_side(self):
    self.assertEqual(_mirror_construction_labels_to_original(("construction_left",)), ("construction_right",))
    self.assertEqual(_mirror_construction_labels_to_original(("construction_right",)), ("construction_left",))
    self.assertEqual(_mirror_construction_labels_to_original(("none",)), ("none",))

  def test_construction_mirror_consistency_accepts_opposite_mirror_side(self):
    original = {
      "labels": ["construction_right"],
      "label_scores": {"construction_right": 2.0, "construction_left": -1.0},
      "choice": {"answer": "left"},
      "rtp_text": _labels_to_rtp(("construction_right",)),
      "text": _labels_to_rtp(("construction_right",)),
    }
    mirrored = {
      "labels": ["construction_left"],
      "label_scores": {"construction_left": 2.0, "construction_right": -1.0},
      "choice": {"answer": "right"},
    }

    adjusted = _apply_construction_mirror_consistency(original, mirrored, negative_clear_threshold=2.0)

    self.assertEqual(adjusted["labels"], ["construction_right"])
    self.assertTrue(adjusted["choice"]["construction_mirror_consistency"]["accepted"])

  def test_construction_mirror_consistency_rejects_same_mirror_side(self):
    original = {
      "labels": ["construction_right"],
      "labels_text": "construction_right",
      "label_scores": {"construction_right": 2.0, "construction_left": -1.0},
      "choice": {"answer": "left"},
      "rtp_text": _labels_to_rtp(("construction_right",)),
      "text": _labels_to_rtp(("construction_right",)),
    }
    mirrored = {
      "labels": ["construction_right"],
      "label_scores": {"construction_right": 2.0, "construction_left": -1.0},
      "choice": {"answer": "left"},
    }

    adjusted = _apply_construction_mirror_consistency(original, mirrored, negative_clear_threshold=2.0)

    self.assertEqual(adjusted["labels"], ["none"])
    self.assertLessEqual(adjusted["label_scores"]["construction_right"], -2.25)
    self.assertFalse(adjusted["choice"]["construction_mirror_consistency"]["accepted"])
    self.assertIn("scene=nominal", adjusted["rtp_text"])

  def test_construction_mirror_fusion_clears_weak_original_when_mirror_disagrees(self):
    original = {
      "labels": ["construction_left"],
      "label_scores": {"construction_left": 0.26, "construction_right": -0.26},
      "choice": {"answer": "left", "word_scores": {"left": 24.827, "right": 24.568}},
      "rtp_text": _labels_to_rtp(("construction_left",)),
      "text": _labels_to_rtp(("construction_left",)),
    }
    mirrored = {
      "labels": ["construction_left"],
      "label_scores": {"construction_left": 1.17, "construction_right": -1.17},
      "choice": {"answer": "left", "word_scores": {"left": 25.390, "right": 24.215}},
    }

    adjusted = _apply_construction_mirror_fusion(
      original,
      mirrored,
      ("construction_left", "construction_right"),
    )

    self.assertEqual(adjusted["labels"], ["none"])
    self.assertEqual(parse_rtp(adjusted["rtp_text"]).meta, "BASE")
    self.assertLessEqual(adjusted["label_scores"]["construction_left"], -2.25)
    self.assertLessEqual(adjusted["label_scores"]["construction_right"], -2.25)
    self.assertIn(
      adjusted["choice"]["construction_mirror_fusion"]["policy"],
      ("disagreement_cleared_weak_original", "cleared_margin_rejected_original"),
    )

  def test_construction_mirror_fusion_boosts_when_mirror_maps_to_original_side(self):
    original = {
      "labels": ["construction_right"],
      "label_scores": {"construction_left": -1.2, "construction_right": 1.2},
      "choice": {"answer": "right", "word_scores": {"left": 4.0, "right": 5.2}},
      "rtp_text": _labels_to_rtp(("construction_right",)),
      "text": _labels_to_rtp(("construction_right",)),
    }
    mirrored = {
      "labels": ["construction_left"],
      "label_scores": {"construction_left": 1.4, "construction_right": -1.4},
      "choice": {"answer": "left", "word_scores": {"left": 6.0, "right": 4.6}},
    }

    adjusted = _apply_construction_mirror_fusion(
      original,
      mirrored,
      ("construction_left", "construction_right"),
    )

    self.assertEqual(adjusted["labels"], ["construction_right"])
    self.assertEqual(parse_rtp(adjusted["rtp_text"]).meta, "BIAS_LEFT")
    self.assertGreater(adjusted["label_scores"]["construction_right"], adjusted["label_scores"]["construction_left"])
    self.assertEqual(adjusted["choice"]["construction_mirror_fusion"]["policy"], "agreement_fused")

  def test_construction_mirror_fusion_preserves_original_left_when_mirror_is_biased_left(self):
    original = {
      "labels": ["construction_left"],
      "label_scores": {"construction_left": 1.11, "construction_right": -1.11},
      "choice": {"answer": "left", "word_scores": {"left": 25.293, "right": 24.181}},
      "rtp_text": _labels_to_rtp(("construction_left",)),
      "text": _labels_to_rtp(("construction_left",)),
    }
    mirrored = {
      "labels": ["construction_left"],
      "label_scores": {"construction_left": 0.21, "construction_right": -0.21},
      "choice": {"answer": "left", "word_scores": {"left": 24.824, "right": 24.618}},
    }

    adjusted = _apply_construction_mirror_fusion(
      original,
      mirrored,
      ("construction_left", "construction_right"),
    )

    self.assertEqual(adjusted["labels"], ["construction_left"])
    self.assertEqual(parse_rtp(adjusted["rtp_text"]).meta, "BIAS_RIGHT")
    self.assertGreater(adjusted["label_scores"]["construction_left"], adjusted["label_scores"]["construction_right"])


class TestLeadCompiler(unittest.TestCase):
  def test_metadrive_harness_declares_required_vehicle_scenes(self):
    self.assertEqual(
      VEHICLE_SCENES,
      {
        "true_moving_lead",
        "slower_lead",
        "braking_lead",
        "stopped_lead",
        "cut_in_vehicle",
        "crossing_vehicle",
        "irrelevant_vehicle",
      },
    )
    self.assertEqual(
      _expected_lead_class_from_spawned([{"kind": "lead_vehicle_slower", "expected_lead_class": "slower_lead"}]),
      "slower_lead",
    )
    self.assertEqual(ROUTE_VEHICLE_MODEL_CLASSES, ("DefaultVehicle", "LVehicle", "MVehicle", "SVehicle", "XLVehicle"))
    self.assertEqual(ROUTE_VEHICLE_MODEL_CLASS, "DefaultVehicle")
    self.assertFalse(ROUTE_VEHICLE_USE_SPECIAL_COLOR)
    self.assertEqual(ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD, 0.0)
    self.assertAlmostEqual(route_vehicle_visual_heading(0.0), 0.0)
    self.assertAlmostEqual(route_vehicle_visual_heading(0.5), 0.5)
    self.assertEqual(route_vehicle_class_from_name(ROUTE_VEHICLE_MODEL_CLASS).__name__, "DefaultVehicle")

  def test_metadrive_camera_frame_defaults_from_bgr_to_rgb(self):
    bgr = np.asarray([[[10, 20, 240], [30, 180, 40]]], dtype=np.uint8)
    rgb = _convert_camera_frame_color(bgr, "bgr")
    np.testing.assert_array_equal(rgb, np.asarray([[[240, 20, 10], [40, 180, 30]]], dtype=np.uint8))
    np.testing.assert_array_equal(_convert_camera_frame_color(rgb, "rgb"), rgb)

  def test_route_vehicle_state_exposes_physics_without_expected_label(self):
    tracks = {}
    first_frame = [
        {
          "kind": "lead_vehicle_slower",
          "id": "lead-1",
          "route_s_m": 30.0,
          "lateral_m": 0.25,
          "expected_lead_class": "slower_lead",
        },
        {
          "kind": "irrelevant_vehicle",
          "id": "side-1",
          "route_s_m": 25.0,
          "lateral_m": 6.0,
          "expected_lead_class": "irrelevant_vehicle",
        },
      ]
    state0 = nearest_route_vehicle_state(
      first_frame,
      current_long_m=10.0,
      ego_speed_mps=5.0,
      track_history=tracks,
      dt=0.1,
    )
    self.assertEqual(state0["lead_present"], 0)
    second_frame = [
        {
          "kind": "lead_vehicle_slower",
          "id": "lead-1",
          "route_s_m": 30.2,
          "lateral_m": 0.26,
          "expected_lead_class": "slower_lead",
        },
        {
          "kind": "irrelevant_vehicle",
          "id": "side-1",
          "route_s_m": 25.5,
          "lateral_m": 6.0,
          "expected_lead_class": "irrelevant_vehicle",
        },
      ]
    state = nearest_route_vehicle_state(
      second_frame,
      current_long_m=10.0,
      ego_speed_mps=5.0,
      track_history=tracks,
      dt=0.1,
    )
    self.assertEqual(state["lead_present"], 1)
    self.assertEqual(state["lead_source"], "track")
    self.assertAlmostEqual(state["lead_distance_m"], 20.2)
    self.assertAlmostEqual(state["lead_lateral_m"], 0.26)
    self.assertAlmostEqual(state["lead_speed_mps"], 2.0)
    self.assertAlmostEqual(state["lead_rel_speed_mps"], -3.0)
    self.assertAlmostEqual(state["lead_closing_mps"], 3.0)
    self.assertAlmostEqual(state["lead_accel_mps2"], 0.0)
    self.assertAlmostEqual(state["lead_lateral_velocity_mps"], 0.1)
    self.assertNotIn("expected_lead_class", state)

  def test_route_vehicle_state_uses_initial_physical_track_speed_when_available(self):
    state = nearest_route_vehicle_state(
      [
        {
          "kind": "lead_vehicle_true_moving",
          "id": "lead-1",
          "route_s_m": 30.0,
          "lateral_m": 0.20,
          "speed_mps": 5.1,
          "accel_mps2": 0.0,
          "lateral_rate_mps": 0.0,
          "expected_lead_class": "stopped_lead",
        }
      ],
      current_long_m=10.0,
      ego_speed_mps=5.0,
      track_history={},
      dt=0.1,
    )
    self.assertEqual(state["lead_present"], 1)
    self.assertEqual(state["lead_source"], "track")
    self.assertAlmostEqual(state["lead_distance_m"], 20.0)
    self.assertAlmostEqual(state["lead_speed_mps"], 5.1)
    self.assertAlmostEqual(state["lead_rel_speed_mps"], 0.1)
    self.assertAlmostEqual(state["lead_closing_mps"], -0.1)
    self.assertNotIn("expected_lead_class", state)

  def test_route_vehicle_state_reports_openpilot_lateral_with_runtime_route_sign(self):
    tracks = {}
    first_frame = [{"kind": "cut_in_vehicle", "id": "cut-in", "route_s_m": 20.0, "lateral_m": 2.0}]
    state0 = nearest_route_vehicle_state(
      first_frame,
      current_long_m=0.0,
      ego_speed_mps=5.0,
      track_history=tracks,
      dt=0.1,
      route_lateral_sign_to_openpilot=-1.0,
    )
    self.assertEqual(state0["lead_present"], 0)

    second_frame = [{"kind": "cut_in_vehicle", "id": "cut-in", "route_s_m": 20.5, "lateral_m": 1.8}]
    state = nearest_route_vehicle_state(
      second_frame,
      current_long_m=0.0,
      ego_speed_mps=5.0,
      track_history=tracks,
      dt=0.1,
      route_lateral_sign_to_openpilot=-1.0,
    )
    self.assertEqual(state["lead_present"], 1)
    self.assertAlmostEqual(state["lead_lateral_m"], -1.8)
    self.assertAlmostEqual(state["lead_lateral_velocity_mps"], 2.0)

  def _args(self):
    return Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=10.0,
      durable_speed_recover_m=3.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
    )

  def test_true_moving_lead_does_not_create_speed_plan(self):
    program = parse_rtp(_labels_to_rtp(("true_moving_lead",)))
    self.assertEqual(program.scene, "true_moving_lead")
    self.assertEqual(program.meta, "BASE")
    self.assertEqual(program.avoid, ())
    self.assertIsNone(durable_speed_plan_from_program(program, current_long_m=0.0, args=self._args()))

  def test_construction_presence_without_side_does_not_create_speed_plan(self):
    program = parse_rtp(_labels_to_rtp(("cones",)))
    self.assertEqual(program.scene, "construction_presence_unknown")
    self.assertEqual(program.meta, "BASE")
    self.assertEqual(program.avoid, ())
    self.assertIsNone(durable_speed_plan_from_program(program, current_long_m=0.0, args=self._args()))

  def test_slower_lead_creates_proportional_speed_cap(self):
    program = parse_rtp(_labels_to_rtp(("slower_lead",)))
    self.assertEqual(program.scene, "slower_lead")
    self.assertEqual(program.meta, "SLOW")
    self.assertAlmostEqual(program.speed_scale, 0.80)
    self.assertIn("lead_vehicle_s12_45", program.avoid)
    plan = durable_speed_plan_from_program(program, current_long_m=4.0, args=self._args())
    self.assertIsNotNone(plan)
    self.assertEqual(plan.source_token, "lead_vehicle_s12_45")
    self.assertAlmostEqual(plan.speed_cap_mps, 4.0)

  def test_braking_lead_creates_stronger_speed_cap(self):
    program = parse_rtp(_labels_to_rtp(("braking_lead",)))
    self.assertEqual(program.scene, "braking_lead")
    self.assertEqual(program.meta, "YIELD")
    self.assertAlmostEqual(program.speed_scale, 0.45)
    plan = durable_speed_plan_from_program(program, current_long_m=4.0, args=self._args())
    self.assertIsNotNone(plan)
    self.assertEqual(plan.source_token, "lead_vehicle_s8_45")
    self.assertAlmostEqual(plan.speed_cap_mps, 2.25)

  def test_stopped_lead_creates_stop_plan(self):
    program = parse_rtp(_labels_to_rtp(("stopped_lead",)))
    self.assertEqual(program.scene, "stopped_lead")
    self.assertEqual(program.meta, "STOP")
    self.assertEqual(program.speed_cap_mps, 0.0)
    self.assertAlmostEqual(program.stop_s, 18.0)
    plan = durable_speed_plan_from_program(program, current_long_m=4.0, args=self._args())
    self.assertIsNotNone(plan)
    self.assertEqual(plan.source_token, "lead_vehicle_s8_35")
    self.assertEqual(plan.speed_cap_mps, 0.0)

  def test_cut_in_vehicle_choice_maps_to_yield_plan(self):
    word_scores = _choice_word_scores(**{
      "moving": 0.0,
      "slower": 0.1,
      "braking": 0.2,
      "stopped": 0.3,
      "merge": 3.0,
      "crossing": 0.5,
      "irrelevant": 0.2,
      "none": 0.0,
    })
    group = ("true_moving_lead", "slower_lead", "braking_lead", "stopped_lead", "cut_in_vehicle", "crossing_vehicle", "irrelevant_vehicle")
    labels, scores, choice = _choice_scores_from_word_scores(word_scores, group)
    self.assertEqual(labels, ("cut_in_vehicle",))
    self.assertEqual(choice["answer"], "merge")
    self.assertGreater(scores["cut_in_vehicle"], scores["slower_lead"])
    program = parse_rtp(_labels_to_rtp(labels))
    self.assertEqual(program.scene, "cut_in_vehicle")
    self.assertIn("cut_in_vehicle_s8_30", program.avoid)

  def test_irrelevant_vehicle_clears_stale_lead_speed_plan(self):
    args = self._args()
    stale_lead = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=45.0,
      ramp_out_end_long_m=50.0,
      speed_cap_mps=2.25,
      stop_s=None,
      source_token="lead_vehicle_s8_45",
      source_meta="YIELD",
      confidence=0.74,
    )
    program = parse_rtp(_labels_to_rtp(("irrelevant_vehicle",)))
    updated = update_durable_speed_plans(
      {"lead_vehicle_s8_45": stale_lead},
      None,
      program,
      current_long_m=10.0,
      args=args,
    )
    self.assertEqual(updated, {})

  def test_physical_lead_state_vetoes_impossible_none_choice(self):
    group = ("true_moving_lead", "slower_lead", "braking_lead", "stopped_lead", "cut_in_vehicle", "crossing_vehicle", "irrelevant_vehicle")
    record = {
      "lead_present": 1,
      "lead_distance_m": 16.6,
      "lead_lateral_m": 0.0,
      "lead_speed_mps": 1.4,
      "lead_rel_speed_mps": -0.7,
      "lead_closing_mps": 0.7,
      "lead_accel_mps2": 0.0,
      "lead_lateral_velocity_mps": 0.0,
    }
    requirement = lead_semantics.classify_lead_track(record)
    self.assertEqual(requirement["expected_class"], "slower_lead")
    self.assertEqual(lead_semantics.lead_choice_word(str(requirement["expected_class"])), "slower")
    selected, scores, choice = _apply_lead_state_consistency(
      ("none",),
      {label: -1.0 for label in group},
      {"answer": "none"},
      group,
      (
        "lead present yes; source track; distance 16.6 m; lateral offset 0.0 m; "
        "lead speed 1.4 m/s; relative speed -0.7 m/s; closing 0.7 m/s; "
        "acceleration 0.0 m/s2; lateral velocity 0.0 m/s"
      ),
    )
    self.assertEqual(selected, ("slower_lead",))
    self.assertGreater(scores["slower_lead"], 0.0)
    self.assertEqual(choice["answer"], "slower")
    self.assertEqual(choice["answer_before_physical_filter"], "none")
    self.assertTrue(choice["physical_state_override"])

  def test_physical_track_state_distinguishes_cut_in_crossing_and_irrelevant(self):
    group = ("true_moving_lead", "slower_lead", "braking_lead", "stopped_lead", "cut_in_vehicle", "crossing_vehicle", "irrelevant_vehicle")
    cases = (
      (
        "lead present yes; source track; distance 18.0 m; lateral offset 2.0 m; "
        "lead speed 2.0 m/s; relative speed -0.5 m/s; closing 0.5 m/s; "
        "acceleration 0.0 m/s2; lateral velocity -0.6 m/s",
        "cut_in_vehicle",
        "merge",
      ),
      (
        "lead present yes; source track; distance 18.0 m; lateral offset -2.0 m; "
        "lead speed 0.2 m/s; relative speed -2.3 m/s; closing 2.3 m/s; "
        "acceleration 0.0 m/s2; lateral velocity 1.2 m/s",
        "crossing_vehicle",
        "crossing",
      ),
      (
        "lead present yes; source track; distance 18.0 m; lateral offset 2.2 m; "
        "lead speed 2.5 m/s; relative speed 0.0 m/s; closing 0.0 m/s; "
        "acceleration 0.0 m/s2; lateral velocity 0.0 m/s",
        "irrelevant_vehicle",
        "irrelevant",
      ),
    )
    for state_text, expected_label, expected_word in cases:
      with self.subTest(expected_label=expected_label):
        record = {
          "lead_present": 1,
          "lead_distance_m": 18.0,
          "lead_lateral_m": 2.0 if expected_label == "cut_in_vehicle" else (-2.0 if expected_label == "crossing_vehicle" else 2.2),
          "lead_speed_mps": 2.0 if expected_label == "cut_in_vehicle" else (0.2 if expected_label == "crossing_vehicle" else 2.5),
          "lead_rel_speed_mps": -0.5 if expected_label == "cut_in_vehicle" else (-2.3 if expected_label == "crossing_vehicle" else 0.0),
          "lead_closing_mps": 0.5 if expected_label == "cut_in_vehicle" else (2.3 if expected_label == "crossing_vehicle" else 0.0),
          "lead_accel_mps2": 0.0,
          "lead_lateral_velocity_mps": -0.6 if expected_label == "cut_in_vehicle" else (1.2 if expected_label == "crossing_vehicle" else 0.0),
        }
        requirement = lead_semantics.classify_lead_track(record)
        self.assertEqual(requirement["expected_class"], expected_label)
        self.assertEqual(lead_semantics.lead_choice_word(str(requirement["expected_class"])), expected_word)
        selected, scores, choice = _apply_lead_state_consistency(
          ("none",),
          {label: -1.0 for label in group},
          {"answer": "none"},
          group,
          state_text,
        )
        self.assertEqual(selected, (expected_label,))
        self.assertGreater(scores[expected_label], 0.0)
        self.assertEqual(choice["answer"], expected_word)

  def test_desired_speed_keeps_slow_matched_lead_from_becoming_true_moving(self):
    group = ("true_moving_lead", "slower_lead", "braking_lead", "stopped_lead", "cut_in_vehicle", "crossing_vehicle", "irrelevant_vehicle")
    record = {
      "lead_present": 1,
      "desired_speed_mps": 2.5,
      "lead_distance_m": 12.0,
      "lead_lateral_m": 0.0,
      "lead_speed_mps": 1.875,
      "lead_rel_speed_mps": -0.36,
      "lead_closing_mps": 0.36,
      "lead_accel_mps2": 0.0,
      "lead_lateral_velocity_mps": 0.0,
    }
    requirement = lead_semantics.classify_lead_track(record)
    self.assertEqual(requirement["expected_class"], "slower_lead")
    self.assertEqual(requirement["reason"], "closing_on_slower_lead")

    selected, scores, choice = _apply_lead_state_consistency(
      ("true_moving_lead",),
      {label: -1.0 for label in group},
      {"answer": "moving"},
      group,
      (
        "frame=62 v_ego=2.2mps desired speed 2.5 m/s curv=0.00000 "
        "lead present yes; source track; distance 12.0 m; lateral offset 0.0 m; "
        "lead speed 1.9 m/s; relative speed -0.4 m/s; closing 0.4 m/s; "
        "acceleration 0.0 m/s2; lateral velocity 0.0 m/s"
      ),
    )
    self.assertEqual(selected, ("slower_lead",))
    self.assertGreater(scores["slower_lead"], scores["true_moving_lead"])
    self.assertEqual(choice["answer"], "slower")
    self.assertEqual(choice["answer_before_physical_filter"], "moving")
    self.assertTrue(choice["physical_state_override"])


class TestSceneBoardOverlay(unittest.TestCase):
  def test_default_scene_board_uses_vehicle_width_corridor_without_base_reference(self):
    geometry = OverlayGeometry()
    self.assertGreaterEqual(geometry.planned_corridor_half_width_m, 1.125)
    self.assertGreater(geometry.focus_corridor_extra_width_m, 0.0)
    self.assertLessEqual(geometry.focus_corridor_extra_width_m, 0.50)
    self.assertGreaterEqual(geometry.outside_corridor_dim_alpha, 120)
    self.assertTrue(geometry.dim_outside_corridor)
    self.assertFalse(geometry.draw_base_path_reference)
    self.assertTrue(geometry.draw_corridor_side_guides)
    self.assertTrue(geometry.draw_corridor_side_fill)
    self.assertFalse(geometry.draw_corridor_side_labels)
    self.assertFalse(geometry.draw_edge_insets)

  def test_metadrive_scene_board_args_expose_geometry_without_changing_defaults(self):
    renderer = scene_board_renderer_for_args(Namespace(
      board_width=320,
      board_height=200,
      scene_board_candidate_guides=False,
      candidate_guide_offset_m=1.25,
    ))
    geometry = renderer.geometry
    self.assertAlmostEqual(geometry.planned_corridor_half_width_m, OverlayGeometry.planned_corridor_half_width_m)
    self.assertAlmostEqual(geometry.focus_corridor_extra_width_m, OverlayGeometry.focus_corridor_extra_width_m)
    self.assertEqual(geometry.dim_outside_corridor, OverlayGeometry.dim_outside_corridor)
    self.assertEqual(geometry.outside_corridor_dim_alpha, OverlayGeometry.outside_corridor_dim_alpha)
    self.assertFalse(geometry.draw_candidate_labels)
    self.assertFalse(geometry.draw_base_path_reference)
    self.assertTrue(geometry.draw_corridor_side_guides)
    self.assertFalse(geometry.draw_corridor_side_labels)
    self.assertAlmostEqual(geometry.candidate_lateral_offset_m, 0.0)

  def test_metadrive_scene_board_args_can_tune_visibility_for_qwen_probes(self):
    renderer = scene_board_renderer_for_args(Namespace(
      board_width=320,
      board_height=200,
      scene_board_candidate_guides=True,
      candidate_guide_offset_m=1.1,
      scene_board_corridor_half_width_m=1.35,
      scene_board_focus_extra_width_m=0.25,
      scene_board_dim_outside_corridor=False,
      scene_board_dim_alpha=35,
      scene_board_candidate_labels=True,
      scene_board_base_path_reference=True,
      scene_board_corridor_side_guides=False,
      scene_board_corridor_side_fill=False,
      scene_board_corridor_side_labels=True,
      scene_board_edge_insets=True,
      scene_board_candidate_obstruction_boards=True,
      scene_board_candidate_obstruction_offset_m=1.4,
    ))
    geometry = renderer.geometry
    self.assertAlmostEqual(geometry.planned_corridor_half_width_m, 1.35)
    self.assertAlmostEqual(geometry.focus_corridor_extra_width_m, 0.25)
    self.assertFalse(geometry.dim_outside_corridor)
    self.assertEqual(geometry.outside_corridor_dim_alpha, 35)
    self.assertAlmostEqual(geometry.candidate_lateral_offset_m, 1.1)
    self.assertTrue(geometry.draw_candidate_labels)
    self.assertTrue(geometry.draw_base_path_reference)
    self.assertFalse(geometry.draw_corridor_side_guides)
    self.assertFalse(geometry.draw_corridor_side_fill)
    self.assertTrue(geometry.draw_corridor_side_labels)
    self.assertTrue(geometry.draw_edge_insets)
    self.assertTrue(geometry.draw_candidate_obstruction_boards)
    self.assertAlmostEqual(geometry.candidate_obstruction_offset_m, 1.4)

  def test_scene_board_draws_edge_local_insets_when_enabled(self):
    base = BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(5.0 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    board = UiSceneBoardRenderer(320, 200, geometry=OverlayGeometry(draw_edge_insets=True)).render(base, {"path_lateral_offset_m": 0.0})
    arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
    lower = arr[int(board.height * 0.62):, :, :]
    blue_border = (lower[:, :, 2] > 170) & (lower[:, :, 1] > 100) & (lower[:, :, 0] < 90)
    purple_border = (lower[:, :, 0] > 120) & (lower[:, :, 2] > 170) & (lower[:, :, 1] < 150)
    self.assertGreater(int(blue_border.sum()), 150)
    self.assertGreater(int(purple_border.sum()), 150)

  def test_scene_board_default_dims_off_corridor_background(self):
    base = BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(5.0 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    frame = np.full((240, 320, 3), 210, dtype=np.uint8)
    renderer = UiSceneBoardRenderer(320, 240, geometry=OverlayGeometry(draw_corridor_side_guides=False))
    board = renderer.render(base, {"path_lateral_offset_m": 0.0, "road_frame": frame})
    arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
    center_green = float(arr[120:160, 145:175, 1].mean())
    side_green = float(np.concatenate((arr[120:160, 0:35, 1], arr[120:160, 285:320, 1]), axis=1).mean())
    self.assertGreater(center_green, side_green + 20.0)

  def test_green_path_overlay_uses_active_lateral_offset(self):
    base = BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(5.0 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    renderer = UiSceneBoardRenderer(320, 240, geometry=OverlayGeometry(draw_base_path_reference=True))
    centered = renderer.render(base, {"path_lateral_offset_m": 0.0})
    shifted_left = renderer.render(base, {"path_lateral_offset_m": 1.0})

    def green_x_mean(board):
      arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
      mask = (arr[:, :, 1] > 70) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 90)
      _, xs = np.nonzero(mask)
      self.assertGreater(len(xs), 0)
      return float(xs.mean())

    self.assertLess(green_x_mean(shifted_left), green_x_mean(centered))
    self.assertIn("tracked_path_lat=1.00m", shifted_left.state_text)
    self.assertIn("base_path_lat=0.00m", shifted_left.state_text)

  def test_scene_board_candidate_obstruction_aux_boards_shift_green_path(self):
    base = BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(5.0 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    renderer = UiSceneBoardRenderer(
      320,
      240,
      geometry=OverlayGeometry(draw_candidate_obstruction_boards=True, candidate_obstruction_offset_m=1.25),
    )
    board = renderer.render(base, {"path_lateral_offset_m": 0.0})
    self.assertEqual(set(board.aux_pngs), {"candidate_left", "candidate_right", "candidate_pair"})

    def green_x_mean_png(data: bytes) -> float:
      image = Image.open(BytesIO(data)).convert("RGB")
      arr = np.asarray(image)
      mask = (arr[:, :, 1] > 70) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 90)
      _, xs = np.nonzero(mask)
      self.assertGreater(len(xs), 0)
      return float(xs.mean())

    self.assertLess(green_x_mean_png(board.aux_pngs["candidate_left"]), green_x_mean_png(board.aux_pngs["candidate_right"]))

    pair_image = Image.open(BytesIO(board.aux_pngs["candidate_pair"])).convert("RGB")
    pair_arr = np.asarray(pair_image)
    cyan = (pair_arr[:, :, 0] < 90) & (pair_arr[:, :, 1] > 150) & (pair_arr[:, :, 2] > 150)
    magenta = (pair_arr[:, :, 0] > 150) & (pair_arr[:, :, 1] < 130) & (pair_arr[:, :, 2] > 150)
    self.assertGreater(int(cyan.sum()), 10)
    self.assertGreater(int(magenta.sum()), 10)

  def test_scene_board_draws_base_reference_when_tracked_path_is_shifted(self):
    base = BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(5.0 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    renderer = UiSceneBoardRenderer(320, 240, geometry=OverlayGeometry(draw_base_path_reference=True))
    board = renderer.render(base, {"path_lateral_offset_m": 1.0})
    arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
    green = (arr[:, :, 1] > 70) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 90)
    magenta = (arr[:, :, 0] > 150) & (arr[:, :, 2] > 150) & (arr[:, :, 1] < 120)
    _, green_xs = np.nonzero(green)
    _, magenta_xs = np.nonzero(magenta)
    self.assertGreater(len(green_xs), 10)
    self.assertGreater(len(magenta_xs), 10)
    self.assertGreater(float(magenta_xs.mean()), float(green_xs.mean()))

  def test_scene_board_draws_left_and_right_candidate_guides(self):
    base = BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(5.0 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    renderer = UiSceneBoardRenderer(320, 240, geometry=OverlayGeometry(candidate_lateral_offset_m=1.25))
    board = renderer.render(base, {"path_lateral_offset_m": 0.0})
    arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
    cyan_candidate = (arr[:, :, 2] > 170) & (arr[:, :, 1] > 160) & (arr[:, :, 0] < 130)
    pink_candidate = (arr[:, :, 0] > 170) & (arr[:, :, 2] > 120) & (arr[:, :, 1] < 140)
    _, cyan_xs = np.nonzero(cyan_candidate)
    _, pink_xs = np.nonzero(pink_candidate)
    self.assertGreater(int(cyan_candidate.sum()), 10)
    self.assertGreater(int(pink_candidate.sum()), 10)
    self.assertLess(float(cyan_xs.mean()), float(pink_xs.mean()))

  def test_scene_board_can_draw_side_text_labels(self):
    base = BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(5.0 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    renderer = UiSceneBoardRenderer(320, 240, geometry=OverlayGeometry(draw_corridor_side_labels=True))
    board = renderer.render(base, {"path_lateral_offset_m": 0.0})
    arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
    label_box = (arr[:, :, 0] < 15) & (arr[:, :, 1] < 15) & (arr[:, :, 2] < 15)
    self.assertGreater(int(label_box.sum()), 100)

  def test_scene_board_side_text_labels_follow_projected_edges_when_enabled(self):
    renderer = UiSceneBoardRenderer(320, 240, geometry=OverlayGeometry(draw_corridor_side_labels=True))
    centered_specs = renderer._corridor_side_label_specs(0.0)
    shifted_left_specs = renderer._corridor_side_label_specs(1.0)
    self.assertEqual(centered_specs[0][0], "BLUE LEFT")
    self.assertEqual(centered_specs[1][0], "PURPLE RIGHT")
    self.assertLess(centered_specs[0][1], centered_specs[1][1])
    self.assertLess(shifted_left_specs[0][1], centered_specs[0][1])
    self.assertLess(shifted_left_specs[1][1], centered_specs[1][1])

  def test_scene_board_draws_corridor_side_guides_by_default(self):
    base = BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(5.0 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    board = UiSceneBoardRenderer(320, 240).render(base, {"path_lateral_offset_m": 0.0})
    arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
    blue_left_edge = (arr[:, :, 2] > 160) & (arr[:, :, 1] > 100) & (arr[:, :, 0] < 90)
    purple_right_edge = (arr[:, :, 0] > 120) & (arr[:, :, 2] > 170) & (arr[:, :, 1] < 150)
    self.assertGreater(int(blue_left_edge.sum()), 10)
    self.assertGreater(int(purple_right_edge.sum()), 10)

  def test_scene_board_draws_corridor_side_fill_by_default(self):
    base = BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(5.0 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    frame = np.full((240, 320, 3), 80, dtype=np.uint8)
    board = UiSceneBoardRenderer(320, 240).render(base, {"path_lateral_offset_m": 0.0, "road_frame": frame})
    arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
    corridor = arr[120:230, :, :]
    blue_side = (corridor[:, :, 2] > corridor[:, :, 0] + 35) & (corridor[:, :, 1] > corridor[:, :, 0] + 20)
    purple_side = (corridor[:, :, 0] > corridor[:, :, 1] + 20) & (corridor[:, :, 2] > corridor[:, :, 1] + 35)
    _, blue_xs = np.nonzero(blue_side)
    _, purple_xs = np.nonzero(purple_side)
    self.assertGreater(int(blue_side.sum()), 50)
    self.assertGreater(int(purple_side.sum()), 50)
    self.assertLess(float(blue_xs.mean()), float(purple_xs.mean()))

  def test_scene_board_state_text_includes_production_lead_cues(self):
    base = BasePlan(
      frame_id=4,
      model_log_mono_time_ns=0,
      t=(0.0, 0.2),
      x=(1.0, 10.0),
      y=(0.0, 0.0),
      speeds=(5.0, 5.0),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    board = UiSceneBoardRenderer(320, 240).render(base, {
      "lead_present": 1,
      "lead_source": "track",
      "lead_distance_m": 22.5,
      "lead_lateral_m": -0.2,
      "lead_speed_mps": 2.5,
      "lead_rel_speed_mps": -2.5,
      "lead_closing_mps": 2.5,
      "lead_accel_mps2": -0.4,
      "lead_lateral_velocity_mps": 0.0,
    })
    self.assertIn("desired speed 5.0 m/s", board.state_text)
    self.assertIn("lead present yes", board.state_text)
    self.assertIn("source track", board.state_text)
    self.assertIn("distance 22.5 m", board.state_text)
    self.assertIn("lateral offset -0.2 m", board.state_text)
    self.assertIn("lead speed 2.5 m/s", board.state_text)
    self.assertIn("relative speed -2.5 m/s", board.state_text)
    self.assertIn("closing 2.5 m/s", board.state_text)
    self.assertIn("acceleration -0.4 m/s2", board.state_text)
    self.assertIn("lateral velocity 0.0 m/s", board.state_text)
    self.assertNotIn("lead_distance_m=", board.state_text)

  def test_scene_board_state_text_omits_sim_expected_lead_label(self):
    base = BasePlan(
      frame_id=4,
      model_log_mono_time_ns=0,
      t=(0.0, 0.2),
      x=(1.0, 10.0),
      y=(0.0, 0.0),
      speeds=(5.0, 5.0),
      desired_curvature=0.0,
      v_ego=5.0,
    )
    board = UiSceneBoardRenderer(320, 240).render(base, {
      "lead_present": 1,
      "lead_source": "track",
      "lead_distance_m": 18.0,
      "lead_lateral_m": 0.0,
      "lead_speed_mps": 1.5,
      "lead_rel_speed_mps": -1.0,
      "lead_closing_mps": 1.0,
      "lead_accel_mps2": 0.0,
      "lead_lateral_velocity_mps": 0.0,
      "expected_lead_class": "slower_lead",
      "kind": "lead_vehicle_slower",
    })
    self.assertIn("lead present yes", board.state_text)
    self.assertIn("source track", board.state_text)
    self.assertNotIn("expected_lead_class", board.state_text)
    self.assertNotIn("slower_lead", board.state_text)
    self.assertNotIn("lead_vehicle_slower", board.state_text)


class _FixedAgeRtpEngine(RtpEngine):
  def __init__(self, age_frames: int):
    self.age_frames = age_frames

  def generate(self, frame_id: int, board: SceneBoard, vehicle_state: dict[str, float], deadline_ms: float) -> RtpEngineResult:
    return RtpEngineResult(
      text=_labels_to_rtp(("construction_left",)),
      generated_token_count=0,
      prefill_ms=0.0,
      decode_ms=0.0,
      backend="fixed_age_test",
      source_frame_id=frame_id - self.age_frames,
      labels=("construction_left",),
      label_scores={"construction_left": 1.2, "construction_right": -0.4},
      raw_labels=("construction_left",),
      raw_label_scores={"construction_left": 1.2, "construction_right": -0.4},
      labels_scored_this_request=("construction_left", "construction_right"),
      score_group_index=1,
      label_state_debug={"construction_locked_side": "left"},
      choice={"answer": "left"},
    )


class _ResettableFixedAgeRtpEngine(_FixedAgeRtpEngine):
  def __init__(self, age_frames: int):
    super().__init__(age_frames)
    self.reset_count = 0

  def reset_runtime_state(self) -> None:
    self.reset_count += 1


class TestAsyncRtpEnginePrewarmState(unittest.TestCase):
  def test_reset_runtime_state_clears_prewarm_frame_state(self):
    inner = _ResettableFixedAgeRtpEngine(age_frames=0)
    engine = AsyncRtpEngine(inner, update_period_frames=2, max_age_frames=6)
    board = SceneBoard(1, 1, bytearray([0, 0, 0]), "")
    with engine._lock:
      engine._in_flight = True
      engine._last_submitted_frame = 25
      engine._last_request_frame = 30
      engine._pending = (31, board, {})
      engine._latest = RtpEngineResult(
        text=_labels_to_rtp(("construction_left",)),
        generated_token_count=0,
        prefill_ms=0.0,
        decode_ms=0.0,
        backend="warm",
        source_frame_id=25,
        labels=("construction_left",),
      )
      engine._last_error = "warm stale"
      old_epoch = engine._epoch

    engine.reset_runtime_state()

    with engine._lock:
      self.assertEqual(engine._epoch, old_epoch + 1)
      self.assertFalse(engine._in_flight)
      self.assertIsNone(engine._last_submitted_frame)
      self.assertIsNone(engine._last_request_frame)
      self.assertIsNone(engine._pending)
      self.assertIsNone(engine._latest)
      self.assertEqual(engine._last_error, "")
    self.assertEqual(inner.reset_count, 1)


class TestPlannerTraceabilityAndAge(unittest.TestCase):
  def _base_plan(self) -> BasePlan:
    return BasePlan(
      frame_id=10,
      model_log_mono_time_ns=10,
      t=(0.0, 0.2, 0.4),
      x=(0.0, 5.0, 10.0),
      y=(0.0, 0.0, 0.0),
      speeds=(5.0, 5.0, 5.0),
      desired_curvature=0.0,
      v_ego=5.0,
    )

  def test_accepted_vlm_age_within_configured_bound_publishes_trace(self):
    planner = ReasonedPlanner(
      config=ReasonedPlannerConfig(deadline_ms=50.0, allow_async_rtp=True, max_async_age_frames=2),
      renderer=UiSceneBoardRenderer(160, 120),
      engine=_FixedAgeRtpEngine(age_frames=2),
    )
    result = planner.step(self._base_plan(), {})
    self.assertTrue(result.should_publish)
    self.assertEqual(result.rtp_age_frames, 2)
    self.assertEqual(result.labels, ("construction_left",))
    self.assertEqual(result.label_scores["construction_left"], 1.2)
    self.assertEqual(result.raw_labels, ("construction_left",))
    self.assertEqual(result.raw_label_scores["construction_right"], -0.4)
    self.assertEqual(result.labels_scored_this_request, ("construction_left", "construction_right"))
    self.assertEqual(result.score_group_index, 1)
    self.assertEqual(result.label_state_debug["construction_locked_side"], "left")
    self.assertEqual(result.choice["answer"], "left")

  def test_vlm_age_beyond_configured_bound_is_not_published(self):
    planner = ReasonedPlanner(
      config=ReasonedPlannerConfig(deadline_ms=50.0, allow_async_rtp=True, max_async_age_frames=2),
      renderer=UiSceneBoardRenderer(160, 120),
      engine=_FixedAgeRtpEngine(age_frames=3),
    )
    result = planner.step(self._base_plan(), {})
    self.assertFalse(result.should_publish)
    self.assertEqual(result.rtp_age_frames, 3)
    self.assertIn("rtp_stale_or_not_same_frame", result.invalid_reason)


class TestRotatingScoreState(unittest.TestCase):
  def test_default_score_groups_cover_required_agent_classes_once(self):
    flattened = [label for group in DEFAULT_SCORE_LABEL_GROUPS for label in group]
    self.assertEqual(len(DEFAULT_SCORE_LABEL_GROUPS), len(set(DEFAULT_SCORE_LABEL_GROUPS)))
    for label in (
      "red_stop_light",
      "green_go_light",
      "stop_sign",
      "construction_left",
      "construction_right",
      "construction_blue_edge",
      "construction_purple_edge",
      "pedestrian_in_path",
      "pedestrian_entering_path",
      "vehicle_in_path",
      "vehicle_entering_path",
      "true_moving_lead",
      "slower_lead",
      "braking_lead",
      "stopped_lead",
      "cut_in_vehicle",
      "crossing_vehicle",
      "irrelevant_vehicle",
      "animal_in_path",
      "animal_entering_path",
    ):
      self.assertIn(label, flattened)

  def test_durable_construction_label_survives_moderate_negative(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"),),
      cache_ttl_frames=60,
      durable_labels=("cones", "barrier"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(state.update(("cones", "barrier"), ("cones",), {"cones": 0.5, "barrier": -0.5}, 204), ("cones",))
    self.assertEqual(state.update(("cones", "barrier"), ("none",), {"cones": -1.4, "barrier": -3.0}, 252), ("cones",))
    self.assertEqual(state.update(("cones", "barrier"), ("none",), {"cones": -2.2, "barrier": -3.0}, 264), ("none",))

  def test_construction_side_before_presence_does_not_arm_when_cones_arrive(self):
    state = RotatingScoreState(
      groups=(("construction_left", "construction_right"), ("cones", "barrier")),
      cache_ttl_frames=3,
      durable_labels=("construction_left", "construction_right", "cones", "barrier"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(("construction_left", "construction_right"), ("construction_left",), {"construction_left": 0.8}, 10),
      ("none",),
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones",), {"cones": 1.0}, 11),
      ("cones",),
    )

  def test_construction_side_after_presence_arms(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right")),
      cache_ttl_frames=3,
      durable_labels=("construction_left", "construction_right", "cones", "barrier"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones",), {"cones": 1.0}, 10),
      ("cones",),
    )
    self.assertEqual(
      state.update(("construction_left", "construction_right"), ("construction_left",), {"construction_left": 0.8}, 11),
      ("cones", "construction_left"),
    )

  def test_construction_shift_choice_is_self_evidencing_and_not_cone_gated(self):
    state = RotatingScoreState(
      groups=(("construction_shift_left", "construction_shift_right"), ("construction_left", "construction_right")),
      cache_ttl_frames=3,
      durable_labels=("construction_shift_left", "construction_shift_right", "construction_left", "construction_right"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_shift_left", "construction_shift_right"),
        ("construction_shift_left",),
        {"construction_shift_left": 1.1, "construction_shift_right": 0.1},
        10,
      ),
      ("construction_shift_left",),
    )
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_left",),
        {"construction_left": 1.1, "construction_right": 0.1},
        11,
      ),
      ("construction_shift_left",),
    )

  def test_construction_drive_choice_is_self_evidencing_and_not_cone_gated(self):
    state = RotatingScoreState(
      groups=(("construction_drive_left", "construction_drive_right"), ("construction_left", "construction_right")),
      cache_ttl_frames=3,
      durable_labels=("construction_drive_left", "construction_drive_right", "construction_left", "construction_right"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_drive_left", "construction_drive_right"),
        ("construction_drive_left",),
        {"construction_drive_left": CONSTRUCTION_ACTION_IMMEDIATE_SCORE + 0.1, "construction_drive_right": CONSTRUCTION_ACTION_IMMEDIATE_SCORE + 0.1 - CONSTRUCTION_ACTION_IMMEDIATE_MARGIN},
        10,
        vehicle_state_text="tracked_path_lat=0.00m",
      ),
      ("construction_drive_left",),
    )
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_left",),
        {"construction_left": 1.1, "construction_right": 0.1},
        11,
        vehicle_state_text="tracked_path_lat=0.02m",
      ),
      ("construction_drive_left",),
    )

  def test_weak_construction_drive_choice_is_raw_only_not_control_active(self):
    state = RotatingScoreState(
      groups=(("construction_drive_left", "construction_drive_right"),),
      cache_ttl_frames=3,
      durable_labels=("construction_drive_left", "construction_drive_right"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_drive_left", "construction_drive_right"),
        ("construction_drive_left",),
        {"construction_drive_left": CONSTRUCTION_ACTION_IMMEDIATE_SCORE - 0.02, "construction_drive_right": CONSTRUCTION_ACTION_IMMEDIATE_SCORE - 1.13},
        26,
        vehicle_state_text="tracked_path_lat=0.00m",
      ),
      ("none",),
    )

  def test_construction_drive_action_suppressed_after_path_has_shifted_without_confidence(self):
    state = RotatingScoreState(
      groups=(("construction_drive_left", "construction_drive_right"),),
      cache_ttl_frames=3,
      durable_labels=("construction_drive_left", "construction_drive_right"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_drive_left", "construction_drive_right"),
        ("construction_drive_right",),
        {"construction_drive_left": 0.0, "construction_drive_right": 1.0},
        25,
        vehicle_state_text="tracked_path_lat=0.20m",
      ),
      ("none",),
    )

  def test_construction_drive_action_can_rescue_when_confident_and_path_toward_hazard(self):
    state = RotatingScoreState(
      groups=(("construction_drive_left", "construction_drive_right"),),
      cache_ttl_frames=3,
      durable_labels=("construction_drive_left", "construction_drive_right"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_drive_left", "construction_drive_right"),
        ("construction_drive_right",),
        {"construction_drive_left": 0.0, "construction_drive_right": 3.2},
        25,
        vehicle_state_text="tracked_path_lat=0.45m",
      ),
      ("construction_drive_right",),
    )

  def test_recent_construction_drive_action_vetoes_opposite_edge_bootstrap(self):
    state = RotatingScoreState(
      groups=(("construction_drive_left", "construction_drive_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("construction_drive_left", "construction_drive_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_drive_left", "construction_drive_right"),
        ("construction_drive_right",),
        {"construction_drive_left": 0.0, "construction_drive_right": 2.6},
        29,
        "tracked_path_lat=0.00m",
      ),
      ("construction_drive_right",),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 1.0, "construction_purple_edge": 3.2},
        36,
        "tracked_path_lat=0.00m",
      ),
      ("construction_drive_right",),
    )

  def test_recent_construction_drive_action_survives_short_cache_and_vetoes_opposite_edge_bootstrap(self):
    state = RotatingScoreState(
      groups=(("construction_drive_left", "construction_drive_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("construction_drive_left", "construction_drive_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_drive_left", "construction_drive_right"),
        ("construction_drive_right",),
        {"construction_drive_left": 0.08, "construction_drive_right": 2.66},
        25,
        "tracked_path_lat=0.00m",
      ),
      ("construction_drive_right",),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 1.80, "construction_purple_edge": 3.75},
        32,
        "tracked_path_lat=0.00m",
      ),
      ("construction_drive_right",),
    )

  def test_direct_construction_consensus_requires_presence(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_right",),
        {"construction_left": 0.32, "construction_right": 2.52},
        6,
        "tracked_path_lat=0.00m",
      ),
      ("none",),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": -0.14, "construction_purple_edge": 1.67},
        9,
        "tracked_path_lat=0.00m",
      ),
      ("none",),
    )
    self.assertEqual(
      state.update(
        ("cones", "barrier"),
        ("cones", "barrier"),
        {"cones": 0.68, "barrier": 0.03},
        13,
        "tracked_path_lat=0.00m",
      ),
      ("cones", "barrier"),
    )

  def test_weak_recent_construction_drive_action_does_not_veto_strong_opposite_edge(self):
    state = RotatingScoreState(
      groups=(("construction_drive_left", "construction_drive_right"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("construction_drive_left", "construction_drive_right", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_drive_left", "construction_drive_right"),
        ("construction_drive_right",),
        {"construction_drive_left": 0.12, "construction_drive_right": 0.77},
        25,
        "tracked_path_lat=0.00m",
      ),
      ("none",),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.88, "construction_purple_edge": 2.67},
        32,
        "tracked_path_lat=0.00m",
      ),
      ("construction_purple_edge",),
    )

  def test_construction_drive_action_continues_when_path_already_away_from_hazard(self):
    state = RotatingScoreState(
      groups=(("construction_drive_left", "construction_drive_right"), ("construction_left", "construction_right")),
      cache_ttl_frames=3,
      durable_labels=("construction_drive_left", "construction_drive_right", "construction_left", "construction_right"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_drive_left", "construction_drive_right"),
        ("construction_drive_left",),
        {"construction_drive_left": 3.59, "construction_drive_right": 2.25},
        27,
        f"tracked_path_lat={CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M + 0.04:.2f}m",
      ),
      ("construction_drive_left",),
    )
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_left",),
        {"construction_left": 2.94, "construction_right": 1.30},
        32,
        f"tracked_path_lat={CONSTRUCTION_ACTION_CONTINUE_MIN_TRACKED_OFFSET_M + 0.18:.2f}m",
      ),
      ("construction_drive_left",),
    )

  def test_recent_locked_construction_side_blocks_opposite_drive_near_neutral(self):
    state = RotatingScoreState(
      groups=(
        ("cones", "barrier"),
        ("construction_drive_left", "construction_drive_right"),
        ("construction_left", "construction_right"),
        ("construction_blue_edge", "construction_purple_edge"),
      ),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_drive_left", "construction_drive_right", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 2.1, "barrier": 0.0}, 24, "tracked_path_lat=0.00m")
    state.update(
      ("construction_drive_left", "construction_drive_right"),
      ("construction_drive_right",),
      {"construction_drive_left": 0.08, "construction_drive_right": 2.66},
      25,
      "tracked_path_lat=0.00m",
    )
    state.update(
      ("construction_left", "construction_right"),
      ("construction_left",),
      {"construction_left": 2.92, "construction_right": 0.12},
      29,
      "tracked_path_lat=-0.00m",
    )
    self.assertIn(
      "construction_blue_edge",
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": 1.95, "construction_purple_edge": 1.16},
        32,
        "tracked_path_lat=-0.01m",
      ),
    )
    state.update(
      ("construction_left", "construction_right"),
      ("construction_left",),
      {"construction_left": 2.92, "construction_right": 0.12},
      33,
      "tracked_path_lat=-0.02m",
    )

    # Fresh construction presence after a short side-label gap must not discard
    # the recent locked side while the avoidance maneuver is still developing.
    labels = state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.04, "barrier": 0.75}, 36, "tracked_path_lat=-0.07m")
    self.assertEqual(side_semantics.construction_hazard_side_from_labels(labels), "left")
    labels = state.update(
      ("construction_drive_left", "construction_drive_right"),
      ("construction_drive_left",),
      {"construction_drive_left": 2.98, "construction_drive_right": 1.31},
      37,
      "tracked_path_lat=-0.10m",
    )
    self.assertNotIn("construction_drive_left", labels)
    self.assertEqual(side_semantics.construction_hazard_side_from_labels(labels), "left")

  def test_recent_same_side_semantic_preserves_lock_on_presence_reacquire(self):
    state = RotatingScoreState(
      groups=(
        ("cones", "barrier"),
        ("construction_drive_left", "construction_drive_right"),
        ("construction_left", "construction_right"),
        ("construction_blue_edge", "construction_purple_edge"),
      ),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_drive_left", "construction_drive_right", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 1.2, "barrier": 0.6}, 36, "tracked_path_lat=0.00m")
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_blue_edge",),
      {"construction_blue_edge": 1.55, "construction_purple_edge": -1.36},
      40,
      "tracked_path_lat=0.00m",
    )
    state.update(
      ("construction_left", "construction_right"),
      ("construction_left",),
      {"construction_left": 2.83, "construction_right": -0.20},
      45,
      "tracked_path_lat=-0.00m",
    )
    labels = state.update(("cones", "barrier"), ("barrier",), {"cones": -0.42, "barrier": 1.08}, 48, "tracked_path_lat=-0.06m")
    self.assertEqual(side_semantics.construction_hazard_side_from_labels(labels), "left")
    labels = state.update(
      ("construction_drive_left", "construction_drive_right"),
      ("construction_drive_left",),
      {"construction_drive_left": 2.50, "construction_drive_right": 1.36},
      49,
      "tracked_path_lat=-0.09m",
    )
    self.assertNotIn("construction_drive_left", labels)
    self.assertEqual(side_semantics.construction_hazard_side_from_labels(labels), "left")

  def test_construction_semantic_side_can_bootstrap_while_path_neutral(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 2.2, "barrier": 0.0}, 30, "tracked_path_lat=0.00m")
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_left",),
        {
          "construction_left": CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE + 0.12,
          "construction_right": CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE - CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_MARGIN - 0.10,
        },
        31,
        "tracked_path_lat=0.00m",
      ),
      ("cones", "construction_left"),
    )

  def test_construction_semantic_side_does_not_bootstrap_after_path_moves(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 2.2, "barrier": 0.0}, 30, "tracked_path_lat=-0.40m")
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_left",),
        {
          "construction_left": CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE + 0.12,
          "construction_right": CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_SCORE - CONSTRUCTION_SEMANTIC_NEUTRAL_BOOTSTRAP_MARGIN - 0.10,
        },
        31,
        "tracked_path_lat=-0.40m",
      ),
      ("cones",),
    )

  def test_construction_color_edge_before_presence_does_not_arm_when_cones_arrive(self):
    state = RotatingScoreState(
      groups=(("construction_blue_edge", "construction_purple_edge"), ("cones", "barrier")),
      cache_ttl_frames=3,
      durable_labels=("construction_blue_edge", "construction_purple_edge", "cones", "barrier"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 0.8}, 10),
      ("none",),
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones",), {"cones": 1.0}, 11),
      ("cones",),
    )

  def test_construction_color_edge_after_presence_arms_and_survives_presence_refresh(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=5,
      durable_labels=("construction_blue_edge", "construction_purple_edge", "cones", "barrier"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones",), {"cones": 1.0}, 10),
      ("cones",),
    )
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 0.8}, 11),
      ("cones", "construction_blue_edge"),
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones",), {"cones": 1.2}, 12),
      ("cones", "construction_blue_edge"),
    )

  def test_construction_side_lock_rejects_single_late_contradiction(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10)
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 11),
      ("cones", "construction_purple_edge"),
    )
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 12)
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 13)
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 1.6, "construction_purple_edge": -1.6}, 14),
      ("cones", "construction_purple_edge"),
    )

  def test_construction_side_does_not_lock_on_first_observation(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10)
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 1.4, "construction_purple_edge": -1.4}, 11),
      ("cones", "construction_blue_edge"),
    )
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.6, "construction_purple_edge": 1.6}, 12),
      ("cones", "construction_purple_edge"),
    )

  def test_construction_weak_edge_observation_does_not_lock_wrong_side(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": -1.0}, 10)
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": 2.6, "construction_purple_edge": 2.2},
        11,
      ),
      ("cones", "construction_blue_edge"),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
        12,
      ),
      ("cones", "construction_purple_edge"),
    )

  def test_construction_strong_edge_lock_persists_through_presence_only_frames(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": -1.0}, 10)
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
        11,
      ),
      ("cones", "construction_purple_edge"),
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones",), {"cones": 3.7, "barrier": -1.0}, 13),
      ("cones", "construction_purple_edge"),
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones",), {"cones": 3.6, "barrier": -1.0}, 15),
      ("cones", "construction_purple_edge"),
    )

  def test_construction_presence_hold_survives_rotating_source_frame_gap(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": -1.0}, 23)
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
        26,
      ),
      ("cones", "construction_purple_edge"),
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones",), {"cones": 3.6, "barrier": -1.0}, 31),
      ("cones", "construction_purple_edge"),
    )

  def test_construction_edge_clear_requires_tracked_path_already_away(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": -1.0}, 10)
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
      11,
      "tracked_path_lat=0.00m",
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("none",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": 0.0},
        12,
        "tracked_path_lat=0.72m",
      ),
      ("cones", "construction_purple_edge"),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": 0.0},
        13,
        "tracked_path_lat=1.05m",
      ),
      ("cones",),
    )

  def test_strong_construction_edge_bootstraps_side_before_presence_group(self):
    state = RotatingScoreState(
      groups=(("construction_blue_edge", "construction_purple_edge"), ("cones", "barrier")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE + 0.2},
        10,
        "tracked_path_lat=0.00m",
      ),
      ("construction_purple_edge",),
    )
    self.assertEqual(
      state.update(
        ("cones", "barrier"),
        ("cones", "barrier"),
        {"cones": 0.0, "barrier": 0.0},
        12,
        "tracked_path_lat=0.00m",
      ),
      ("cones", "barrier", "construction_purple_edge"),
    )

  def test_near_bootstrap_construction_edge_seeds_side_before_presence_group(self):
    state = RotatingScoreState(
      groups=(("construction_blue_edge", "construction_purple_edge"), ("cones", "barrier")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE + 0.05,
          "construction_purple_edge": 0.0,
        },
        10,
        "tracked_path_lat=0.00m",
      ),
      ("construction_blue_edge",),
    )
    self.assertEqual(
      state.update(
        ("cones", "barrier"),
        ("cones", "barrier"),
        {"cones": 0.0, "barrier": 0.0},
        12,
        "tracked_path_lat=0.00m",
      ),
      ("cones", "barrier", "construction_blue_edge"),
    )

  def test_expired_construction_edge_bootstrap_does_not_seed_later_presence_group(self):
    state = RotatingScoreState(
      groups=(("construction_blue_edge", "construction_purple_edge"), ("cones", "barrier")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE + 0.2},
      10,
      "tracked_path_lat=0.00m",
    )
    self.assertEqual(
      state.update(
        ("cones", "barrier"),
        ("cones", "barrier"),
        {"cones": 0.0, "barrier": 0.0},
        20,
        "tracked_path_lat=0.00m",
      ),
      ("cones", "barrier"),
    )

  def test_weak_or_ambiguous_construction_edge_does_not_bootstrap_side(self):
    weak = RotatingScoreState(
      groups=(("construction_blue_edge", "construction_purple_edge"), ("cones", "barrier")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      weak.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE - 0.1},
        10,
        "tracked_path_lat=0.00m",
      ),
      ("none",),
    )

    ambiguous = RotatingScoreState(
      groups=(("construction_blue_edge", "construction_purple_edge"), ("cones", "barrier")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      ambiguous.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE - 0.1,
          "construction_purple_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE + 0.2,
        },
        10,
        "tracked_path_lat=0.00m",
      ),
      ("none",),
    )

  def test_construction_cleared_side_suppresses_weak_same_side_reactivation_while_shifted_away(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": -1.0}, 10)
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
      11,
      "tracked_path_lat=0.00m",
    )
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("none",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 0.0},
      12,
      "tracked_path_lat=1.05m",
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 2.3, "construction_purple_edge": 2.6},
        13,
        "tracked_path_lat=1.10m",
      ),
      ("cones",),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
        14,
        "tracked_path_lat=1.10m",
      ),
      ("cones",),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
        15,
        "tracked_path_lat=0.20m",
      ),
      ("cones", "construction_purple_edge"),
    )

  def test_construction_presence_clear_accepts_no_presence_once_path_shifted(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": 0.0}, 10, "tracked_path_lat=0.00m")
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
      11,
      "tracked_path_lat=0.00m",
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.0, "barrier": 0.0}, 14, "tracked_path_lat=1.05m"),
      ("none",),
    )
    self.assertTrue(state.construction_clear_active(14))

  def test_construction_presence_clear_does_not_fire_when_path_is_not_shifted_away(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": 0.0}, 10, "tracked_path_lat=0.00m")
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
      11,
      "tracked_path_lat=0.00m",
    )
    state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.0, "barrier": 0.0}, 14, "tracked_path_lat=0.45m")
    labels = state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.0, "barrier": 0.0}, 18, "tracked_path_lat=0.50m")
    self.assertIn("construction_purple_edge", labels)
    self.assertFalse(state.construction_clear_active(18))

  def test_construction_clear_suppresses_edge_only_reactivation_while_path_remains_shifted(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": 0.0}, 10, "tracked_path_lat=0.00m")
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
      11,
      "tracked_path_lat=0.00m",
    )
    state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.0, "barrier": 0.0}, 14, "tracked_path_lat=1.05m")
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.0, "barrier": 0.0}, 18, "tracked_path_lat=1.08m"),
      ("none",),
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("barrier",), {"cones": 0.0, "barrier": CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE - 0.4}, 19, "tracked_path_lat=1.08m"),
      ("barrier",),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": 2.6, "construction_purple_edge": 0.0},
        20,
        "tracked_path_lat=1.08m",
      ),
      ("barrier",),
    )

  def test_construction_clear_allows_reactivation_with_fresh_strong_presence(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": 0.0}, 10, "tracked_path_lat=0.00m")
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
      11,
      "tracked_path_lat=0.00m",
    )
    state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.0, "barrier": 0.0}, 14, "tracked_path_lat=1.05m")
    self.assertEqual(
      state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.0, "barrier": 0.0}, 18, "tracked_path_lat=1.08m"),
      ("none",),
    )
    self.assertEqual(
      state.update(("cones", "barrier"), ("barrier",), {"cones": 0.0, "barrier": CONSTRUCTION_REACTIVATE_MIN_PRESENCE_SCORE + 0.2}, 19, "tracked_path_lat=1.08m"),
      ("barrier",),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": 2.6, "construction_purple_edge": 0.0},
        20,
        "tracked_path_lat=1.08m",
      ),
      ("barrier", "construction_blue_edge"),
    )

  def test_construction_clear_suppresses_shift_only_reactivation_while_path_remains_shifted(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge"), ("construction_shift_left", "construction_shift_right")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge", "construction_shift_left", "construction_shift_right"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": 0.0}, 10, "tracked_path_lat=0.00m")
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
      11,
      "tracked_path_lat=0.00m",
    )
    state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.0, "barrier": 0.0}, 14, "tracked_path_lat=1.05m")
    state.update(("cones", "barrier"), ("cones", "barrier"), {"cones": 0.0, "barrier": 0.0}, 18, "tracked_path_lat=1.08m")
    self.assertEqual(
      state.update(
        ("construction_shift_left", "construction_shift_right"),
        ("construction_shift_left",),
        {"construction_shift_left": 3.0, "construction_shift_right": 0.0},
        19,
        "tracked_path_lat=1.08m",
      ),
      ("none",),
    )

  def test_construction_locked_side_suppresses_single_later_contradiction(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8, "barrier": -1.0}, 10)
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_purple_edge",),
      {"construction_blue_edge": 0.0, "construction_purple_edge": 2.2},
      11,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.7, "barrier": -1.0}, 13)
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": 2.7, "construction_purple_edge": 0.0},
        14,
      ),
      ("cones", "construction_purple_edge"),
    )

  def test_construction_side_lock_allows_repeated_or_high_confidence_override(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10)
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 11)
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 12)
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 13)
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 1.7, "construction_purple_edge": -1.7}, 14),
      ("cones", "construction_purple_edge"),
    )
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 1.8, "construction_purple_edge": -1.8}, 15),
      ("cones", "construction_purple_edge"),
    )
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 1.9, "construction_purple_edge": -1.9}, 16),
      ("cones", "construction_blue_edge"),
    )

    immediate = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    immediate.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10)
    immediate.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 11)
    immediate.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 12)
    immediate.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 13)
    self.assertEqual(
      immediate.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 3.2, "construction_purple_edge": -3.2}, 14),
      ("cones", "construction_blue_edge"),
    )

  def test_construction_side_lock_allows_confident_override_when_path_is_toward_new_hazard(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_drive_left", "construction_drive_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_drive_left", "construction_drive_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 11, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 12, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 13, "tracked_path_lat=0.00m")
    state.update(("construction_drive_left", "construction_drive_right"), ("construction_drive_right",), {"construction_drive_left": 0.0, "construction_drive_right": 3.2}, 14, "tracked_path_lat=1.10m")
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": 3.2, "construction_purple_edge": -3.2},
        15,
        "tracked_path_lat=1.10m",
      ),
      ("cones", "construction_blue_edge", "construction_drive_right"),
    )

  def test_construction_side_lock_blocks_confident_opposite_side_without_action_when_committed_away(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 11, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 12, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 13, "tracked_path_lat=0.00m")
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": 3.2, "construction_purple_edge": -3.2},
        14,
        "tracked_path_lat=1.10m",
      ),
      ("cones", "construction_purple_edge"),
    )

  def test_construction_side_lock_blocks_moderate_override_when_path_is_toward_new_hazard(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 11, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 12, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 13, "tracked_path_lat=0.00m")
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_SIDE_TOWARD_PATH_OVERRIDE_SCORE - 0.1,
          "construction_purple_edge": 0.0,
        },
        14,
        f"tracked_path_lat={CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M + 0.05:.2f}m",
      ),
      ("cones", "construction_purple_edge"),
    )

  def test_construction_side_lock_allows_early_high_confidence_reversal_before_path_moves(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10, "tracked_path_lat=0.00m")
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": 2.34},
        11,
        "tracked_path_lat=0.00m",
      ),
      ("cones", "construction_purple_edge"),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_SIDE_EARLY_REVERSAL_SCORE + 0.14,
          "construction_purple_edge": CONSTRUCTION_SIDE_EARLY_REVERSAL_OPPOSITE_MAX,
        },
        18,
        f"tracked_path_lat={CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_TRACKED_OFFSET_M - 0.02:.2f}m",
      ),
      ("cones", "construction_blue_edge"),
    )

  def test_construction_side_lock_blocks_late_opposite_edge_after_stable_avoidance(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10, "tracked_path_lat=0.00m")
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_blue_edge",),
      {"construction_blue_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE + 0.05, "construction_purple_edge": 0.0},
      11,
      "tracked_path_lat=0.00m",
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 19, "tracked_path_lat=-0.02m")
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 27, "tracked_path_lat=-0.18m")
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {
          "construction_blue_edge": 0.0,
          "construction_purple_edge": CONSTRUCTION_SIDE_EARLY_REVERSAL_SCORE + 0.12,
        },
        35,
        "tracked_path_lat=-0.32m",
      ),
      ("cones", "construction_blue_edge"),
    )

  def test_construction_side_lock_rescues_wrong_lock_when_path_is_toward_new_hazard(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_drive_left", "construction_drive_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_drive_left", "construction_drive_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 2.4}, 11, "tracked_path_lat=0.00m")
    state.update(("construction_drive_left", "construction_drive_right"), ("construction_drive_right",), {"construction_drive_left": 0.0, "construction_drive_right": 3.2}, 17, f"tracked_path_lat={CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M + 0.05:.2f}m")
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_SIDE_TOWARD_PATH_OVERRIDE_SCORE + 0.1,
          "construction_purple_edge": 0.0,
        },
        18,
        f"tracked_path_lat={CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M + 0.05:.2f}m",
      ),
      ("cones", "construction_blue_edge", "construction_drive_right"),
    )

  def test_construction_side_lock_still_allows_immediate_override_before_path_commits_away(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 11, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 12, "tracked_path_lat=0.00m")
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 13, "tracked_path_lat=0.00m")
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": 3.2, "construction_purple_edge": -3.2},
        14,
        "tracked_path_lat=0.20m",
      ),
      ("cones", "construction_blue_edge"),
    )

  def test_construction_side_lock_uses_canonical_side_for_shift_and_candidate_labels(self):
    state = RotatingScoreState(
      groups=(
        ("cones", "barrier"),
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_shift_left", "construction_shift_right"),
        ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
      ),
      cache_ttl_frames=8,
      durable_labels=(
        "cones",
        "barrier",
        "construction_blue_edge",
        "construction_purple_edge",
        "construction_shift_left",
        "construction_shift_right",
        "construction_blocks_left_candidate",
        "construction_blocks_right_candidate",
      ),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10)
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 11)
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 12)
    state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.4, "construction_purple_edge": 1.4}, 13)

    self.assertEqual(
      state.update(("construction_shift_left", "construction_shift_right"), ("construction_shift_right",), {"construction_shift_left": -1.4, "construction_shift_right": 1.4}, 14),
      ("cones", "construction_purple_edge"),
    )
    self.assertEqual(
      state.update(("construction_blocks_left_candidate", "construction_blocks_right_candidate"), ("construction_blocks_left_candidate",), {"construction_blocks_left_candidate": 1.4, "construction_blocks_right_candidate": -1.4}, 15),
      ("cones", "construction_purple_edge"),
    )
    self.assertEqual(
      state.update(("construction_shift_left", "construction_shift_right"), ("construction_shift_left",), {"construction_shift_left": 1.4, "construction_shift_right": -1.4}, 16),
      ("cones", "construction_purple_edge", "construction_shift_left"),
    )
    self.assertEqual(
      state.update(("construction_blocks_left_candidate", "construction_blocks_right_candidate"), ("construction_blocks_right_candidate",), {"construction_blocks_left_candidate": -1.4, "construction_blocks_right_candidate": 1.4}, 17),
      ("cones", "construction_purple_edge", "construction_shift_left", "construction_blocks_right_candidate"),
    )

  def test_construction_direct_consensus_overrides_stale_opposite_candidate(self):
    state = RotatingScoreState(
      groups=(
        ("cones", "barrier"),
        ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
        ("construction_left", "construction_right"),
        ("construction_blue_edge", "construction_purple_edge"),
      ),
      cache_ttl_frames=8,
      durable_labels=(
        "cones",
        "barrier",
        "construction_blocks_left_candidate",
        "construction_blocks_right_candidate",
        "construction_left",
        "construction_right",
        "construction_blue_edge",
        "construction_purple_edge",
      ),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(state.update(("cones", "barrier"), ("cones",), {"cones": 2.2, "barrier": -0.3}, 60), ("cones",))
    self.assertEqual(
      state.update(
        ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
        ("construction_blocks_left_candidate",),
        {"construction_blocks_left_candidate": 2.3, "construction_blocks_right_candidate": 0.0},
        61,
      ),
      ("cones", "construction_blocks_left_candidate"),
    )
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_right",),
        {"construction_left": 1.2, "construction_right": 2.4},
        62,
      ),
      ("cones", "construction_blocks_left_candidate"),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 1.4, "construction_purple_edge": 3.4},
        65,
      ),
      ("cones", "construction_right", "construction_purple_edge"),
    )
    self.assertNotIn("construction_blocks_left_candidate", state.active_labels(65))
    self.assertEqual(state.debug_state(65)["construction_direct_consensus_labels"], ["construction_right", "construction_purple_edge"])

  def test_construction_candidate_survives_without_direct_semantic_edge_consensus(self):
    state = RotatingScoreState(
      groups=(
        ("cones", "barrier"),
        ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
        ("construction_left", "construction_right"),
        ("construction_blue_edge", "construction_purple_edge"),
      ),
      cache_ttl_frames=8,
      durable_labels=(
        "cones",
        "barrier",
        "construction_blocks_left_candidate",
        "construction_blocks_right_candidate",
        "construction_left",
        "construction_right",
        "construction_blue_edge",
        "construction_purple_edge",
      ),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("barrier",), {"cones": -0.2, "barrier": 0.3}, 0)
    self.assertEqual(
      state.update(
        ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
        ("construction_blocks_left_candidate",),
        {"construction_blocks_left_candidate": 2.4, "construction_blocks_right_candidate": 0.0},
        1,
      ),
      ("barrier", "construction_blocks_left_candidate"),
    )
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_right",),
        {"construction_left": 0.0, "construction_right": 1.8},
        2,
      ),
      ("barrier", "construction_blocks_left_candidate"),
    )
    self.assertEqual(state.debug_state(2)["construction_direct_consensus_labels"], [])

  def test_construction_direct_consensus_requires_fresh_scores(self):
    state = RotatingScoreState(
      groups=(
        ("cones", "barrier"),
        ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
        ("construction_left", "construction_right"),
        ("construction_blue_edge", "construction_purple_edge"),
      ),
      cache_ttl_frames=8,
      durable_labels=(
        "cones",
        "barrier",
        "construction_blocks_left_candidate",
        "construction_blocks_right_candidate",
        "construction_left",
        "construction_right",
        "construction_blue_edge",
        "construction_purple_edge",
      ),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 2.2, "barrier": -0.3}, 0)
    state.update(
      ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
      ("construction_blocks_left_candidate",),
      {"construction_blocks_left_candidate": 2.3, "construction_blocks_right_candidate": 0.0},
      1,
    )
    state.update(
      ("construction_left", "construction_right"),
      ("construction_right",),
      {"construction_left": 0.0, "construction_right": CONSTRUCTION_DIRECT_SEMANTIC_SCORE + CONSTRUCTION_DIRECT_SEMANTIC_MARGIN + 0.1},
      2,
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.0, "construction_purple_edge": CONSTRUCTION_DIRECT_EDGE_SCORE + CONSTRUCTION_DIRECT_EDGE_MARGIN + 0.1},
        2 + CONSTRUCTION_DIRECT_CONSENSUS_MAX_AGE_FRAMES + 1,
      ),
      ("none",),
    )

  def test_construction_combined_side_waits_for_corridor_edge_label(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10), ("cones",))
    self.assertEqual(
      state.update(("construction_left", "construction_right"), ("construction_left",), {"construction_left": 1.4, "construction_right": -1.4}, 11),
      ("cones",),
    )
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 1.2, "construction_purple_edge": -1.2}, 12),
      ("cones", "construction_left", "construction_blue_edge"),
    )

  def test_construction_combined_side_suppresses_edge_when_semantic_disagrees(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10)
    state.update(("construction_left", "construction_right"), ("construction_left",), {"construction_left": 1.4, "construction_right": -1.4}, 11)
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_purple_edge",), {"construction_blue_edge": -1.2, "construction_purple_edge": 1.2}, 12),
      ("cones",),
    )
    self.assertEqual(
      state.update(("construction_left", "construction_right"), ("construction_right",), {"construction_left": -1.3, "construction_right": 1.3}, 13),
      ("cones", "construction_right", "construction_purple_edge"),
    )

  def test_construction_combined_side_suppresses_edge_bootstrap_without_semantic_corroboration(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE + 0.2, "construction_purple_edge": 0.0},
        8,
      ),
      ("none",),
    )

  def test_construction_combined_side_allows_strong_neutral_edge_bootstrap_before_presence(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE - CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN,
          "construction_purple_edge": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE + 0.1,
        },
        8,
        "tracked_path_lat=0.00m",
      ),
      ("construction_purple_edge",),
    )

  def test_construction_combined_side_blocks_neutral_edge_bootstrap_after_path_moves(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE - CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN,
          "construction_purple_edge": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE + 0.1,
        },
        8,
        f"tracked_path_lat={CONSTRUCTION_SIDE_EARLY_REVERSAL_MAX_TRACKED_OFFSET_M + 0.10:.2f}m",
      ),
      ("none",),
    )

  def test_construction_presence_hold_blocks_edge_bootstrap_after_score_cache_expiry(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(state.update(("cones", "barrier"), ("cones",), {"cones": 1.1, "barrier": -2.0}, 0), ("cones",))
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE + 0.4, "construction_purple_edge": 0.6},
        5,
        "tracked_path_lat=0.00m",
      ),
      ("cones",),
    )

  def test_construction_direct_consensus_uses_rotating_score_window_beyond_cache_ttl(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=3,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 1.1, "barrier": -2.0}, 30)
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_right",),
        {"construction_left": 0.2, "construction_right": CONSTRUCTION_DIRECT_SEMANTIC_SCORE + CONSTRUCTION_DIRECT_SEMANTIC_MARGIN + 0.2},
        31,
      ),
      ("cones",),
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {"construction_blue_edge": 0.2, "construction_purple_edge": CONSTRUCTION_DIRECT_EDGE_SCORE + CONSTRUCTION_DIRECT_EDGE_MARGIN + 0.2},
        37,
      ),
      ("cones", "construction_right", "construction_purple_edge"),
    )

  def test_construction_semantic_side_rescues_when_path_is_toward_hazard(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(
      ("construction_blue_edge", "construction_purple_edge"),
      ("construction_blue_edge",),
      {"construction_blue_edge": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE + 0.4, "construction_purple_edge": 0.6},
      5,
      "tracked_path_lat=0.00m",
    )
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_right",),
        {
          "construction_left": CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE - CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN,
          "construction_right": CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE + 1.1,
        },
        8,
        f"tracked_path_lat=-{CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M + 0.20:.2f}m",
      ),
      ("construction_right",),
    )

  def test_construction_edge_side_rescues_when_path_is_toward_hazard(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE - CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN,
          "construction_purple_edge": CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE + 0.1,
        },
        33,
        f"tracked_path_lat=-{CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M + 0.05:.2f}m",
      ),
      ("construction_purple_edge",),
    )

  def test_construction_edge_side_does_not_rescue_when_path_not_toward_hazard(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_purple_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE - CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN,
          "construction_purple_edge": CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE + 0.1,
        },
        33,
        f"tracked_path_lat={CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M + 0.05:.2f}m",
      ),
      ("none",),
    )

  def test_construction_semantic_side_does_not_rescue_when_path_not_toward_hazard(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_right",),
        {
          "construction_left": CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE - CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN,
          "construction_right": CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE + 1.1,
        },
        8,
        f"tracked_path_lat={CONSTRUCTION_SIDE_COMMITTED_AWAY_MIN_TRACKED_OFFSET_M + 0.20:.2f}m",
      ),
      ("none",),
    )

  def test_construction_edge_only_schedule_allows_strong_edge_bootstrap_before_presence(self):
    state = RotatingScoreState(
      groups=(("construction_blue_edge", "construction_purple_edge"),),
      cache_ttl_frames=8,
      durable_labels=("construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("construction_blue_edge", "construction_purple_edge"),
        ("construction_blue_edge",),
        {"construction_blue_edge": CONSTRUCTION_EDGE_BOOTSTRAP_SCORE + 0.2, "construction_purple_edge": 0.0},
        8,
      ),
      ("construction_blue_edge",),
    )

  def test_construction_combined_side_ignores_stale_edge_from_previous_episode(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right"), ("construction_blue_edge", "construction_purple_edge")),
      cache_ttl_frames=8,
      durable_labels=("cones", "barrier", "construction_left", "construction_right", "construction_blue_edge", "construction_purple_edge"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones", "barrier"), ("cones",), {"cones": 3.8}, 10)
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 1.2, "construction_purple_edge": -1.2}, 11),
      ("cones",),
    )
    self.assertEqual(state.update(("cones", "barrier"), ("cones",), {"cones": 3.7}, 22), ("cones",))
    self.assertEqual(
      state.update(("construction_left", "construction_right"), ("construction_left",), {"construction_left": 1.5, "construction_right": -1.5}, 22),
      ("cones",),
    )
    self.assertEqual(
      state.update(("construction_blue_edge", "construction_purple_edge"), ("construction_blue_edge",), {"construction_blue_edge": 1.3, "construction_purple_edge": -1.3}, 23),
      ("cones", "construction_left", "construction_blue_edge"),
    )

  def test_active_exclusive_construction_edges_resolve_by_score(self):
    state = RotatingScoreState(
      groups=(("construction_blue_edge",), ("construction_purple_edge",), ("cones",)),
      cache_ttl_frames=3,
      durable_labels=("construction_blue_edge", "construction_purple_edge", "cones"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones",), ("cones",), {"cones": 1.2}, 10)
    state.update(("construction_blue_edge",), ("construction_blue_edge",), {"construction_blue_edge": 1.4}, 11)
    self.assertEqual(state.active_labels(11), ("cones", "construction_blue_edge"))
    state.update(("construction_purple_edge",), ("construction_purple_edge",), {"construction_purple_edge": 3.2}, 12)
    self.assertEqual(state.active_labels(12), ("cones", "construction_purple_edge"))

  def test_active_exclusive_candidate_obstruction_overrides_stale_opposite_candidate(self):
    state = RotatingScoreState(
      groups=(("construction_blocks_left_candidate", "construction_blocks_right_candidate"), ("cones",)),
      cache_ttl_frames=8,
      durable_labels=("construction_blocks_left_candidate", "construction_blocks_right_candidate", "cones"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones",), ("cones",), {"cones": 3.8}, 10)
    self.assertEqual(
      state.update(
        ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
        ("construction_blocks_left_candidate",),
        {"construction_blocks_left_candidate": 4.0, "construction_blocks_right_candidate": 1.0},
        11,
      ),
      ("cones", "construction_blocks_left_candidate"),
    )
    self.assertEqual(
      state.update(
        ("construction_blocks_left_candidate", "construction_blocks_right_candidate"),
        ("construction_blocks_right_candidate",),
        {"construction_blocks_left_candidate": 1.0, "construction_blocks_right_candidate": 4.0},
        12,
      ),
      ("cones", "construction_blocks_right_candidate"),
    )

  def test_active_scores_keep_recent_exclusive_competitor_for_calibration(self):
    state = RotatingScoreState(
      groups=(("construction_blue_edge",), ("construction_purple_edge",), ("cones",)),
      cache_ttl_frames=3,
      durable_labels=("construction_blue_edge", "construction_purple_edge", "cones"),
      negative_clear_threshold=2.0,
    )
    state.update(("cones",), ("cones",), {"cones": 3.8}, 10)
    state.update(("construction_blue_edge",), ("construction_blue_edge",), {"construction_blue_edge": 1.4}, 11)
    state.update(("construction_purple_edge",), ("none",), {"construction_purple_edge": -0.3}, 12)

    self.assertEqual(state.active_labels(12), ("cones", "construction_blue_edge"))
    scores = state.active_scores(12)
    self.assertEqual(scores["construction_blue_edge"], 1.4)
    self.assertEqual(scores["construction_purple_edge"], -0.3)
    self.assertEqual(scores["cones"], 3.8)

  def test_construction_confidence_requires_fresh_competitor_score(self):
    self.assertEqual(
      _score_calibrated_construction_confidence(("construction_blue_edge",), {"construction_blue_edge": 2.0}),
      0.72,
    )
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_blue_edge",),
        {"construction_blue_edge": 2.0, "construction_purple_edge": 0.4},
      ),
      0.84,
    )
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_purple_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE - CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_MARGIN,
          "construction_purple_edge": CONSTRUCTION_EDGE_NEUTRAL_BOOTSTRAP_SCORE + 0.1,
        },
        strong_edge_immediate=True,
      ),
      0.96,
    )
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_right",),
        {
          "construction_left": CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE - CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_MARGIN,
          "construction_right": CONSTRUCTION_SEMANTIC_TOWARD_PATH_OVERRIDE_SCORE + 0.2,
        },
      ),
      0.96,
    )
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_purple_edge",),
        {
          "construction_blue_edge": CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE - CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_MARGIN,
          "construction_purple_edge": CONSTRUCTION_EDGE_TOWARD_PATH_OVERRIDE_SCORE + 0.1,
        },
        strong_edge_immediate=True,
      ),
      0.96,
    )
    self.assertEqual(
      _score_calibrated_construction_confidence(
        ("construction_right", "construction_purple_edge"),
        {
          "construction_left": 0.5,
          "construction_right": 1.3,
          "construction_blue_edge": 0.2,
          "construction_purple_edge": 3.1,
        },
      ),
      0.96,
    )

  def test_path_conflict_label_survives_moderate_negative_when_durable(self):
    state = RotatingScoreState(
      groups=(("pedestrian_in_path", "pedestrian_entering_path"),),
      cache_ttl_frames=60,
      durable_labels=("cones", "barrier", "pedestrian_in_path", "pedestrian_entering_path"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("pedestrian_in_path",), {"pedestrian_in_path": 0.5}, 10),
      ("pedestrian_in_path",),
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("none",), {"pedestrian_in_path": -0.1}, 11),
      ("pedestrian_in_path",),
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("none",), {"pedestrian_in_path": -2.1}, 12),
      ("none",),
    )

  def test_non_durable_label_can_still_clear_immediately(self):
    state = RotatingScoreState(
      groups=(("pedestrian_in_path", "pedestrian_entering_path"),),
      cache_ttl_frames=60,
      durable_labels=(),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("pedestrian_in_path",), {"pedestrian_in_path": 0.5}, 10),
      ("pedestrian_in_path",),
    )
    self.assertEqual(
      state.update(("pedestrian_in_path", "pedestrian_entering_path"), ("none",), {"pedestrian_in_path": -0.1}, 11),
      ("none",),
    )

  def test_mutually_exclusive_construction_side_keeps_higher_score(self):
    state = RotatingScoreState(
      groups=(("cones", "barrier"), ("construction_left", "construction_right")),
      cache_ttl_frames=60,
      durable_labels=("cones", "barrier", "construction_left", "construction_right"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(state.update(("cones", "barrier"), ("cones",), {"cones": 1.0}, 9), ("cones",))
    self.assertEqual(
      state.update(
        ("construction_left", "construction_right"),
        ("construction_left", "construction_right"),
        {"construction_left": 0.2, "construction_right": 1.0},
        10,
      ),
      ("cones", "construction_right"),
    )

  def test_green_light_replaces_stale_red_light(self):
    state = RotatingScoreState(
      groups=(("red_stop_light", "green_go_light"),),
      cache_ttl_frames=60,
      durable_labels=("red_stop_light", "green_go_light"),
      negative_clear_threshold=2.0,
    )
    self.assertEqual(
      state.update(
        ("red_stop_light", "green_go_light"),
        ("red_stop_light",),
        {"red_stop_light": 1.2, "green_go_light": -1.0},
        10,
      ),
      ("red_stop_light",),
    )
    self.assertEqual(
      state.update(
        ("red_stop_light", "green_go_light"),
        ("green_go_light",),
        {"red_stop_light": -0.2, "green_go_light": 1.4},
        11,
      ),
      ("green_go_light",),
    )


class TestVisualSignalFallback(unittest.TestCase):
  def test_visual_fallbacks_are_disabled_by_default(self):
    image = Image.new("RGB", (512, 384), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((250, 35, 320, 150), fill=(45, 45, 45), outline=(245, 245, 245), width=4)
    draw.ellipse((270, 108, 300, 138), fill=(20, 225, 60))
    self.assertEqual(_with_visual_fallbacks(image, ("red_stop_light",)), ("red_stop_light",))

  def test_visual_green_signal_overrides_stale_red_label(self):
    image = Image.new("RGB", (512, 384), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((250, 35, 320, 150), fill=(45, 45, 45), outline=(245, 245, 245), width=4)
    draw.ellipse((270, 108, 300, 138), fill=(20, 225, 60))
    self.assertEqual(_with_visual_fallbacks(image, ("red_stop_light",), enable_signal=True), ("green_go_light",))

  def test_visual_red_signal_overrides_stale_green_label(self):
    image = Image.new("RGB", (512, 384), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((250, 35, 320, 150), fill=(45, 45, 45), outline=(245, 245, 245), width=4)
    draw.ellipse((270, 42, 300, 72), fill=(235, 20, 20))
    self.assertEqual(_with_visual_fallbacks(image, ("green_go_light",), enable_signal=True), ("red_stop_light",))

  def test_visual_construction_right_side_overrides_none(self):
    image = Image.new("RGB", (512, 384), (120, 120, 120))
    draw = ImageDraw.Draw(image)
    draw.rectangle((210, 180, 285, 360), fill=(40, 170, 80))
    draw.polygon(((382, 280), (422, 280), (446, 360), (358, 360)), fill=(20, 85, 210))
    self.assertEqual(_with_visual_fallbacks(image, ("none",), enable_construction=True), ("construction_right",))

  def test_visual_construction_left_side_overrides_none(self):
    image = Image.new("RGB", (512, 384), (120, 120, 120))
    draw = ImageDraw.Draw(image)
    draw.rectangle((225, 180, 300, 360), fill=(40, 170, 80))
    draw.polygon(((92, 280), (132, 280), (156, 360), (68, 360)), fill=(20, 85, 210))
    self.assertEqual(_with_visual_fallbacks(image, ("none",), enable_construction=True), ("construction_left",))


class TestLeadTraceEvaluation(unittest.TestCase):
  def _record(
    self,
    *,
    lead_present=1,
    distance=24.0,
    lateral=0.0,
    lead_speed=5.0,
    rel_speed=0.0,
    closing=0.0,
    accel=0.0,
    lateral_velocity=0.0,
    lead_class="",
    target_speed=5.0,
    frame_id=0,
    route_heading=0.0,
    actual_heading=0.0,
  ):
    heading_error = route_vehicle_heading_error_rad(route_heading, actual_heading)
    heading_alignment = route_vehicle_heading_alignment_cos(route_heading, actual_heading)
    return {
      "frame_id": frame_id,
      "lead_present": lead_present,
      "lead_distance_m": distance,
      "lead_lateral_m": lateral,
      "lead_speed_mps": lead_speed,
      "lead_rel_speed_mps": rel_speed,
      "lead_closing_mps": closing,
      "lead_accel_mps2": accel,
      "lead_lateral_velocity_mps": lateral_velocity,
      "lead_route_heading_theta": route_heading,
      "lead_actual_heading_theta": actual_heading,
      "lead_heading_error_rad": heading_error,
      "lead_heading_alignment_cos": heading_alignment,
      "lead_heading_same_direction": int(route_vehicle_heading_same_direction(route_heading, actual_heading)),
      "lead_class": lead_class,
      "target_speed_mps": target_speed,
      "control_consumed_age_frames": 2,
      "info_flags": {},
    }

  def test_physical_track_classifies_core_lead_cases_without_sim_expected_label(self):
    self.assertEqual(lead_requirement_from_record(self._record(rel_speed=0.1, closing=-0.1))["expected_class"], "true_moving_lead")
    self.assertEqual(lead_requirement_from_record(self._record(lead_speed=4.2, rel_speed=-0.8, closing=0.8))["expected_class"], "slower_lead")
    self.assertEqual(lead_requirement_from_record(self._record(lead_speed=3.5, rel_speed=-1.5, closing=1.5))["expected_class"], "braking_lead")
    self.assertEqual(lead_requirement_from_record(self._record(lead_speed=0.0, rel_speed=-5.0, closing=5.0))["expected_class"], "stopped_lead")
    self.assertEqual(lead_requirement_from_record(self._record(lateral=2.0, lateral_velocity=-0.8, lead_speed=3.0))["expected_class"], "cut_in_vehicle")
    self.assertEqual(lead_requirement_from_record(self._record(lateral=-2.0, lateral_velocity=1.0, lead_speed=0.4))["expected_class"], "crossing_vehicle")
    self.assertEqual(lead_requirement_from_record(self._record(lateral=2.0, lateral_velocity=0.0, lead_speed=5.0))["expected_class"], "irrelevant_vehicle")

  def test_qwen_lead_class_normalizes_program_evidence_names(self):
    self.assertEqual(qwen_lead_class({"lead_class": "slower_lead_closing"}), "slower_lead")
    self.assertEqual(qwen_lead_class({"lead_class": "stopped_lead_in_path"}), "stopped_lead")
    self.assertEqual(qwen_lead_class({"qwen_labels": ["cut_in_vehicle"]}), "cut_in_vehicle")

  def test_lead_trace_evaluator_scores_required_and_false_slow_frames(self):
    episode = {
      "records": [
        self._record(frame_id=0, rel_speed=0.1, closing=-0.1, lead_class="true_moving_lead", target_speed=5.0),
        self._record(frame_id=1, lead_speed=4.2, rel_speed=-0.8, closing=0.8, lead_class="slower_lead_closing", target_speed=4.0),
        self._record(frame_id=2, lateral=2.0, lateral_velocity=0.0, lead_speed=5.0, lead_class="irrelevant_vehicle", target_speed=4.0),
      ]
    }
    result = evaluate_lead_episode(episode, max_allowed_age_frames=3)

    self.assertEqual(result["required_lead_frames"], 1)
    self.assertEqual(result["required_qwen_correct"], 1)
    self.assertEqual(result["required_control_ok"], 1)
    self.assertEqual(result["no_slow_frames"], 2)
    self.assertEqual(result["false_slow_frames"], 1)
    self.assertEqual(result["max_consumed_age_frames"], 2)
    self.assertEqual(result["age_violation_count"], 0)
    self.assertEqual(result["vehicle_heading_checked_frames"], 3)
    self.assertEqual(result["vehicle_heading_missing_frames"], 0)
    self.assertEqual(result["vehicle_heading_violation_count"], 0)
    self.assertEqual(result["per_class"]["slower_lead"]["qwen_success_rate"], 1.0)
    self.assertEqual(result["per_class"]["irrelevant_vehicle"]["false_slow_rate"], 1.0)

  def test_lead_suite_gate_scores_required_and_no_slow_cases(self):
    cases = {
      "true_moving_lead": {"nominal_speed_mps": 5.0, "records": [self._record(rel_speed=0.1, closing=-0.1, lead_class="true_moving_lead", target_speed=5.0)]},
      "slower_lead": {"nominal_speed_mps": 5.0, "records": [self._record(lead_speed=4.2, rel_speed=-0.8, closing=0.8, lead_class="slower_lead", target_speed=4.0)]},
      "cut_in_vehicle": {
        "nominal_speed_mps": 5.0,
        "records": [
          {
            **self._record(lateral=2.0, lateral_velocity=-0.8, lead_speed=3.0, lead_class="cut_in_vehicle", target_speed=5.0),
            "min_vehicle_route_clearance_m": 3.0,
          }
        ],
      },
      "irrelevant_vehicle": {"nominal_speed_mps": 5.0, "records": [self._record(lateral=2.0, lateral_velocity=0.0, lead_speed=5.0, lead_class="irrelevant_vehicle", target_speed=5.0)]},
    }
    result = evaluate_lead_suite(
      cases,
      required_cases=("true_moving_lead", "slower_lead", "cut_in_vehicle", "irrelevant_vehicle"),
      max_allowed_age_frames=3,
    )

    self.assertTrue(result["ok"], result["issues"])
    self.assertEqual(result["cases"]["slower_lead"]["per_class"]["slower_lead"]["qwen_success_rate"], 1.0)
    self.assertEqual(result["cases"]["cut_in_vehicle"]["per_class"]["cut_in_vehicle"]["control_success_rate"], 1.0)
    self.assertEqual(result["cases"]["true_moving_lead"]["per_class"]["true_moving_lead"]["false_slow_rate"], 0.0)
    self.assertEqual(result["cases"]["irrelevant_vehicle"]["per_class"]["irrelevant_vehicle"]["false_slow_rate"], 0.0)

  def test_lead_suite_gate_fails_missing_case_and_false_slow(self):
    cases = {
      "true_moving_lead": {"nominal_speed_mps": 5.0, "records": [self._record(rel_speed=0.1, closing=-0.1, lead_class="true_moving_lead", target_speed=4.0)]},
    }
    result = evaluate_lead_suite(
      cases,
      required_cases=("true_moving_lead", "slower_lead"),
      max_allowed_age_frames=3,
    )

    self.assertFalse(result["ok"])
    self.assertIn("true_moving_lead: false_slow_rate 1.0 > 0.05", result["issues"])
    self.assertIn("true_moving_lead: episode false_slow_rate 1.0 > 0.05", result["issues"])
    self.assertIn("slower_lead: missing episode", result["issues"])

  def test_lead_suite_gate_fails_no_slow_case_misclassification(self):
    cases = {
      "true_moving_lead": {"nominal_speed_mps": 5.0, "records": [self._record(rel_speed=0.1, closing=-0.1, lead_class="", target_speed=5.0)]},
    }
    result = evaluate_lead_suite(
      cases,
      required_cases=("true_moving_lead",),
      max_allowed_age_frames=3,
    )

    self.assertFalse(result["ok"])
    self.assertIn("true_moving_lead: qwen_success_rate 0.0 < 0.95", result["issues"])
    self.assertEqual(result["cases"]["true_moving_lead"]["per_class"]["true_moving_lead"]["false_slow_rate"], 0.0)

  def test_lead_suite_gate_fails_episode_level_false_slow_inside_required_case(self):
    cases = {
      "cut_in_vehicle": {
        "nominal_speed_mps": 5.0,
        "records": [
          {
            **self._record(frame_id=0, lateral=2.0, lateral_velocity=-0.8, lead_speed=3.0, lead_class="cut_in_vehicle", target_speed=5.0),
            "min_vehicle_route_clearance_m": 3.0,
          },
          self._record(frame_id=1, rel_speed=0.1, closing=-0.1, lead_class="true_moving_lead", target_speed=4.0),
        ],
      },
    }
    result = evaluate_lead_suite(
      cases,
      required_cases=("cut_in_vehicle",),
      max_allowed_age_frames=3,
    )

    self.assertFalse(result["ok"])
    self.assertIn("cut_in_vehicle: episode false_slow_rate 1.0 > 0.05", result["issues"])

  def test_lead_suite_gate_fails_required_frames_misclassified_inside_required_case(self):
    cases = {
      "cut_in_vehicle": {
        "nominal_speed_mps": 5.0,
        "records": [
          {
            **self._record(frame_id=0, lateral=2.0, lateral_velocity=-0.8, lead_speed=3.0, lead_class="cut_in_vehicle", target_speed=4.0),
            "min_vehicle_route_clearance_m": 3.0,
          },
          self._record(frame_id=1, lead_speed=3.0, rel_speed=-0.3, closing=0.3, lead_class="true_moving_lead", target_speed=5.0),
        ],
      },
    }
    result = evaluate_lead_suite(
      cases,
      required_cases=("cut_in_vehicle",),
      max_allowed_age_frames=3,
    )

    self.assertFalse(result["ok"])
    self.assertIn("cut_in_vehicle: required_qwen_success_rate 0.5 < 0.95", result["issues"])
    self.assertIn("cut_in_vehicle: required_control_success_rate 0.5 < 0.95", result["issues"])

  def test_lead_suite_gate_fails_head_on_vehicle_heading(self):
    cases = {
      "true_moving_lead": {
        "nominal_speed_mps": 5.0,
        "records": [
          self._record(
            rel_speed=0.1,
            closing=-0.1,
            lead_class="true_moving_lead",
            target_speed=5.0,
            route_heading=0.0,
            actual_heading=np.pi,
          )
        ],
      },
    }
    result = evaluate_lead_suite(
      cases,
      required_cases=("true_moving_lead",),
      max_allowed_age_frames=3,
    )

    self.assertFalse(result["ok"])
    self.assertIn("true_moving_lead: vehicle_heading_violation_count 1 != 0", result["issues"])


class TestDurableSpeedPlans(unittest.TestCase):
  def test_entering_agent_percent_speed_cap_uses_desired_speed(self):
    program = parse_rtp(_labels_to_rtp(("pedestrian_entering_path",)))
    self.assertEqual(program.scene, "path_conflict_agent")
    self.assertEqual(program.meta, "YIELD")
    self.assertAlmostEqual(program.speed_scale, 0.50)
    self.assertIsNone(program.stop_s)
    args = Namespace(
      speed_mps=12.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=30.0,
      durable_speed_recover_m=10.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
    )
    plan = durable_speed_plan_from_program(program, current_long_m=0.0, args=args)
    self.assertIsNotNone(plan)
    self.assertAlmostEqual(plan.speed_cap_mps, 6.0)

  def test_in_path_agent_creates_stop_plan(self):
    program = parse_rtp(_labels_to_rtp(("pedestrian_in_path",)))
    self.assertEqual(program.scene, "path_blocking_agent")
    self.assertEqual(program.meta, "STOP")
    self.assertEqual(program.speed_cap_mps, 0.0)
    self.assertAlmostEqual(program.stop_s, 18.0)
    self.assertIn("corridor_object_s18_28", program.avoid)
    args = Namespace(
      speed_mps=12.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=30.0,
      durable_speed_recover_m=10.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
    )
    plan = durable_speed_plan_from_program(program, current_long_m=0.0, args=args)
    self.assertIsNotNone(plan)
    self.assertEqual(plan.source_token, "corridor_object_s18_28")
    self.assertEqual(plan.speed_cap_mps, 0.0)
    self.assertAlmostEqual(plan.stop_s, 18.0)

  def test_in_path_agent_stop_plan_approaches_before_stopping(self):
    program = parse_rtp(_labels_to_rtp(("pedestrian_in_path",)))
    args = Namespace(
      speed_mps=2.5,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=10.0,
      durable_speed_recover_m=3.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
    )
    plan = durable_speed_plan_from_program(program, current_long_m=5.0, args=args)
    self.assertIsNotNone(plan)
    self.assertAlmostEqual(plan.target_speed_cap(5.0, args.speed_mps), 2.5)
    self.assertGreater(plan.target_speed_cap(18.0, args.speed_mps), 0.0)
    self.assertLess(plan.target_speed_cap(18.0, args.speed_mps), 2.5)
    self.assertEqual(plan.target_speed_cap(19.1, args.speed_mps), 0.0)

  def test_same_source_new_speed_cap_overrides_stale_stop(self):
    old = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=0.0,
      stop_s=18.0,
      source_token="corridor_object_s18_28",
      source_meta="YIELD",
      confidence=0.70,
    )
    new = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=1.5,
      stop_s=None,
      source_token="corridor_object_s18_28",
      source_meta="mixed_agent_construction",
      confidence=0.72,
    )
    merged = _merge_durable_speed_plan(old, new)
    self.assertAlmostEqual(merged.speed_cap_mps, 1.5)
    self.assertIsNone(merged.stop_s)
    self.assertEqual(merged.source_meta, "mixed_agent_construction")

  def test_green_light_base_program_clears_stale_stop_plan(self):
    args = Namespace(
      speed_mps=2.5,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=30.0,
      durable_speed_recover_m=10.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
    )
    old = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=0.0,
      stop_s=18.0,
      source_token="stop_line_s18",
      source_meta="STOP",
      confidence=0.74,
    )
    green_program = parse_rtp(_labels_to_rtp(("green_go_light",)))
    updated = update_durable_speed_plans(
      {"stop_line_s18": old},
      None,
      green_program,
      current_long_m=12.0,
      args=args,
    )
    self.assertEqual(updated, {})

  def test_green_signal_clear_uses_signal_threshold_not_lateral_conflict_threshold(self):
    args = Namespace(
      speed_mps=2.5,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=30.0,
      durable_speed_recover_m=10.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.90,
      durable_signal_clear_confidence=0.80,
    )
    old = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=0.0,
      stop_s=18.0,
      source_token="traffic_light_stop",
      source_meta="STOP",
      confidence=0.74,
    )
    green_program = parse_rtp(_labels_to_rtp(("green_go_light",)))
    self.assertLess(green_program.confidence, args.durable_conflict_override_confidence)
    self.assertGreaterEqual(green_program.confidence, args.durable_signal_clear_confidence)
    updated = update_durable_speed_plans(
      {"traffic_light_stop": old},
      None,
      green_program,
      current_long_m=12.0,
      args=args,
    )
    self.assertEqual(updated, {})

  def test_green_signal_in_mixed_program_clears_stale_signal_stop(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=30.0,
      durable_speed_recover_m=10.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.70,
    )
    old_stop = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=0.0,
      stop_s=18.0,
      source_token="stop_line_s18.0",
      source_meta="STOP",
      confidence=0.74,
    )
    mixed_green_program = parse_rtp(_labels_to_rtp(("green_go_light", "pedestrian_entering_path")))
    new_plan = durable_speed_plan_from_program(mixed_green_program, current_long_m=12.0, args=args)
    updated = update_durable_speed_plans(
      {"stop_line_s18.0": old_stop},
      new_plan,
      mixed_green_program,
      current_long_m=12.0,
      args=args,
    )
    self.assertNotIn("stop_line_s18.0", updated)
    self.assertIn("corridor_object_s18_28", updated)
    self.assertGreater(updated["corridor_object_s18_28"].speed_cap_mps, 0.0)
    self.assertIsNone(updated["corridor_object_s18_28"].stop_s)

  def test_green_signal_with_blocking_agent_clears_stale_signal_stop_but_keeps_agent_stop(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=30.0,
      durable_speed_recover_m=10.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.70,
    )
    old_stop = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=0.0,
      stop_s=18.0,
      source_token="stop_line_s18.0",
      source_meta="STOP",
      confidence=0.74,
    )
    mixed_green_program = parse_rtp(_labels_to_rtp(("green_go_light", "pedestrian_in_path")))
    self.assertEqual(mixed_green_program.meta, "STOP")
    self.assertIn("green_signal_for_path", mixed_green_program.evidence)
    new_plan = durable_speed_plan_from_program(mixed_green_program, current_long_m=12.0, args=args)
    updated = update_durable_speed_plans(
      {"stop_line_s18.0": old_stop},
      new_plan,
      mixed_green_program,
      current_long_m=12.0,
      args=args,
    )
    self.assertNotIn("stop_line_s18.0", updated)
    self.assertIn("corridor_object_s18_28", updated)
    self.assertEqual(updated["corridor_object_s18_28"].speed_cap_mps, 0.0)
    self.assertAlmostEqual(updated["corridor_object_s18_28"].stop_s, 18.0)

  def test_current_construction_program_clears_stale_agent_speed_cap(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=30.0,
      durable_speed_recover_m=10.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.70,
    )
    stale_agent = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=1.25,
      stop_s=None,
      source_token="corridor_object_s18_28",
      source_meta="YIELD",
      confidence=0.70,
    )
    construction_program = parse_rtp(_labels_to_rtp(("construction_left",)))
    updated = update_durable_speed_plans(
      {"corridor_object_s18_28": stale_agent},
      None,
      construction_program,
      current_long_m=12.0,
      args=args,
    )
    self.assertEqual(updated, {})

  def test_current_agent_program_preserves_agent_speed_cap(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=30.0,
      durable_speed_recover_m=10.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.70,
    )
    stale_agent = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=1.25,
      stop_s=None,
      source_token="corridor_object_s18_28",
      source_meta="YIELD",
      confidence=0.70,
    )
    agent_program = parse_rtp(_labels_to_rtp(("pedestrian_in_path",)))
    updated = update_durable_speed_plans(
      {"corridor_object_s18_28": stale_agent},
      None,
      agent_program,
      current_long_m=12.0,
      args=args,
    )
    self.assertIn("corridor_object_s18_28", updated)

  def test_physical_no_lead_track_clears_only_stale_lead_speed_plans(self):
    args = Namespace()
    stale_lead = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=45.0,
      ramp_out_end_long_m=50.0,
      speed_cap_mps=2.25,
      stop_s=None,
      source_token="lead_vehicle_s8_45",
      source_meta="YIELD",
      confidence=0.74,
    )
    crossing = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=35.0,
      speed_cap_mps=2.5,
      stop_s=None,
      source_token="crossing_vehicle_s8_30",
      source_meta="YIELD",
      confidence=0.73,
    )

    lead_state = {"lead_present": 0}
    self.assertEqual(_physical_lead_clear_reason(lead_state, args, allow_true_moving_clear=True), "no_lead_track")
    updated = _apply_current_lead_state_guard(
      {"lead_vehicle_s8_45": stale_lead, "crossing_vehicle_s8_30": crossing},
      lead_state,
      args,
      allow_true_moving_clear=False,
    )
    self.assertEqual(tuple(updated), ("crossing_vehicle_s8_30",))

  def test_physical_true_moving_lead_clears_stale_lead_slow_between_vlm_updates(self):
    args = Namespace(
      lead_clear_path_lateral_m=1.35,
      lead_clear_true_moving_closing_mps=0.35,
      lead_clear_true_moving_rel_loss_mps=0.35,
      lead_clear_non_braking_accel_mps2=0.60,
    )
    stale_lead = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=45.0,
      ramp_out_end_long_m=50.0,
      speed_cap_mps=2.25,
      stop_s=None,
      source_token="lead_vehicle_s8_45",
      source_meta="YIELD",
      confidence=0.74,
    )
    true_moving_state = {
      "lead_present": 1,
      "lead_distance_m": 22.0,
      "lead_lateral_m": 0.15,
      "lead_rel_speed_mps": 0.05,
      "lead_closing_mps": -0.05,
      "lead_accel_mps2": 0.0,
    }

    self.assertEqual(_physical_lead_clear_reason(true_moving_state, args, allow_true_moving_clear=True), "true_moving_or_opening_lead")
    self.assertEqual(
      _apply_current_lead_state_guard({"lead_vehicle_s8_45": stale_lead}, true_moving_state, args, allow_true_moving_clear=True),
      {},
    )
    self.assertEqual(
      _apply_current_lead_state_guard({"lead_vehicle_s8_45": stale_lead}, true_moving_state, args, allow_true_moving_clear=False),
      {"lead_vehicle_s8_45": stale_lead},
    )

  def test_physical_braking_or_closing_lead_does_not_clear_lead_speed_plan(self):
    args = Namespace(
      lead_clear_path_lateral_m=1.35,
      lead_clear_true_moving_closing_mps=0.35,
      lead_clear_true_moving_rel_loss_mps=0.35,
      lead_clear_non_braking_accel_mps2=0.60,
    )
    stale_lead = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=45.0,
      ramp_out_end_long_m=50.0,
      speed_cap_mps=2.25,
      stop_s=None,
      source_token="lead_vehicle_s8_45",
      source_meta="YIELD",
      confidence=0.74,
    )
    braking_state = {
      "lead_present": 1,
      "lead_distance_m": 18.0,
      "lead_lateral_m": 0.10,
      "lead_rel_speed_mps": -1.2,
      "lead_closing_mps": 1.2,
      "lead_accel_mps2": -1.0,
    }

    self.assertEqual(_physical_lead_clear_reason(braking_state, args, allow_true_moving_clear=True), "")
    self.assertEqual(
      _apply_current_lead_state_guard({"lead_vehicle_s8_45": stale_lead}, braking_state, args, allow_true_moving_clear=True),
      {"lead_vehicle_s8_45": stale_lead},
    )

  def test_red_signal_stop_plan_overrides_stale_slow_caps(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=10.0,
      durable_speed_recover_m=3.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
      include_traffic_light=True,
      novel_scene="random_mixed",
      traffic_light_route_s=48.0,
      traffic_light_stop_before_m=5.0,
      traffic_light_full_stop_m=4.0,
      traffic_light_decel_distance_m=28.0,
      traffic_light_stop_hold_radius_m=0.75,
      traffic_light_comfort_decel_mps2=1.1,
    )
    stale_slow = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=30.0,
      ramp_out_end_long_m=40.0,
      speed_cap_mps=1.25,
      stop_s=None,
      source_token="corridor_object_s18_28",
      source_meta="YIELD",
      confidence=0.70,
    )
    red_program = parse_rtp(_labels_to_rtp(("red_stop_light",)))
    new_plan = durable_speed_plan_from_program(red_program, current_long_m=12.0, args=args)
    new_plan = _adjust_signal_speed_plan(new_plan, red_program, current_long_m=12.0, args=args)
    updated = update_durable_speed_plans(
      {"corridor_object_s18_28": stale_slow},
      new_plan,
      red_program,
      current_long_m=12.0,
      args=args,
    )
    self.assertEqual(tuple(updated), ("traffic_light_stop",))

  def test_red_signal_speed_profile_stops_at_hold_radius_not_far_before_line(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=10.0,
      durable_speed_recover_m=3.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
      include_traffic_light=True,
      novel_scene="random_mixed",
      traffic_light_route_s=48.0,
      traffic_light_stop_before_m=5.0,
      traffic_light_full_stop_m=4.0,
      traffic_light_decel_distance_m=28.0,
      traffic_light_stop_hold_radius_m=0.75,
      traffic_light_comfort_decel_mps2=1.1,
    )
    red_program = parse_rtp(_labels_to_rtp(("red_stop_light",)))
    far_plan = _adjust_signal_speed_plan(None, red_program, current_long_m=38.5, args=args)
    full_stop_zone_plan = _adjust_signal_speed_plan(None, red_program, current_long_m=39.5, args=args)
    hold_plan = _adjust_signal_speed_plan(None, red_program, current_long_m=42.4, args=args)
    self.assertIsNotNone(far_plan)
    self.assertIsNotNone(full_stop_zone_plan)
    self.assertIsNotNone(hold_plan)
    self.assertGreater(far_plan.speed_cap_mps, 0.0)
    self.assertEqual(full_stop_zone_plan.speed_cap_mps, 0.0)
    self.assertEqual(hold_plan.speed_cap_mps, 0.0)

  def test_red_signal_after_stop_line_is_ignored(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=10.0,
      durable_speed_recover_m=3.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
      include_traffic_light=True,
      novel_scene="random_mixed",
      traffic_light_route_s=48.0,
      traffic_light_stop_before_m=5.0,
      traffic_light_full_stop_m=4.0,
      traffic_light_decel_distance_m=28.0,
      traffic_light_stop_hold_radius_m=0.75,
      traffic_light_comfort_decel_mps2=1.1,
      traffic_light_passed_ignore_m=2.0,
    )
    red_program = parse_rtp(_labels_to_rtp(("red_stop_light",)))
    self.assertIsNone(_adjust_signal_speed_plan(None, red_program, current_long_m=45.5, args=args))

  def test_current_green_visual_guard_clears_stale_signal_stop(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=10.0,
      durable_speed_recover_m=3.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
      disable_vlm_speed_control=False,
      include_traffic_light=True,
      novel_scene="random_mixed",
      traffic_light_route_s=48.0,
      traffic_light_stop_before_m=5.0,
      traffic_light_full_stop_m=4.0,
      traffic_light_decel_distance_m=28.0,
      traffic_light_stop_hold_radius_m=0.75,
      traffic_light_comfort_decel_mps2=1.1,
    )
    stale_stop = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=43.0,
      ramp_out_end_long_m=46.0,
      speed_cap_mps=0.0,
      stop_s=3.0,
      source_token="traffic_light_stop",
      source_meta="STOP",
      confidence=0.74,
    )
    updated = _apply_current_visual_signal_guard(
      {"traffic_light_stop": stale_stop},
      "green_go_light",
      current_long_m=40.0,
      args=args,
    )
    self.assertEqual(updated, {})

  def test_passed_signal_guard_clears_stale_stop_without_current_green_pixels(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=10.0,
      durable_speed_recover_m=3.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
      disable_vlm_speed_control=False,
      include_traffic_light=True,
      novel_scene="random_mixed",
      traffic_light_route_s=48.0,
      traffic_light_stop_before_m=5.0,
      traffic_light_full_stop_m=4.0,
      traffic_light_decel_distance_m=28.0,
      traffic_light_stop_hold_radius_m=0.75,
      traffic_light_comfort_decel_mps2=1.1,
      traffic_light_passed_ignore_m=2.0,
    )
    stale_stop = DurableSpeedPlan(
      start_long_m=0.0,
      end_long_m=43.0,
      ramp_out_end_long_m=55.0,
      speed_cap_mps=0.0,
      stop_s=0.0,
      source_token="traffic_light_stop",
      source_meta="STOP",
      confidence=0.74,
    )
    updated = _apply_current_visual_signal_guard(
      {"traffic_light_stop": stale_stop},
      None,
      current_long_m=45.5,
      args=args,
    )
    self.assertEqual(updated, {})

  def test_current_red_visual_guard_reinstates_stop_after_stale_green(self):
    args = Namespace(
      speed_mps=5.0,
      durable_slow_speed_scale=0.25,
      durable_speed_min_horizon_m=10.0,
      durable_speed_recover_m=3.0,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
      disable_vlm_speed_control=False,
      include_traffic_light=True,
      novel_scene="random_mixed",
      traffic_light_route_s=48.0,
      traffic_light_stop_before_m=5.0,
      traffic_light_full_stop_m=4.0,
      traffic_light_decel_distance_m=28.0,
      traffic_light_stop_hold_radius_m=0.75,
      traffic_light_comfort_decel_mps2=1.1,
    )
    updated = _apply_current_visual_signal_guard({}, "red_stop_light", current_long_m=20.0, args=args)
    self.assertEqual(tuple(updated), ("traffic_light_stop",))
    self.assertLess(updated["traffic_light_stop"].speed_cap_mps, args.speed_mps)

  def test_visual_signal_guard_is_explicit_demo_only_and_never_stock(self):
    base_args = Namespace(enable_visual_signal_guard=False, disable_vlm_speed_control=False)
    enabled_args = Namespace(enable_visual_signal_guard=True, disable_vlm_speed_control=False)
    disabled_speed_args = Namespace(enable_visual_signal_guard=True, disable_vlm_speed_control=True)

    self.assertFalse(_should_apply_visual_signal_guard("vlm", base_args, "red_stop_light"))
    self.assertFalse(_should_apply_visual_signal_guard("stock", enabled_args, "red_stop_light"))
    self.assertFalse(_should_apply_visual_signal_guard("vlm", disabled_speed_args, "red_stop_light"))
    self.assertFalse(_should_apply_visual_signal_guard("vlm", enabled_args, None))
    self.assertTrue(_should_apply_visual_signal_guard("vlm", enabled_args, "green_go_light"))

  def test_visual_signal_label_from_frame_detects_green_light(self):
    frame = np.zeros((320, 512, 3), dtype=np.uint8)
    frame[80:100, 280:300] = (20, 224, 60)
    self.assertEqual(_visual_traffic_signal_label_from_frame(frame), "green_go_light")


class TestMetaDriveLateralConvention(unittest.TestCase):
  def _base_plan(self):
    return BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(2.5 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=2.5,
    )

  def _green_x_mean(self, board):
    arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
    mask = (arr[:, :, 1] > 70) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 90)
    _, xs = np.nonzero(mask)
    self.assertGreater(len(xs), 10)
    return float(xs.mean())

  def _compile_construction_chain(self, labels):
    program = parse_rtp(_labels_to_rtp(labels))
    base = self._base_plan()
    synth = PathSynth().compile(base, program)
    selected_openpilot_offset_m = selected_lateral_offset_m(synth)
    args = Namespace(min_construction_offset_m=1.25, max_durable_offset_m=1.3, avoid_lead_m=10.0, avoid_recover_m=10.0)
    durable = durable_avoidance_from_program(program, current_long_m=0.0, selected_offset_m=selected_openpilot_offset_m, args=args)
    self.assertTrue(synth.valid)
    self.assertIsNotNone(durable)
    return base, program, synth, selected_openpilot_offset_m, durable

  def test_metadrive_harness_declares_left_and_right_construction_scenes(self):
    self.assertEqual(CONSTRUCTION_SCENES, {"construction", "construction_left", "construction_right"})
    self.assertEqual(construction_scene_side("construction"), "right")
    self.assertEqual(construction_scene_side("construction_right"), "right")
    self.assertEqual(construction_scene_side("construction_left"), "left")

  def test_openpilot_left_is_metadrive_negative(self):
    self.assertIs(openpilot_to_metadrive_lateral_m, side_semantics.openpilot_to_metadrive_lateral_m)
    self.assertIs(metadrive_to_openpilot_lateral_m, side_semantics.metadrive_to_openpilot_lateral_m)
    self.assertIs(lateral_side_openpilot, side_semantics.lateral_side_openpilot)
    self.assertIs(lateral_side_metadrive, side_semantics.lateral_side_metadrive)
    self.assertIs(construction_avoidance_side_valid, side_semantics.construction_avoidance_metadrive_side_valid)
    self.assertLess(openpilot_to_metadrive_lateral_m(1.25), 0.0)
    self.assertGreater(openpilot_to_metadrive_lateral_m(-1.25), 0.0)

  def test_metadrive_to_openpilot_round_trip(self):
    for offset in (-1.25, -0.4, 0.0, 0.6, 1.25):
      self.assertAlmostEqual(metadrive_to_openpilot_lateral_m(openpilot_to_metadrive_lateral_m(offset)), offset)

  def test_lateral_side_helpers_name_each_coordinate_space(self):
    self.assertEqual(lateral_side_openpilot(1.0), "left")
    self.assertEqual(lateral_side_openpilot(-1.0), "right")
    self.assertEqual(lateral_side_openpilot(0.0), "none")

    self.assertEqual(lateral_side_metadrive(-1.0), "left")
    self.assertEqual(lateral_side_metadrive(1.0), "right")
    self.assertEqual(lateral_side_metadrive(0.0), "none")

    self.assertEqual(construction_hazard_side_from_token("right_edge_s8_48_margin1.25"), "right")
    self.assertEqual(construction_hazard_side_from_token("left_edge_s8_48_margin1.25"), "left")
    self.assertEqual(construction_hazard_side_from_token("corridor_object_s18_28"), "none")
    self.assertTrue(construction_avoidance_side_valid("right_edge_s8_48_margin1.25", -1.0))
    self.assertTrue(construction_avoidance_side_valid("left_edge_s8_48_margin1.25", 1.0))
    self.assertFalse(construction_avoidance_side_valid("right_edge_s8_48_margin1.25", 1.0))
    self.assertFalse(construction_avoidance_side_valid("left_edge_s8_48_margin1.25", -1.0))

  def test_construction_spawn_and_avoidance_are_opposite_in_metadrive_adapter(self):
    cases = (
      ("left", "left_edge_s8_48_margin1.25", "right"),
      ("right", "right_edge_s8_48_margin1.25", "left"),
    )
    for hazard_side, avoid_token, expected_avoid_side in cases:
      with self.subTest(hazard_side=hazard_side):
        spawned_lateral = side_semantics.construction_hazard_metadrive_lateral_for_side(hazard_side, 1.35)
        avoidance_lateral = side_semantics.construction_avoidance_metadrive_lateral_for_hazard_side(hazard_side, 1.25)
        self.assertEqual(lateral_side_metadrive(spawned_lateral), hazard_side)
        self.assertEqual(lateral_side_metadrive(avoidance_lateral), expected_avoid_side)
        self.assertEqual(lateral_side_openpilot(metadrive_to_openpilot_lateral_m(avoidance_lateral)), expected_avoid_side)
        self.assertTrue(construction_avoidance_side_valid(avoid_token, avoidance_lateral))
        self.assertLess(spawned_lateral * avoidance_lateral, 0.0)

  def test_runtime_route_lateral_adapter_preserves_openpilot_side_when_lane_sign_flips(self):
    sign_negative = Namespace(route_lateral_sign_to_openpilot=-1.0)
    sign_positive = Namespace(route_lateral_sign_to_openpilot=1.0)

    self.assertEqual(route_to_openpilot_lateral_m_from_args(sign_negative, 1.25), -1.25)
    self.assertEqual(route_to_openpilot_lateral_m_from_args(sign_positive, 1.25), 1.25)
    self.assertEqual(openpilot_to_route_lateral_m_from_args(sign_negative, 1.25), -1.25)
    self.assertEqual(openpilot_to_route_lateral_m_from_args(sign_positive, 1.25), 1.25)

    for args in (sign_negative, sign_positive):
      with self.subTest(sign=args.route_lateral_sign_to_openpilot):
        right_hazard_route = route_lateral_for_side_from_args(args, "right", 1.35)
        left_hazard_route = route_lateral_for_side_from_args(args, "left", 1.35)
        self.assertEqual(lateral_side_openpilot(route_to_openpilot_lateral_m_from_args(args, right_hazard_route)), "right")
        self.assertEqual(lateral_side_openpilot(route_to_openpilot_lateral_m_from_args(args, left_hazard_route)), "left")
        self.assertLess(right_hazard_route * left_hazard_route, 0.0)

  def test_runtime_route_lateral_adapter_delegates_to_shared_semantics(self):
    args = Namespace(route_lateral_sign_to_openpilot=1.0)
    self.assertEqual(
      route_to_openpilot_lateral_m_from_args(args, -1.25),
      side_semantics.route_to_openpilot_lateral_m(-1.25, args.route_lateral_sign_to_openpilot),
    )
    self.assertEqual(
      openpilot_to_route_lateral_m_from_args(args, 1.25),
      side_semantics.openpilot_to_route_lateral_m(1.25, args.route_lateral_sign_to_openpilot),
    )
    self.assertEqual(
      route_lateral_for_side_from_args(args, "right", 1.25),
      side_semantics.route_lateral_for_openpilot_side("right", 1.25, args.route_lateral_sign_to_openpilot),
    )

  def test_runtime_route_lateral_adapter_validates_durable_avoidance_after_lane_sign_flip(self):
    program = parse_rtp(_labels_to_rtp(("construction_right",)))
    base = self._base_plan()
    synth = PathSynth().compile(base, program)
    selected_openpilot_offset_m = selected_lateral_offset_m(synth)
    args = Namespace(
      route_lateral_sign_to_openpilot=1.0,
      min_construction_offset_m=1.25,
      max_durable_offset_m=1.3,
      avoid_lead_m=10.0,
      avoid_recover_m=10.0,
    )
    durable = durable_avoidance_from_program(program, current_long_m=0.0, selected_offset_m=selected_openpilot_offset_m, args=args)
    self.assertIsNotNone(durable)
    self.assertGreater(durable.offset_m, 0.0)
    self.assertEqual(lateral_side_openpilot(route_to_openpilot_lateral_m_from_args(args, durable.offset_m)), "left")
    self.assertTrue(durable_avoidance_sign_valid_for_args(durable, args))

  def test_existing_durable_lateral_plans_are_revalidated_with_runtime_route_sign(self):
    args = Namespace(
      route_lateral_sign_to_openpilot=1.0,
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.80,
    )
    valid_right_hazard_plan = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.90,
    )
    wrong_right_hazard_plan = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.90,
    )

    kept = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": valid_right_hazard_plan},
      None,
      None,
      current_long_m=5.0,
      args=args,
    )
    rejected = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": wrong_right_hazard_plan},
      None,
      None,
      current_long_m=5.0,
      args=args,
    )

    self.assertEqual(kept, {"right_edge_s8_48_margin1.25": valid_right_hazard_plan})
    self.assertEqual(rejected, {})

  def test_route_vehicle_visual_heading_default_is_same_direction_not_head_on(self):
    self.assertEqual(ROUTE_VEHICLE_VISUAL_HEADING_OFFSET_RAD, 0.0)
    for heading in (-1.2, 0.0, 1.1):
      with self.subTest(heading=heading):
        self.assertAlmostEqual(route_vehicle_visual_heading(heading), heading)

  def test_canonical_construction_side_semantics_cover_all_qwen_families(self):
    cases = (
      (("construction_right",), "right", "BIAS_LEFT", 1.25, "right_edge_s8_48_margin1.25"),
      (("construction_left",), "left", "BIAS_RIGHT", -1.25, "left_edge_s8_48_margin1.25"),
      (("construction_purple_edge",), "right", "BIAS_LEFT", 1.25, "right_edge_s8_48_margin1.25"),
      (("construction_blue_edge",), "left", "BIAS_RIGHT", -1.25, "left_edge_s8_48_margin1.25"),
      (("construction_shift_left",), "right", "BIAS_LEFT", 1.25, "right_edge_s8_48_margin1.25"),
      (("construction_shift_right",), "left", "BIAS_RIGHT", -1.25, "left_edge_s8_48_margin1.25"),
      (("construction_blocks_right_candidate",), "right", "BIAS_LEFT", 1.25, "right_edge_s8_48_margin1.25"),
      (("construction_blocks_left_candidate",), "left", "BIAS_RIGHT", -1.25, "left_edge_s8_48_margin1.25"),
    )
    for labels, expected_side, expected_meta, expected_bias, expected_avoid in cases:
      with self.subTest(labels=labels):
        side = side_semantics.construction_hazard_side_from_labels(labels)
        fields = side_semantics.construction_rtp_fields_for_hazard_side(side)
        self.assertEqual(side, expected_side)
        self.assertEqual(fields.meta, expected_meta)
        self.assertEqual(fields.lat_bias_m, expected_bias)
        self.assertEqual(fields.avoid_token, expected_avoid)
        self.assertEqual(side_semantics.construction_hazard_side_from_avoid_token(fields.avoid_token), expected_side)
        self.assertTrue(side_semantics.construction_avoidance_openpilot_side_valid(fields.avoid_token, fields.lat_bias_m))

  def test_qwen_construction_labels_have_one_away_direction_through_full_chain(self):
    cases = (
      (("construction_right",), "right", "left"),
      (("construction_left",), "left", "right"),
      (("construction_purple_edge",), "right", "left"),
      (("construction_blue_edge",), "left", "right"),
      (("construction_shift_left",), "right", "left"),
      (("construction_shift_right",), "left", "right"),
      (("construction_blocks_left_candidate",), "left", "right"),
      (("construction_blocks_right_candidate",), "right", "left"),
    )
    renderer = UiSceneBoardRenderer(320, 240, geometry=OverlayGeometry(draw_base_path_reference=True))

    for labels, expected_hazard_side, expected_target_side in cases:
      with self.subTest(labels=labels):
        base, program, synth, selected_openpilot_offset_m, durable = self._compile_construction_chain(labels)
        tracked_openpilot_offset_m = metadrive_to_openpilot_lateral_m(durable.offset_m)

        self.assertEqual(construction_hazard_side_from_token(durable.source_token), expected_hazard_side)
        self.assertEqual(lateral_side_openpilot(selected_openpilot_offset_m), expected_target_side)
        self.assertEqual(lateral_side_metadrive(durable.offset_m), expected_target_side)
        self.assertEqual(lateral_side_openpilot(tracked_openpilot_offset_m), expected_target_side)
        self.assertTrue(durable_avoidance_sign_valid(durable))

        centered = renderer.render(base, {"path_lateral_offset_m": 0.0})
        shifted = renderer.render(base, {"path_lateral_offset_m": tracked_openpilot_offset_m})
        if expected_target_side == "left":
          self.assertLess(self._green_x_mean(shifted), self._green_x_mean(centered))
        else:
          self.assertGreater(self._green_x_mean(shifted), self._green_x_mean(centered))

  def test_spawned_route_clearance_can_filter_construction_only(self):
    spawned = [
      {"kind": "controlled_traffic_light", "route_s_m": 10.0, "lateral_m": 0.0},
      {"kind": "traffic_cone_right_edge", "route_s_m": 12.0, "lateral_m": 1.3},
      {"kind": "moving_pedestrian_crossing", "route_s_m": 8.0, "lateral_m": 0.0},
    ]
    self.assertAlmostEqual(
      spawned_route_clearance_m(spawned, current_route_long_m=10.0, current_lateral_m=0.0, kind_tokens=("traffic_cone",)),
      (2.0 ** 2 + 1.3 ** 2) ** 0.5,
    )

  def test_spawned_route_proximity_logs_ahead_and_lateral_delta(self):
    spawned = [
      {"kind": "traffic_cone_right_edge", "route_s_m": 18.0, "lateral_m": 1.35},
      {"kind": "traffic_barrier_right_edge", "route_s_m": 14.0, "lateral_m": 1.20},
      {"kind": "moving_pedestrian_crossing", "route_s_m": 13.0, "lateral_m": 0.0},
    ]
    proximity = spawned_route_proximity(
      spawned,
      current_route_long_m=12.0,
      current_lateral_m=-0.25,
      kind_tokens=("traffic_cone", "traffic_barrier"),
    )
    self.assertEqual(proximity["kind"], "traffic_barrier_right_edge")
    self.assertAlmostEqual(proximity["ahead_m"], 2.0)
    self.assertAlmostEqual(proximity["lateral_delta_m"], 1.45)
    self.assertAlmostEqual(proximity["object_lateral_m"], 1.20)


class TestConstructionTraceEvaluation(unittest.TestCase):
  def _base_plan(self):
    return BasePlan(
      frame_id=0,
      model_log_mono_time_ns=0,
      t=tuple(i * 0.2 for i in range(20)),
      x=tuple(np.linspace(2.0, 60.0, 20)),
      y=tuple(0.0 for _ in range(20)),
      speeds=tuple(2.5 for _ in range(20)),
      desired_curvature=0.0,
      v_ego=2.5,
    )

  def _green_x_mean(self, board):
    arr = np.frombuffer(bytes(board.pixels), dtype=np.uint8).reshape(board.height, board.width, 3)
    mask = (arr[:, :, 1] > 70) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 90)
    _, xs = np.nonzero(mask)
    self.assertGreater(len(xs), 10)
    return float(xs.mean())

  def _compile_construction_chain(self, labels):
    program = parse_rtp(_labels_to_rtp(labels))
    base = self._base_plan()
    synth = PathSynth().compile(base, program)
    selected_openpilot_offset_m = selected_lateral_offset_m(synth)
    args = Namespace(min_construction_offset_m=1.25, max_durable_offset_m=1.3, avoid_lead_m=10.0, avoid_recover_m=10.0)
    durable = durable_avoidance_from_program(program, current_long_m=0.0, selected_offset_m=selected_openpilot_offset_m, args=args)
    self.assertTrue(synth.valid)
    self.assertIsNotNone(durable)
    return base, program, synth, selected_openpilot_offset_m, durable

  def test_construction_requirement_uses_ahead_and_lateral_delta(self):
    required = construction_requirement_from_record(
      {"construction_nearest_ahead_m": 8.0, "construction_nearest_lateral_delta_m": 1.2},
      horizon_m=14.0,
      intrusion_m=1.65,
    )
    self.assertTrue(required["required"])
    self.assertEqual(required["required_side"], "right")
    self.assertEqual(required["required_shift_label"], "construction_shift_left")

    far = construction_requirement_from_record(
      {"construction_nearest_ahead_m": 18.0, "construction_nearest_lateral_delta_m": 1.2},
      horizon_m=14.0,
      intrusion_m=1.65,
    )
    self.assertFalse(far["required"])

    clear_lateral = construction_requirement_from_record(
      {"construction_nearest_ahead_m": 8.0, "construction_nearest_lateral_delta_m": 2.2},
      horizon_m=14.0,
      intrusion_m=1.65,
    )
    self.assertFalse(clear_lateral["required"])

  def test_construction_requirement_falls_back_to_spawned_scene_trace(self):
    required = construction_requirement_from_record(
      {
        "route_longitudinal_m": 10.0,
        "control_debug": {"lane_lateral_m": -0.1},
        "spawned_scene": [
          {"kind": "traffic_cone_right_edge", "route_s_m": 18.0, "lateral_m": 1.2},
          {"kind": "moving_pedestrian_crossing", "route_s_m": 11.0, "lateral_m": 0.0},
        ],
      },
      horizon_m=14.0,
      intrusion_m=1.65,
    )
    self.assertTrue(required["required"])
    self.assertEqual(required["required_side"], "right")
    self.assertAlmostEqual(required["ahead_m"], 8.0)
    self.assertAlmostEqual(required["lateral_delta_m"], 1.3)

  def test_pedestrian_requirement_distinguishes_in_path_entering_and_irrelevant(self):
    in_path = pedestrian_requirement_from_record(
      {
        "route_longitudinal_m": 10.0,
        "control_debug": {"lane_lateral_m": 0.0},
        "spawned_scene": [
          {"kind": "moving_pedestrian_crossing", "route_s_m": 18.0, "lateral_m": 0.2, "target_lateral_m": -1.2},
        ],
      },
      horizon_m=30.0,
      in_path_lateral_m=0.9,
      entering_lateral_m=2.4,
    )
    self.assertTrue(in_path["required"])
    self.assertEqual(in_path["expected_label"], "pedestrian_in_path")

    entering = pedestrian_requirement_from_record(
      {
        "route_longitudinal_m": 10.0,
        "control_debug": {"lane_lateral_m": 0.0},
        "spawned_scene": [
          {"kind": "moving_pedestrian_crossing", "route_s_m": 18.0, "lateral_m": 1.8, "target_lateral_m": 0.0},
        ],
      },
      horizon_m=30.0,
      in_path_lateral_m=0.9,
      entering_lateral_m=2.4,
    )
    self.assertTrue(entering["required"])
    self.assertEqual(entering["expected_label"], "pedestrian_entering_path")

    irrelevant = pedestrian_requirement_from_record(
      {
        "route_longitudinal_m": 10.0,
        "control_debug": {"lane_lateral_m": 0.0},
        "spawned_scene": [
          {"kind": "moving_pedestrian_crossing", "route_s_m": 18.0, "lateral_m": 3.1, "target_lateral_m": 2.8},
        ],
      },
      horizon_m=30.0,
      in_path_lateral_m=0.9,
      entering_lateral_m=2.4,
    )
    self.assertFalse(irrelevant["required"])

  def test_pedestrian_qwen_class_maps_only_pedestrian_conflict_labels(self):
    self.assertEqual(qwen_pedestrian_class(["pedestrian_in_path"]), "in_path")
    self.assertEqual(qwen_pedestrian_class(["pedestrian_entering_path"]), "entering_path")
    self.assertEqual(qwen_pedestrian_class(["pedestrian_in_path", "pedestrian_entering_path"]), "in_path")
    self.assertEqual(qwen_pedestrian_class(["vehicle_in_path"]), "none")

  def test_pedestrian_evaluator_counts_both_labels_as_in_path_precedence(self):
    episode = {
      "records": [{
        "frame_id": 1,
        "model_frame_id": 1,
        "rtp_source_frame_id": 1,
        "route_longitudinal_m": 10.0,
        "control_debug": {"lane_lateral_m": 0.0},
        "spawned_scene": [
          {"kind": "moving_pedestrian_crossing", "route_s_m": 18.0, "lateral_m": 0.2, "target_lateral_m": -1.0},
        ],
        "qwen_labels": ["pedestrian_in_path", "pedestrian_entering_path"],
        "desired_speed_mps": 5.0,
        "target_speed_mps": 0.0,
        "durable_speed_plan_sources": ["corridor_object_s18_28"],
        "durable_speed_cap_mps": 0.0,
        "control_consumed_age_frames": 0,
        "rtp_age_frames": 0,
        "info_flags": {},
        "green_path_matches_tracked_path": True,
      }],
    }
    result = evaluate_pedestrian_episode(episode, horizon_m=30.0, in_path_lateral_m=0.9, entering_lateral_m=2.4, slow_scale=0.75)
    self.assertEqual(result["path_relevant_pedestrian_frames"], 1)
    self.assertEqual(result["qwen_pedestrian_path_relevant"], 1)
    self.assertEqual(result["qwen_pedestrian_exact"], 1)
    self.assertEqual(result["qwen_pedestrian_wrong"], 0)
    self.assertEqual(result["qwen_pedestrian_missing"], 0)
    self.assertEqual(result["post_first_qwen_pedestrian_path_relevant"], 1)

  def test_evaluate_pedestrian_episode_scores_required_control_and_false_slow(self):
    episode = {
      "records": [
        {
          "frame_id": 1,
          "model_frame_id": 1,
          "rtp_source_frame_id": 1,
          "route_longitudinal_m": 10.0,
          "control_debug": {"lane_lateral_m": 0.0},
          "spawned_scene": [
            {"kind": "moving_pedestrian_crossing", "route_s_m": 18.0, "lateral_m": 0.2, "target_lateral_m": -1.0},
          ],
          "qwen_labels": ["pedestrian_in_path"],
          "desired_speed_mps": 5.0,
          "target_speed_mps": 2.5,
          "durable_speed_plan_sources": ["corridor_object_s18_28"],
          "durable_speed_cap_mps": 2.5,
          "control_consumed_age_frames": 0,
          "rtp_age_frames": 0,
          "info_flags": {},
          "green_path_matches_tracked_path": True,
        },
        {
          "frame_id": 2,
          "model_frame_id": 2,
          "route_longitudinal_m": 10.0,
          "control_debug": {"lane_lateral_m": 0.0},
          "spawned_scene": [
            {"kind": "moving_pedestrian_crossing", "route_s_m": 35.0, "lateral_m": 0.2, "target_lateral_m": -1.0},
          ],
          "qwen_labels": ["pedestrian_in_path"],
          "desired_speed_mps": 5.0,
          "target_speed_mps": 5.0,
          "durable_speed_plan_sources": ["corridor_object_s18_28"],
          "durable_speed_cap_mps": None,
          "control_consumed_age_frames": 0,
          "rtp_age_frames": 0,
          "info_flags": {},
          "green_path_matches_tracked_path": True,
        },
        {
          "frame_id": 3,
          "model_frame_id": 3,
          "route_longitudinal_m": 10.0,
          "control_debug": {"lane_lateral_m": 0.0},
          "spawned_scene": [
            {"kind": "moving_pedestrian_crossing", "route_s_m": 18.0, "lateral_m": 3.2, "target_lateral_m": 3.1},
          ],
          "qwen_labels": ["pedestrian_entering_path"],
          "desired_speed_mps": 5.0,
          "target_speed_mps": 2.5,
          "durable_speed_plan_sources": ["corridor_object_s18_28"],
          "durable_speed_cap_mps": 2.5,
          "info_flags": {"crash_human": True},
          "green_path_matches_tracked_path": False,
        },
      ],
    }
    result = evaluate_pedestrian_episode(episode, horizon_m=30.0, in_path_lateral_m=0.9, entering_lateral_m=2.4, slow_scale=0.75)
    self.assertEqual(result["path_relevant_pedestrian_frames"], 2)
    self.assertEqual(result["qwen_pedestrian_path_relevant"], 2)
    self.assertEqual(result["qwen_pedestrian_exact"], 2)
    self.assertEqual(result["consumed_agent_frames"], 2)
    self.assertEqual(result["control_success_frames"], 2)
    self.assertEqual(result["false_qwen_pedestrian_frames"], 1)
    self.assertEqual(result["false_consumed_agent_frames"], 1)
    self.assertEqual(result["false_slow_frames"], 1)
    self.assertEqual(result["collision_count"], 1)
    self.assertEqual(result["green_path_mismatch_count"], 1)

  def test_signal_requirement_ignores_already_passed_signal(self):
    self.assertEqual(
      signal_requirement_from_record(
        {
          "traffic_light_state": "red",
          "traffic_light_remaining_to_stop_m": -3.5,
          "qwen_labels": ["red_stop_light"],
        },
        passed_ignore_m=2.0,
      ),
      "passed_signal",
    )

  def test_signal_evaluator_scores_red_stop_and_green_release_without_visual_guard(self):
    episode = {
      "records": [
        {
          "frame_id": 1,
          "traffic_light_state": "red",
          "traffic_light_remaining_to_stop_m": 18.0,
          "qwen_labels": ["red_stop_light"],
          "rtp_text": "evidence=[red_signal_for_path]",
          "desired_speed_mps": 5.0,
          "target_speed_mps": 4.0,
          "durable_speed_plan_sources": ["traffic_light_stop"],
          "control_consumed_age_frames": 0,
          "rtp_age_frames": 0,
          "green_path_matches_tracked_path": True,
          "visual_signal_guard_enabled": False,
        },
        {
          "frame_id": 2,
          "traffic_light_state": "red",
          "traffic_light_remaining_to_stop_m": 3.0,
          "qwen_labels": ["red_stop_light"],
          "rtp_text": "evidence=[red_signal_for_path]",
          "desired_speed_mps": 5.0,
          "target_speed_mps": 0.0,
          "durable_speed_plan_sources": ["traffic_light_stop"],
          "control_consumed_age_frames": 0,
          "rtp_age_frames": 0,
          "green_path_matches_tracked_path": True,
          "visual_signal_guard_enabled": False,
        },
        {
          "frame_id": 3,
          "traffic_light_state": "green",
          "traffic_light_remaining_to_stop_m": 2.5,
          "qwen_labels": ["green_go_light"],
          "rtp_text": "evidence=[green_signal_for_path]",
          "desired_speed_mps": 5.0,
          "target_speed_mps": 5.0,
          "durable_speed_plan_sources": [],
          "control_consumed_age_frames": 0,
          "rtp_age_frames": 0,
          "green_path_matches_tracked_path": True,
          "visual_signal_guard_enabled": False,
        },
      ],
    }
    result = evaluate_signal_episode(episode, max_age_frames=3, min_success_rate=0.95)
    self.assertTrue(result["ok"])
    self.assertEqual(result["counts"]["red_required"], 2)
    self.assertEqual(result["counts"]["green_required"], 1)
    self.assertEqual(result["red_qwen_success_rate"], 1.0)
    self.assertEqual(result["red_control_success_rate"], 1.0)
    self.assertEqual(result["green_qwen_success_rate"], 1.0)
    self.assertEqual(result["green_control_success_rate"], 1.0)

  def test_signal_evaluator_rejects_visual_signal_guard_as_proof(self):
    episode = {
      "records": [{
        "frame_id": 1,
        "traffic_light_state": "red",
        "traffic_light_remaining_to_stop_m": 3.0,
        "qwen_labels": ["red_stop_light"],
        "rtp_text": "evidence=[red_signal_for_path]",
        "desired_speed_mps": 5.0,
        "target_speed_mps": 0.0,
        "durable_speed_plan_sources": ["traffic_light_stop"],
        "control_consumed_age_frames": 0,
        "rtp_age_frames": 0,
        "green_path_matches_tracked_path": True,
        "visual_signal_guard_enabled": True,
      }],
    }
    result = evaluate_signal_episode(episode, max_age_frames=3, min_success_rate=0.95)
    self.assertFalse(result["ok"])
    self.assertIn("visual_signal_guard_enabled", result["issues"])

  def test_construction_label_side_maps_shift_labels_to_hazard_side(self):
    self.assertEqual(qwen_construction_side(["construction_left"]), "left")
    self.assertEqual(qwen_construction_side(["construction_blue_edge"]), "left")
    self.assertEqual(qwen_construction_side(["construction_purple_edge"]), "right")
    self.assertEqual(qwen_construction_side(["construction_drive_left"]), "right")
    self.assertEqual(qwen_construction_side(["construction_drive_right"]), "left")
    self.assertEqual(qwen_construction_side(["construction_shift_left"]), "right")
    self.assertEqual(qwen_construction_side(["construction_blocks_left_candidate"]), "left")
    self.assertEqual(qwen_construction_side(["construction_blocks_right_candidate"]), "right")
    self.assertEqual(qwen_construction_side(["construction_left", "construction_right"]), "none")

  def test_consumed_construction_side_uses_active_durable_plan_before_falling_back_to_rtp(self):
    self.assertEqual(
      consumed_construction_side({
        "qwen_labels": [],
        "durable_lateral_plan_details": [{"hazard_side": "right", "sign_valid": True}],
        "rtp_text": "",
      }),
      "right",
    )
    self.assertEqual(
      consumed_construction_side({
        "durable_lateral_plan_details": [],
        "rtp_text": "RTPv1\nscene=construction_left\navoid=[left_edge_s8_48_margin1.25]\n",
      }),
      "left",
    )
    self.assertEqual(
      consumed_construction_side({
        "durable_lateral_plan_details": [{"hazard_side": "right"}, {"hazard_side": "left"}],
      }),
      "conflict",
    )

  def test_lateral_command_side_uses_metadrive_sign(self):
    self.assertEqual(lateral_command_side({"desired_lateral_offset_m": -0.4}, min_offset_m=0.1), "left")
    self.assertEqual(lateral_command_side({"desired_lateral_offset_m": 0.4}, min_offset_m=0.1), "right")
    self.assertEqual(lateral_command_side({"desired_lateral_offset_m": 0.02}, min_offset_m=0.1), "none")
    self.assertEqual(
      lateral_command_side({"active_lateral_offset_m": -0.4, "desired_lateral_offset_m": 0.4}, min_offset_m=0.1),
      "left",
    )
    self.assertEqual(
      planned_lateral_command_side({"active_lateral_offset_m": -0.4, "desired_lateral_offset_m": 0.4}, min_offset_m=0.1),
      "right",
    )

  def test_evaluate_episode_counts_side_errors_and_sign_flips(self):
    episode = {
      "records": [
        {
          "frame_id": 1,
          "construction_nearest_ahead_m": 8.0,
          "construction_nearest_lateral_delta_m": 1.2,
          "qwen_labels": ["construction_shift_left"],
          "qwen_raw_labels": ["construction_blue_edge"],
          "qwen_raw_label_scores": {"construction_blue_edge": 2.0, "construction_purple_edge": 0.0},
          "qwen_labels_scored_this_request": ["construction_blue_edge", "construction_purple_edge"],
          "qwen_score_group_index": 1,
          "qwen_label_state_debug": {"construction_locked_side": "right"},
          "desired_lateral_offset_m": -0.5,
          "info_flags": {},
          "rtp_age_frames": 2,
        },
        {
          "frame_id": 2,
          "construction_nearest_ahead_m": 8.0,
          "construction_nearest_lateral_delta_m": -1.2,
          "qwen_labels": ["construction_shift_left"],
          "desired_lateral_offset_m": -0.5,
          "info_flags": {"crash_object": True},
          "rtp_age_frames": 3,
        },
        {
          "frame_id": 3,
          "construction_nearest_ahead_m": 20.0,
          "construction_nearest_lateral_delta_m": 1.2,
          "qwen_labels": ["construction_right"],
          "desired_lateral_offset_m": 0.0,
          "info_flags": {},
          "rtp_age_frames": 1,
        },
      ]
    }
    result = evaluate_episode(episode, horizon_m=14.0, intrusion_m=1.65, min_offset_m=0.1)
    self.assertEqual(result["path_relevant_construction_frames"], 2)
    self.assertEqual(result["qwen_side_correct"], 1)
    self.assertEqual(result["qwen_side_wrong"], 1)
    self.assertEqual(result["consumed_plan_side_correct"], 0)
    self.assertEqual(result["consumed_plan_side_wrong"], 0)
    self.assertEqual(result["consumed_plan_side_missing"], 2)
    self.assertEqual(result["control_away_frames"], 1)
    self.assertEqual(result["control_toward_frames"], 1)
    self.assertEqual(result["false_construction_labels"], 1)
    self.assertEqual(result["collision_count"], 1)
    self.assertEqual(result["max_rtp_age_frames"], 3)
    self.assertEqual(result["rows"][0]["qwen_raw_labels"], ["construction_blue_edge"])
    self.assertEqual(result["rows"][0]["qwen_raw_label_scores"]["construction_blue_edge"], 2.0)
    self.assertEqual(result["rows"][0]["qwen_labels_scored_this_request"], ["construction_blue_edge", "construction_purple_edge"])
    self.assertEqual(result["rows"][0]["qwen_score_group_index"], 1)
    self.assertEqual(result["rows"][0]["qwen_label_state_debug"]["construction_locked_side"], "right")
    self.assertEqual(result["rows"][0]["raw_qwen_side"], "left")

  def test_evaluate_episode_counts_consumed_plan_when_qwen_labels_are_between_updates(self):
    episode = {
      "records": [
        {
          "frame_id": 10,
          "model_frame_id": 10,
          "rtp_source_frame_id": 4,
          "construction_nearest_ahead_m": 8.0,
          "construction_nearest_lateral_delta_m": 1.2,
          "qwen_labels": [],
          "desired_lateral_offset_m": -0.6,
          "durable_lateral_plan_details": [{"hazard_side": "right", "target_side_metadrive": "left", "sign_valid": True}],
          "control_consumed_age_frames": 6,
          "info_flags": {},
          "rtp_age_frames": None,
          "green_path_matches_tracked_path": True,
        },
      ]
    }
    result = evaluate_episode(episode, horizon_m=14.0, intrusion_m=1.65, min_offset_m=0.1)
    self.assertEqual(result["qwen_side_missing"], 1)
    self.assertEqual(result["consumed_plan_side_correct"], 1)
    self.assertEqual(result["consumed_plan_side_success_rate"], 1.0)
    self.assertEqual(result["control_away_frames"], 1)
    self.assertEqual(result["max_consumed_age_frames"], 6)
    self.assertEqual(result["green_path_mismatch_count"], 0)

  def test_evaluate_episode_reports_post_first_side_startup_adjusted_metrics(self):
    episode = {
      "records": [
        {
          "frame_id": 0,
          "construction_nearest_ahead_m": 8.0,
          "construction_nearest_lateral_delta_m": 1.2,
          "qwen_labels": [],
          "desired_lateral_offset_m": 0.0,
          "info_flags": {},
          "rtp_age_frames": None,
        },
        {
          "frame_id": 1,
          "construction_nearest_ahead_m": 7.0,
          "construction_nearest_lateral_delta_m": 1.2,
          "qwen_labels": ["construction_shift_left"],
          "desired_lateral_offset_m": -0.5,
          "durable_lateral_plan_details": [{"hazard_side": "right", "target_side_metadrive": "left", "sign_valid": True}],
          "info_flags": {},
          "rtp_age_frames": 2,
        },
      ]
    }
    result = evaluate_episode(episode, horizon_m=14.0, intrusion_m=1.65, min_offset_m=0.1)
    self.assertEqual(result["qwen_side_correct"], 1)
    self.assertEqual(result["qwen_side_missing"], 1)
    self.assertAlmostEqual(result["qwen_side_success_rate"], 0.5)
    self.assertEqual(result["first_qwen_side_frame"], 1)
    self.assertEqual(result["first_consumed_side_frame"], 1)
    self.assertEqual(result["post_first_side_path_relevant_construction_frames"], 1)
    self.assertEqual(result["post_first_side_qwen_side_correct"], 1)
    self.assertEqual(result["post_first_side_qwen_side_missing"], 0)
    self.assertAlmostEqual(result["post_first_side_qwen_side_success_rate"], 1.0)
    self.assertEqual(result["post_first_consumed_side_path_relevant_construction_frames"], 1)
    self.assertEqual(result["post_first_consumed_side_correct"], 1)
    self.assertAlmostEqual(result["post_first_consumed_side_success_rate"], 1.0)

  def test_right_edge_avoidance_moves_left_in_metadrive_coordinates(self):
    program = parse_rtp(_labels_to_rtp(("construction_right",)))
    args = Namespace(min_construction_offset_m=1.25, max_durable_offset_m=1.3, avoid_lead_m=10.0, avoid_recover_m=10.0)
    plan = durable_avoidance_from_program(program, current_long_m=0.0, selected_offset_m=program.lat_bias_m, args=args)
    self.assertIsNotNone(plan)
    self.assertLess(plan.offset_m, 0.0)
    self.assertTrue(durable_avoidance_sign_valid(plan))

  def test_left_edge_avoidance_moves_right_in_metadrive_coordinates(self):
    program = parse_rtp(_labels_to_rtp(("construction_left",)))
    args = Namespace(min_construction_offset_m=1.25, max_durable_offset_m=1.3, avoid_lead_m=10.0, avoid_recover_m=10.0)
    plan = durable_avoidance_from_program(program, current_long_m=0.0, selected_offset_m=program.lat_bias_m, args=args)
    self.assertIsNotNone(plan)
    self.assertGreater(plan.offset_m, 0.0)
    self.assertTrue(durable_avoidance_sign_valid(plan))

  def test_durable_avoidance_rejects_signs_that_move_toward_edge(self):
    wrong_right = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.90,
    )
    wrong_left = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.90,
    )
    self.assertFalse(durable_avoidance_sign_valid(wrong_right))
    self.assertFalse(durable_avoidance_sign_valid(wrong_left))

  def test_right_construction_full_chain_moves_left_and_renders_left(self):
    base, program, synth, selected_openpilot_offset_m, durable = self._compile_construction_chain(("construction_right",))
    self.assertEqual(program.meta, "BIAS_LEFT")
    self.assertGreater(program.lat_bias_m, 0.0)
    self.assertGreater(selected_openpilot_offset_m, 0.0)
    self.assertLess(durable.offset_m, 0.0)

    tracked_openpilot_offset_m = metadrive_to_openpilot_lateral_m(durable.offset_m)
    self.assertGreater(tracked_openpilot_offset_m, 0.0)
    renderer = UiSceneBoardRenderer(320, 240, geometry=OverlayGeometry(draw_base_path_reference=True))
    centered = renderer.render(base, {"path_lateral_offset_m": 0.0})
    shifted = renderer.render(base, {"path_lateral_offset_m": tracked_openpilot_offset_m})
    self.assertLess(self._green_x_mean(shifted), self._green_x_mean(centered))
    self.assertIn(f"tracked_path_lat={tracked_openpilot_offset_m:.2f}m", shifted.state_text)

  def test_left_construction_full_chain_moves_right_and_renders_right(self):
    base, program, synth, selected_openpilot_offset_m, durable = self._compile_construction_chain(("construction_left",))
    self.assertEqual(program.meta, "BIAS_RIGHT")
    self.assertLess(program.lat_bias_m, 0.0)
    self.assertLess(selected_openpilot_offset_m, 0.0)
    self.assertGreater(durable.offset_m, 0.0)

    tracked_openpilot_offset_m = metadrive_to_openpilot_lateral_m(durable.offset_m)
    self.assertLess(tracked_openpilot_offset_m, 0.0)
    renderer = UiSceneBoardRenderer(320, 240, geometry=OverlayGeometry(draw_base_path_reference=True))
    centered = renderer.render(base, {"path_lateral_offset_m": 0.0})
    shifted = renderer.render(base, {"path_lateral_offset_m": tracked_openpilot_offset_m})
    self.assertGreater(self._green_x_mean(shifted), self._green_x_mean(centered))
    self.assertIn(f"tracked_path_lat={tracked_openpilot_offset_m:.2f}m", shifted.state_text)

  def test_confident_contradictory_lateral_plan_replaces_old_side(self):
    args = Namespace(
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.90,
      durable_construction_conflict_immediate_confidence=0.90,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT_AND_SLOW",
      confidence=0.72,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT_AND_SLOW",
      confidence=0.92,
    )
    program = parse_rtp(_labels_to_rtp(("construction_left",)))
    updated = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": existing},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=DurableLateralOverrideState(),
    )
    self.assertNotIn("right_edge_s8_48_margin1.25", updated)
    self.assertIn("left_edge_s8_48_margin1.25", updated)
    self.assertGreater(compose_lateral_offset(updated, 10.0, args.max_durable_offset_m), 0.0)

  def test_low_confidence_new_lateral_plan_does_not_activate(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_lateral_activation_confidence=0.80,
      durable_lateral_activation_confirm_frames=1,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.72,
    )
    program = parse_rtp(_labels_to_rtp(("construction_shift_right",)))
    updated = update_durable_lateral_plans(
      {},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=DurableLateralOverrideState(),
    )
    self.assertEqual(updated, {})

  def test_high_confidence_new_lateral_plan_activates(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_lateral_activation_confidence=0.80,
      durable_lateral_activation_confirm_frames=1,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.84,
    )
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_shift_right",)), 0.84))
    updated = update_durable_lateral_plans(
      {},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=DurableLateralOverrideState(),
    )
    self.assertIn("left_edge_s8_48_margin1.25", updated)

  def test_new_lateral_plan_requires_repeated_activation_when_configured(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_lateral_activation_confidence=0.80,
      durable_lateral_activation_confirm_frames=2,
      durable_lateral_activation_immediate_confidence=0.95,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.84,
    )
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_left",)), 0.84))
    state = DurableLateralOverrideState()
    updated = update_durable_lateral_plans(
      {},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
    )
    self.assertEqual(updated, {})
    self.assertEqual(state.pending_source_token, "left_edge_s8_48_margin1.25")
    self.assertEqual(state.pending_sign, 1)
    self.assertEqual(state.pending_count, 1)

    updated = update_durable_lateral_plans(
      updated,
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
    )
    self.assertIn("left_edge_s8_48_margin1.25", updated)
    self.assertEqual(state.pending_source_token, "")
    self.assertEqual(state.pending_count, 0)

  def test_stale_reused_rtp_does_not_count_as_repeated_lateral_activation(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_lateral_activation_confidence=0.80,
      durable_lateral_activation_confirm_frames=2,
      durable_lateral_activation_immediate_confidence=0.95,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.84,
    )
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_left",)), 0.84))
    state = DurableLateralOverrideState()
    updated = update_durable_lateral_plans(
      {},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=42,
    )
    self.assertEqual(updated, {})
    self.assertEqual(state.pending_count, 1)
    self.assertEqual(state.pending_observation_id, 42)

    updated = update_durable_lateral_plans(
      updated,
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=42,
    )
    self.assertEqual(updated, {})
    self.assertEqual(state.pending_count, 1)

    updated = update_durable_lateral_plans(
      updated,
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=44,
    )
    self.assertIn("left_edge_s8_48_margin1.25", updated)

  def test_unactivated_lateral_plan_does_not_use_compiled_fallback_by_default(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_lateral_activation_confidence=0.80,
      durable_lateral_activation_confirm_frames=2,
      durable_lateral_activation_immediate_confidence=0.95,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
      allow_compiled_lateral_fallback=False,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.84,
    )
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_left",)), 0.84))
    state = DurableLateralOverrideState()
    updated = update_durable_lateral_plans(
      {},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
    )
    desired, fallback_used = compose_lateral_offset_after_publish(
      updated,
      current_long_m=10.0,
      max_offset_m=args.max_durable_offset_m,
      compiled_lateral_offset_openpilot_m=-1.25,
      args=args,
    )
    self.assertEqual(updated, {})
    self.assertEqual(desired, 0.0)
    self.assertFalse(fallback_used)

  def test_compiled_lateral_fallback_is_explicit_opt_in(self):
    args = Namespace(
      max_durable_offset_m=1.3,
      allow_compiled_lateral_fallback=True,
    )
    desired, fallback_used = compose_lateral_offset_after_publish(
      {},
      current_long_m=10.0,
      max_offset_m=args.max_durable_offset_m,
      compiled_lateral_offset_openpilot_m=-1.25,
      args=args,
    )
    self.assertAlmostEqual(desired, 1.25)
    self.assertTrue(fallback_used)

  def test_very_confident_lateral_plan_activates_without_repetition(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_lateral_activation_confidence=0.80,
      durable_lateral_activation_confirm_frames=3,
      durable_lateral_activation_immediate_confidence=0.95,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.96,
    )
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_right",)), 0.96))
    state = DurableLateralOverrideState()
    updated = update_durable_lateral_plans(
      {},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
    )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertEqual(state.pending_source_token, "")
    self.assertEqual(state.pending_count, 0)

  def test_single_strong_construction_contradiction_does_not_reverse_active_plan_by_default(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_lateral_activation_confidence=0.80,
      durable_lateral_activation_confirm_frames=3,
      durable_lateral_activation_immediate_confidence=0.95,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      durable_construction_conflict_immediate_confidence=0.99,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.96,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.96,
    )
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_left",)), 0.96))
    state = DurableLateralOverrideState()
    updated = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": existing},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=50,
    )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertNotIn("left_edge_s8_48_margin1.25", updated)
    self.assertEqual(state.pending_source_token, "left_edge_s8_48_margin1.25")
    self.assertEqual(state.pending_count, 1)
    self.assertEqual(state.pending_observation_id, 50)

  def test_default_construction_conflict_threshold_accepts_state_machine_corrected_side(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_lateral_activation_confidence=0.80,
      durable_lateral_activation_confirm_frames=3,
      durable_lateral_activation_immediate_confidence=0.95,
      durable_conflict_override_confidence=0.90,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.96,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.96,
    )
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_left",)), 0.96))
    updated = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": existing},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=DurableLateralOverrideState(),
      rtp_source_frame_id=18,
    )
    self.assertNotIn("right_edge_s8_48_margin1.25", updated)
    self.assertIn("left_edge_s8_48_margin1.25", updated)
    self.assertGreater(compose_lateral_offset(updated, 10.0, args.max_durable_offset_m), 0.0)

  def test_single_low_confidence_contradictory_lateral_plan_is_held_pending(self):
    args = Namespace(
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.90,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.72,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.72,
    )
    state = DurableLateralOverrideState()
    program = parse_rtp(_labels_to_rtp(("construction_left",)))
    updated = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": existing},
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
    )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertNotIn("left_edge_s8_48_margin1.25", updated)
    self.assertEqual(state.pending_source_token, "")
    self.assertEqual(state.pending_count, 0)

  def test_repeated_low_confidence_contradictory_lateral_plan_does_not_override(self):
    args = Namespace(
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.90,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.72,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.72,
    )
    state = DurableLateralOverrideState()
    program = parse_rtp(_labels_to_rtp(("construction_left",)))
    updated = {"right_edge_s8_48_margin1.25": existing}
    for _ in range(3):
      updated = update_durable_lateral_plans(
        updated,
        new,
        program,
        current_long_m=10.0,
        args=args,
        override_state=state,
      )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertNotIn("left_edge_s8_48_margin1.25", updated)
    self.assertEqual(state.pending_source_token, "")
    self.assertEqual(state.pending_count, 0)

  def test_repeated_confident_contradictory_lateral_plan_overrides(self):
    args = Namespace(
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.72,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.84,
    )
    state = DurableLateralOverrideState()
    program = parse_rtp(_labels_to_rtp(("construction_left",)))
    updated = {"right_edge_s8_48_margin1.25": existing}
    for _ in range(3):
      updated = update_durable_lateral_plans(
        updated,
        new,
        program,
        current_long_m=10.0,
        args=args,
        override_state=state,
      )
    self.assertNotIn("right_edge_s8_48_margin1.25", updated)
    self.assertIn("left_edge_s8_48_margin1.25", updated)

  def test_stale_reused_rtp_does_not_count_as_repeated_lateral_conflict_override(self):
    args = Namespace(
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=2,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.72,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.84,
    )
    state = DurableLateralOverrideState()
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_left",)), 0.84))
    updated = {"right_edge_s8_48_margin1.25": existing}
    updated = update_durable_lateral_plans(
      updated,
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=42,
    )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertEqual(state.pending_count, 1)

    updated = update_durable_lateral_plans(
      updated,
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=42,
    )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertNotIn("left_edge_s8_48_margin1.25", updated)
    self.assertEqual(state.pending_count, 1)

    updated = update_durable_lateral_plans(
      updated,
      new,
      program,
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=44,
    )
    self.assertNotIn("right_edge_s8_48_margin1.25", updated)
    self.assertIn("left_edge_s8_48_margin1.25", updated)

  def test_absent_lateral_evidence_resets_pending_conflict_override(self):
    args = Namespace(
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.80,
      durable_conflict_confirm_frames=2,
      durable_conflict_immediate_confidence=0.95,
      durable_construction_conflict_immediate_confidence=0.99,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.96,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.96,
    )
    state = DurableLateralOverrideState()
    updated = {"right_edge_s8_48_margin1.25": existing}
    updated = update_durable_lateral_plans(
      updated,
      new,
      parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_left",)), 0.96)),
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=42,
    )
    self.assertEqual(state.pending_count, 1)

    updated = update_durable_lateral_plans(
      updated,
      None,
      parse_rtp(_labels_to_rtp(("cones",))),
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=44,
    )
    self.assertEqual(state.pending_source_token, "")
    self.assertEqual(state.pending_count, 0)

    updated = update_durable_lateral_plans(
      updated,
      new,
      parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_left",)), 0.96)),
      current_long_m=10.0,
      args=args,
      override_state=state,
      rtp_source_frame_id=46,
    )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertNotIn("left_edge_s8_48_margin1.25", updated)
    self.assertEqual(state.pending_count, 1)

  def test_moderate_confidence_contradictory_lateral_plan_is_held_by_production_default(self):
    args = Namespace(
      durable_override_confidence=0.74,
      durable_conflict_override_confidence=0.90,
      durable_conflict_confirm_frames=3,
      durable_conflict_immediate_confidence=0.95,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=0.0,
      end_long_m=40.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=50.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.72,
    )
    new = DurableAvoidance(
      start_long_m=5.0,
      end_long_m=45.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=55.0,
      offset_m=1.25,
      source_token="left_edge_s8_48_margin1.25",
      source_meta="BIAS_RIGHT",
      confidence=0.84,
    )
    state = DurableLateralOverrideState()
    program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("construction_left",)), 0.84))
    updated = {"right_edge_s8_48_margin1.25": existing}
    for _ in range(3):
      updated = update_durable_lateral_plans(
        updated,
        new,
        program,
        current_long_m=10.0,
        args=args,
        override_state=state,
      )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertNotIn("left_edge_s8_48_margin1.25", updated)

  def test_neutral_base_at_threshold_does_not_clear_lateral_plan(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=8.0,
      end_long_m=48.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=58.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.72,
    )
    base_program = parse_rtp(_labels_to_rtp(("none",)))
    updated = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": existing},
      None,
      base_program,
      current_long_m=12.0,
      args=args,
    )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertLess(compose_lateral_offset(updated, 12.0, args.max_durable_offset_m), 0.0)

  def test_confirmed_construction_clear_base_program_clears_lateral_plan(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=8.0,
      end_long_m=48.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=58.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.72,
    )
    clear_program = parse_rtp(_with_rtp_confidence(_labels_to_rtp(("none",)), CONSTRUCTION_CLEAR_RTP_CONFIDENCE))
    updated = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": existing},
      None,
      clear_program,
      current_long_m=12.0,
      args=args,
    )
    self.assertEqual(updated, {})

  def test_construction_presence_unknown_does_not_clear_lateral_plan(self):
    args = Namespace(
      durable_override_confidence=0.70,
      durable_conflict_override_confidence=0.70,
      max_durable_offset_m=1.3,
    )
    existing = DurableAvoidance(
      start_long_m=8.0,
      end_long_m=48.0,
      ramp_in_start_long_m=0.0,
      ramp_out_end_long_m=58.0,
      offset_m=-1.25,
      source_token="right_edge_s8_48_margin1.25",
      source_meta="BIAS_LEFT",
      confidence=0.80,
    )
    presence_without_side = parse_rtp(_labels_to_rtp(("cones",)))
    updated = update_durable_lateral_plans(
      {"right_edge_s8_48_margin1.25": existing},
      None,
      presence_without_side,
      current_long_m=12.0,
      args=args,
    )
    self.assertIn("right_edge_s8_48_margin1.25", updated)
    self.assertLess(compose_lateral_offset(updated, 12.0, args.max_durable_offset_m), 0.0)


if __name__ == "__main__":
  unittest.main()
