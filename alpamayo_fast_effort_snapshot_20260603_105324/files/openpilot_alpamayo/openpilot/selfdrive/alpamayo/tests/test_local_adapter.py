from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from openpilot.selfdrive.alpamayo.local_adapter import (
  _assert_no_cpu_offload,
  _euler_from_rot_mats,
  LocalAlpamayoAdapter,
  LocalAlpamayoConfig,
  semantic_response_from_prediction,
)
from openpilot.selfdrive.alpamayo.pc_endpoint import validate_response


def test_semantic_response_from_prediction_is_complete_contract():
  t = np.arange(64, dtype=np.float32) * 0.1
  pred_xyz = np.column_stack([
    t * 10.0,
    np.sin(t) * 0.25,
    np.zeros_like(t),
  ]).astype(np.float32)
  pred_rot = np.repeat(np.eye(3, dtype=np.float32)[None], 64, axis=0)

  response = semantic_response_from_prediction(pred_xyz, pred_rot, source="remoteServer")

  validate_response(response)
  semantic = response["semanticPlan"]
  assert semantic["status"] == "valid"
  assert semantic["source"] == "remoteServer"
  assert set(semantic["trajectory"]) == {"position", "orientation", "velocity", "orientationRate", "acceleration"}
  assert len(semantic["trajectory"]["position"]["t"]) == 64
  assert semantic["trajectory"]["position"]["x"][0] == 0.0
  assert semantic["trajectory"]["velocity"]["x"][0] > 9.0


def test_euler_from_rot_mats_extracts_yaw():
  yaw = np.float32(0.25)
  c = np.cos(yaw)
  s = np.sin(yaw)
  rot = np.asarray([
    [
      [c, -s, 0.0],
      [s, c, 0.0],
      [0.0, 0.0, 1.0],
    ]
  ], dtype=np.float32)

  euler = _euler_from_rot_mats(rot)

  assert np.isclose(euler[0, 2], yaw)


def test_assert_no_cpu_offload_rejects_cpu_or_disk_device_map():
  class Model:
    hf_device_map = {
      "vlm.model.visual": 0,
      "vlm.model.language_model.layers.0": "cpu",
      "vlm.model.language_model.layers.1": "disk",
    }

  with pytest.raises(RuntimeError, match="CPU/disk offload is forbidden"):
    _assert_no_cpu_offload(Model())


class _FakeAlpamayoModel:
  def __init__(self, generated_sequence_length: int):
    self.generated_sequence_length = generated_sequence_length

  def sample_trajectories_from_data_with_vlm_rollout(self, *args, **kwargs):
    pred_xyz = torch.zeros((1, 1, 1, 64, 3), dtype=torch.float32)
    pred_rot = torch.zeros((1, 1, 1, 64, 3, 3), dtype=torch.float32)
    for idx in range(64):
      pred_rot[0, 0, 0, idx, idx % 3, idx % 3] = 1.0
    return pred_xyz, pred_rot, {"generated_sequence_length": self.generated_sequence_length}


def _fake_model_inputs(generated_input_len: int, missing_visual: bool = False):
  tokenized_data = {
    "input_ids": torch.ones((1, generated_input_len), dtype=torch.long),
    "image_grid_thw": torch.ones((generated_input_len, 3), dtype=torch.long),
    "pixel_values": torch.zeros((generated_input_len, 3, 4, 4), dtype=torch.float32),
  }
  if missing_visual:
    del tokenized_data["pixel_values"]
  return {
    "tokenized_data": tokenized_data,
    "ego_history_xyz": torch.zeros((1, 1, 16, 3), dtype=torch.float32),
    "ego_history_rot": torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(16, 1, 1).unsqueeze(0),
  }


def _make_adapter(config: LocalAlpamayoConfig | None = None) -> LocalAlpamayoAdapter:
  adapter = LocalAlpamayoAdapter(config=config)
  adapter._torch = torch
  adapter._model = _FakeAlpamayoModel(generated_sequence_length=1)
  adapter._ensure_loaded = lambda: None
  return adapter


def _with_model_inputs(adapter: LocalAlpamayoAdapter, generated_input_len: int, generated_sequence_length: int, missing_visual: bool = False):
  adapter._model = _FakeAlpamayoModel(generated_sequence_length=generated_sequence_length)
  adapter._build_model_inputs = lambda request: (
    _fake_model_inputs(generated_input_len, missing_visual=missing_visual),
    {"total_frames": 1},
  )


def test_infer_rejects_skip_vlm_generation_shortcut():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path("."), skip_vlm_generation=True))
  _with_model_inputs(adapter, generated_input_len=2, generated_sequence_length=3)
  with pytest.raises(RuntimeError, match="skip_vlm_generation is forbidden"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})


def test_infer_rejects_missing_visual_inputs():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  _with_model_inputs(adapter, generated_input_len=2, generated_sequence_length=3, missing_visual=True)
  with pytest.raises(RuntimeError, match="missing required visual tokenized field: pixel_values"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})


def test_infer_rejects_missing_runtime_config_reasoning_mode():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  _with_model_inputs(adapter, generated_input_len=2, generated_sequence_length=3)
  with pytest.raises(RuntimeError, match="runtimeConfig.reasoningMode missing"):
    adapter.infer({"runtimeConfig": {}})


def test_infer_rejects_missing_runtime_config_entirely():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  _with_model_inputs(adapter, generated_input_len=2, generated_sequence_length=3)
  with pytest.raises(RuntimeError, match="runtimeConfig missing"):
    adapter.infer({})


def test_infer_rejects_skip_vlm_generation_runtime_config_shortcut():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  _with_model_inputs(adapter, generated_input_len=2, generated_sequence_length=3)
  with pytest.raises(RuntimeError, match="runtimeConfig.skipVlmGeneration is forbidden"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "full", "skipVlmGeneration": True}})


def test_infer_rejects_skip_vlm_generation_runtime_config_shortcut_whitespace():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  _with_model_inputs(adapter, generated_input_len=2, generated_sequence_length=3)
  with pytest.raises(RuntimeError, match="runtimeConfig.skipVlmGeneration is forbidden"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "full", "skipVlmGeneration": " on "}})


def test_infer_rejects_invalid_visual_tensor_shape():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  adapter._build_model_inputs = lambda request: (
    {
      "tokenized_data": {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "pixel_values": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "image_grid_thw": torch.ones((1, 3), dtype=torch.long),
      },
      "ego_history_xyz": torch.zeros((1, 1, 16, 3), dtype=torch.float32),
      "ego_history_rot": torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(16, 1, 1).unsqueeze(0),
    },
    {"total_frames": 1},
  )
  adapter._model = _FakeAlpamayoModel(generated_sequence_length=3)
  with pytest.raises(RuntimeError, match="pixel_values must be \\[N, 3, H, W\\]|pixel_values channel count must be 3"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})


def test_infer_rejects_mismatched_pixel_and_grid_batch_counts():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  adapter._model = _FakeAlpamayoModel(generated_sequence_length=3)
  adapter._build_model_inputs = lambda request: ({
    "tokenized_data": {
      "input_ids": torch.ones((1, 2), dtype=torch.long),
      "pixel_values": torch.zeros((2, 3, 4, 4), dtype=torch.float32),
      "image_grid_thw": torch.ones((1, 3), dtype=torch.long),
    },
    "ego_history_xyz": torch.zeros((1, 1, 16, 3), dtype=torch.float32),
    "ego_history_rot": torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(16, 1, 1).unsqueeze(0),
  }, {"total_frames": 1})
  with pytest.raises(RuntimeError, match="pixel_values batch count must match image_grid_thw"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})


def test_infer_rejects_prefill_only_runtime_mode():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  _with_model_inputs(adapter, generated_input_len=2, generated_sequence_length=3)
  with pytest.raises(RuntimeError, match="runtimeConfig.reasoningMode must be 'full'"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "prefill_only"}})


def test_infer_rejects_reasoning_generation_without_expansion():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  _with_model_inputs(adapter, generated_input_len=4, generated_sequence_length=4)
  with pytest.raises(RuntimeError, match="skipping VLM reasoning output is forbidden"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})


def test_infer_rejects_mismatched_deepstack_image_token_count():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  adapter._processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
  adapter._model = _FakeAlpamayoModel(generated_sequence_length=7)
  adapter._model.vlm = SimpleNamespace(config=SimpleNamespace(image_token_id=2))

  def _tokenized_with_unmatched_image_tokens():
    return {
      "tokenized_data": {
        "input_ids": torch.tensor([[1, 1, 2, 1, 1, 1]], dtype=torch.long),
        "image_grid_thw": torch.tensor([[4, 4, 4]], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
      },
      "ego_history_xyz": torch.zeros((1, 1, 16, 3), dtype=torch.float32),
      "ego_history_rot": torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(16, 1, 1).unsqueeze(0),
    }

  adapter._build_model_inputs = lambda request: (_tokenized_with_unmatched_image_tokens(), {"total_frames": 1})
  with pytest.raises(RuntimeError, match="deepstack image token count does not match image_grid_thw"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})


def test_infer_accepts_matching_deepstack_image_token_count():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  adapter._processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
  adapter._model = _FakeAlpamayoModel(generated_sequence_length=18)
  adapter._model.vlm = SimpleNamespace(config=SimpleNamespace(image_token_id=2))

  def _tokenized_with_matching_image_tokens():
    return {
      "tokenized_data": {
        "input_ids": torch.tensor([[2] * 16], dtype=torch.long),
        "image_grid_thw": torch.tensor([[4, 4, 4]], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
      },
      "ego_history_xyz": torch.zeros((1, 1, 16, 3), dtype=torch.float32),
      "ego_history_rot": torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(16, 1, 1).unsqueeze(0),
    }

  adapter._build_model_inputs = lambda request: (_tokenized_with_matching_image_tokens(), {"total_frames": 1})
  response = adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})

  assert response["semanticPlan"]["status"] == "valid"


def test_infer_rejects_missing_image_token_id_metadata():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  adapter._processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
  adapter._model = _FakeAlpamayoModel(generated_sequence_length=7)
  adapter._model.vlm = SimpleNamespace(config=SimpleNamespace())

  adapter._build_model_inputs = lambda request: ({
    "tokenized_data": {
      "input_ids": torch.tensor([[2, 2]], dtype=torch.long),
      "image_grid_thw": torch.tensor([[4, 4, 4]], dtype=torch.long),
      "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
    },
    "ego_history_xyz": torch.zeros((1, 1, 16, 3), dtype=torch.float32),
    "ego_history_rot": torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(16, 1, 1).unsqueeze(0),
  }, {"total_frames": 1})

  with pytest.raises(RuntimeError, match="model.vlm.config.image_token_id missing required multimodal metadata"):
    adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})


def test_infer_records_deepstack_visual_and_reasoning_evidence():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  _with_model_inputs(adapter, generated_input_len=2, generated_sequence_length=5)

  response = adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})
  semantic = response["semanticPlan"]
  debug = semantic["debug"]

  assert semantic["status"] == "valid"
  assert debug["skipVlmGeneration"] is False
  assert debug["reasoningMode"] == "full"
  assert debug["reasoningGeneratedTokens"] == 3
  assert debug["deepstackInputShapes"]["inputIdsShape"] == [1, 2]
  assert debug["deepstackInputShapes"]["imageGridThwShape"] == [2, 3]
  assert len(debug["deepstackInputShapes"]["pixelValuesShape"]) == 4


def test_infer_records_matching_deepstack_token_counts():
  adapter = _make_adapter(LocalAlpamayoConfig(Path("."), Path(".")))
  _with_model_inputs(adapter, generated_input_len=6, generated_sequence_length=7)

  response = adapter.infer({"runtimeConfig": {"reasoningMode": "full"}})
  debug = response["semanticPlan"]["debug"]

  assert debug["deepstackInputShapes"]["inputIdsShape"][1] == 6
  assert debug["deepstackInputShapes"]["imageGridThwShape"][0] == 6
