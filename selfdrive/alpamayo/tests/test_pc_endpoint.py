import json

from openpilot.selfdrive.alpamayo.pc_endpoint import AlpamayoPcEndpoint, PcTraceLogger
from openpilot.selfdrive.alpamayo.protocol import decode_payload, encode_payload
from openpilot.selfdrive.alpamayo.tests.test_remote_contract import _ctx, _response


def _response_with_deepstack() -> dict:
  response = _response()
  response["semanticPlan"]["debug"] = {
    "reasoningMode": "full",
    "skipVlmGeneration": False,
    "reasoningGeneratedTokens": 3,
    "deepstackInputShapes": {
      "inputIdsShape": [2, 2],
      "pixelValuesShape": [2, 3, 4, 4],
      "imageGridThwShape": [2, 3],
    },
  }
  return response


def _request() -> dict:
  ctx = _ctx()
  frames = []
  for stream_frames in ctx.frame_bundle.values():
    for frame in stream_frames:
      frames.append({
        "stream": frame.stream,
        "encoding": "test",
        "frameId": frame.frame_id,
        "timestampSof": frame.timestamp_sof,
        "timestampEof": frame.timestamp_eof,
        "width": frame.width,
        "height": frame.height,
        "dataBase64": "",
      })

  return {
    "protocolVersion": 1,
    "semanticPlanHz": 10,
    "cameraBundle": {
      "streamOrder": ["wideRoad", "road"],
      "framesPerCamera": 2,
      "frameStepNs": 100_000_000,
    },
    "frames": frames,
    "egoHistory": {
      "dtS": 0.1,
      "xyz": [[[0.0, 0.0, 0.0]][0] for _ in range(16)],
      "rot": [
        [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0],
        ]
        for _ in range(16)
      ],
    },
    "runtimeConfig": {
      "cameraMode": "front2",
      "numFrames": 2,
      "minPixels": 65536,
      "maxPixels": 65536,
      "reasoningMode": "full",
      "diffusionSteps": 6,
      "numTrajSamples": 1,
    },
    "vehicleState": {
      "vEgo": 2.0,
      "aEgo": 0.0,
      "standstill": False,
      "steeringAngleDeg": 0.0,
    },
  }


class GoodAdapter:
  def infer(self, request):
    return _response_with_deepstack()


class IncompleteAdapter:
  def infer(self, request):
    response = _response_with_deepstack()
    del response["semanticPlan"]["trajectory"]["velocity"]
    return response


class NumericEnumAdapter:
  def infer(self, request):
    response = _response_with_deepstack()
    response["semanticPlan"]["source"] = 2
    response["semanticPlan"]["status"] = 1
    return response


def _rows(path):
  return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pc_endpoint_without_adapter_returns_unavailable_without_trajectory(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  endpoint = AlpamayoPcEndpoint(trace_logger=PcTraceLogger(trace_path))

  status, body, content_type = endpoint.handle_payload(encode_payload(_request()))
  response = decode_payload(body)

  assert status == 503
  assert content_type == "application/x-alpamayo-response+zstd"
  assert response["semanticPlan"]["status"] == "unavailable"
  assert "trajectory" not in response["semanticPlan"]
  assert _rows(trace_path)[0]["outcome"] == "adapter_unavailable"


def test_pc_endpoint_returns_valid_adapter_trajectory(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  endpoint = AlpamayoPcEndpoint(GoodAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(_request()))
  response = decode_payload(body)

  assert status == 200
  assert response["semanticPlan"]["status"] == "valid"
  assert "trajectory" in response["semanticPlan"]
  assert _rows(trace_path)[0]["outcome"] == "valid"


def test_pc_endpoint_accepts_numeric_capnp_enum_values(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  endpoint = AlpamayoPcEndpoint(NumericEnumAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(_request()))
  response = decode_payload(body)

  assert status == 200
  assert response["semanticPlan"]["source"] == 2
  assert response["semanticPlan"]["status"] == 1
  assert "trajectory" in response["semanticPlan"]
  assert _rows(trace_path)[0]["outcome"] == "valid"


def test_pc_endpoint_rejects_missing_reasoning_mode(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()
  request["runtimeConfig"].pop("reasoningMode")
  endpoint = AlpamayoPcEndpoint(GoodAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "runtimeConfig.reasoningMode missing"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_missing_runtime_config_entirely(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()
  request.pop("runtimeConfig")
  endpoint = AlpamayoPcEndpoint(GoodAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "runtimeConfig missing"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_nonfull_reasoning_mode(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()
  request["runtimeConfig"]["reasoningMode"] = "prefill_only"
  endpoint = AlpamayoPcEndpoint(GoodAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "runtimeConfig.reasoningMode must be 'full'"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_shortcut_vlm_skip_flag(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()
  request["runtimeConfig"]["skipVlmGeneration"] = True
  endpoint = AlpamayoPcEndpoint(GoodAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "runtimeConfig.skipVlmGeneration is forbidden for production Alpamayo"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_shortcut_vlm_skip_flag_with_whitespace_true(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()
  request["runtimeConfig"]["skipVlmGeneration"] = " on "
  endpoint = AlpamayoPcEndpoint(GoodAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "runtimeConfig.skipVlmGeneration is forbidden for production Alpamayo"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_stock_plan_in_request(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()
  request["stockPlan"] = {"position": {}}
  endpoint = AlpamayoPcEndpoint(GoodAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert "stockPlan is forbidden" in response["semanticPlan"]["error"]
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_incomplete_adapter_trajectory(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  endpoint = AlpamayoPcEndpoint(IncompleteAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(_request()))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert "trajectory.velocity invalid" in response["semanticPlan"]["error"]
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_adapter_response_without_deepstack_evidence(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()

  class NoDeepstackAdapter:
    def infer(self, request):
      return _response()

  endpoint = AlpamayoPcEndpoint(NoDeepstackAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "debug missing"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_adapter_response_with_deepstack_shortcut(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()

  class ShortcutAdapter:
    def infer(self, request):
      response = _response_with_deepstack()
      response["semanticPlan"]["debug"]["skipVlmGeneration"] = True
      return response

  endpoint = AlpamayoPcEndpoint(ShortcutAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "debug.skipVlmGeneration is forbidden for production Alpamayo"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_adapter_response_with_missing_deepstack_shapes(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()

  class MissingDeepstackShapesAdapter:
    def infer(self, request):
      response = _response_with_deepstack()
      del response["semanticPlan"]["debug"]["deepstackInputShapes"]
      return response

  endpoint = AlpamayoPcEndpoint(MissingDeepstackShapesAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "debug.deepstackInputShapes missing"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_adapter_response_with_missing_deepstack_shape_entry(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()

  class MissingDeepstackShapeEntryAdapter:
    def infer(self, request):
      response = _response_with_deepstack()
      del response["semanticPlan"]["debug"]["deepstackInputShapes"]["pixelValuesShape"]
      return response

  endpoint = AlpamayoPcEndpoint(MissingDeepstackShapeEntryAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "debug.deepstackInputShapes.pixelValuesShape missing"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_adapter_response_with_nonarray_deepstack_shape(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()

  class NonArrayDeepstackShapesAdapter:
    def infer(self, request):
      response = _response_with_deepstack()
      response["semanticPlan"]["debug"]["deepstackInputShapes"]["pixelValuesShape"] = {}
      return response

  endpoint = AlpamayoPcEndpoint(NonArrayDeepstackShapesAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "debug.deepstackInputShapes.pixelValuesShape must be an array of positive integers"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_adapter_response_with_invalid_deepstack_shape_rank(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()

  class InvalidDeepstackShapeRankAdapter:
    def infer(self, request):
      response = _response_with_deepstack()
      response["semanticPlan"]["debug"]["deepstackInputShapes"]["inputIdsShape"] = [1, 2, 3]
      return response

  endpoint = AlpamayoPcEndpoint(InvalidDeepstackShapeRankAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "debug.deepstackInputShapes.inputIdsShape must have 2 dimensions"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_adapter_response_with_deepstack_channel_mismatch(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()

  class ChannelMismatchAdapter:
    def infer(self, request):
      response = _response_with_deepstack()
      response["semanticPlan"]["debug"]["deepstackInputShapes"]["pixelValuesShape"] = [2, 4, 4, 4]
      return response

  endpoint = AlpamayoPcEndpoint(ChannelMismatchAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "debug.deepstackInputShapes.pixelValuesShape channel dimension must be 3"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_adapter_response_with_deepstack_negative_shape(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()

  class NegativeDeepstackShapeAdapter:
    def infer(self, request):
      response = _response_with_deepstack()
      response["semanticPlan"]["debug"]["deepstackInputShapes"]["imageGridThwShape"] = [0, 3]
      return response

  endpoint = AlpamayoPcEndpoint(NegativeDeepstackShapeAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "debug.deepstackInputShapes.imageGridThwShape must contain positive integers"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"


def test_pc_endpoint_rejects_adapter_response_with_mismatched_deepstack_token_counts(tmp_path):
  trace_path = tmp_path / "pc_trace.jsonl"
  request = _request()

  class MismatchedDeepstackTokenCountsAdapter:
    def infer(self, request):
      response = _response_with_deepstack()
      response["semanticPlan"]["debug"]["deepstackInputShapes"]["inputIdsShape"] = [1, 2]
      response["semanticPlan"]["debug"]["deepstackInputShapes"]["imageGridThwShape"] = [2, 3]
      return response

  endpoint = AlpamayoPcEndpoint(MismatchedDeepstackTokenCountsAdapter(), PcTraceLogger(trace_path))

  status, body, _ = endpoint.handle_payload(encode_payload(request))
  response = decode_payload(body)

  assert status == 400
  assert response["semanticPlan"]["status"] == "error"
  assert response["semanticPlan"]["error"] == "debug.deepstackInputShapes token count mismatch between inputIds and imageGrid"
  assert _rows(trace_path)[0]["outcome"] == "contract_error"
