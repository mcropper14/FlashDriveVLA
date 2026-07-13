from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.alpamayo import alpamayod
from openpilot.selfdrive.alpamayo.alpamayod import CapturedFrame, ProviderContext, RemoteServerProvider
from openpilot.selfdrive.alpamayo.protocol import encode_payload, xyzt_to_dict
from openpilot.selfdrive.modeld.constants import ModelConstants


def _trajectory(offset: float) -> dict[str, list[float]]:
  idx = np.arange(ModelConstants.IDX_N, dtype=np.float32)
  values = np.column_stack((idx + offset, idx + offset + 10.0, idx + offset + 20.0)).astype(np.float32)
  return xyzt_to_dict(np.asarray(ModelConstants.T_IDXS, dtype=np.float32), values)


def _response() -> dict:
  return {
    "semanticPlan": {
      "source": "remoteServer",
      "status": "valid",
      "confidence": 0.9,
      "consistency": 0.95,
      "desiredCurvature": 0.01,
      "desiredAcceleration": 0.2,
      "shouldStop": False,
      "trajectory": {
        "position": _trajectory(1.0),
        "orientation": _trajectory(2.0),
        "velocity": _trajectory(3.0),
        "orientationRate": _trajectory(4.0),
        "acceleration": _trajectory(5.0),
      },
    },
  }


def _frame(stream: str, frame_id: int) -> CapturedFrame:
  return CapturedFrame(
    stream=stream,
    frame_id=frame_id,
    timestamp_sof=frame_id * 50_000_000,
    timestamp_eof=frame_id * 50_000_000 + 45_000_000,
    width=2,
    height=2,
    stride=2,
    uv_offset=4,
    data=b"\x00" * 6,
  )


def _ctx() -> ProviderContext:
  frame_bundle = {
    "wideRoad": [_frame("wideRoad", 10), _frame("wideRoad", 12)],
    "road": [_frame("road", 10), _frame("road", 12)],
  }
  return ProviderContext(
    model_msg=SimpleNamespace(),
    model_mono_time=123,
    car_state=SimpleNamespace(vEgo=2.0, aEgo=0.0, standstill=False, steeringAngleDeg=0.0,
                              gasPressed=False, brakePressed=False),
    selfdrive_state=SimpleNamespace(enabled=True, active=True, experimentalMode=False),
    nav_instruction=None,
    device_state=SimpleNamespace(deviceType="pc"),
    road_camera_state=SimpleNamespace(sensor="unknown"),
    live_calibration=SimpleNamespace(rpyCalib=[]),
    driver_monitoring_state=SimpleNamespace(isRHD=False),
    frame_bundle=frame_bundle,
    ego_history_xyz=np.zeros((16, 3), dtype=np.float32),
    ego_history_rot=np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0),
    frame_t0_s=0.6,
  )


def test_remote_response_must_contain_complete_alpamayo_trajectory():
  provider = RemoteServerProvider("http://127.0.0.1:1")

  valid = provider._response_to_plan(_response(), SimpleNamespace())
  assert valid is not None
  assert valid.position.shape == (ModelConstants.IDX_N, 3)

  incomplete = _response()
  del incomplete["semanticPlan"]["trajectory"]["velocity"]
  assert provider._response_to_plan(incomplete, SimpleNamespace()) is None


def test_remote_request_does_not_send_stock_plan(monkeypatch):
  monkeypatch.setattr(
    alpamayod,
    "_camera_context",
    lambda ctx: {
      "deviceType": "pc",
      "sensor": "unknown",
      "calibrationRpy": [0.0, 0.0, 0.0],
      "roadIntrinsics": np.eye(3, dtype=np.float32).tolist(),
      "wideIntrinsics": np.eye(3, dtype=np.float32).tolist(),
      "roadWarpMatrix": np.eye(3, dtype=np.float32).tolist(),
      "wideWarpMatrix": np.eye(3, dtype=np.float32).tolist(),
      "frameSyncToleranceNs": alpamayod.FRAME_SYNC_TOLERANCE_NS,
    },
  )
  monkeypatch.setattr(
    alpamayod,
    "_encode_transport_frame",
    lambda frame: {
      "stream": frame.stream,
      "encoding": "test",
      "frameId": frame.frame_id,
      "timestampSof": frame.timestamp_sof,
      "timestampEof": frame.timestamp_eof,
      "width": frame.width,
      "height": frame.height,
      "dataBase64": "",
    },
  )
  provider = RemoteServerProvider("http://127.0.0.1:1")

  ctx = _ctx()
  request = provider._build_request(ctx)

  assert request is not None
  assert request["semanticPlanHz"] == 10
  assert "stockPlan" not in request
  assert "frames" in request
  assert request["runtimeConfig"]["transportEncoding"] == "jpeg_bgr"
  assert request["runtimeConfig"]["reasoningMode"] == "full"
  assert request["cameraBundle"]["streamOrder"] == ["wideRoad", "road"]
  assert request["egoHistory"]["xyz"] == ctx.ego_history_xyz.tolist()


def test_remote_build_drops_failed_cycle_instead_of_reusing_cached_plan(monkeypatch):
  provider = RemoteServerProvider("http://127.0.0.1:1")
  monkeypatch.setattr(provider, "_build_request", lambda ctx: {"protocolVersion": 1})

  class GoodResponse:
    content = encode_payload(_response())

    @staticmethod
    def raise_for_status():
      return None

  calls = {"count": 0}

  def post(*args, **kwargs):
    calls["count"] += 1
    if calls["count"] == 1:
      return GoodResponse()
    raise RuntimeError("network down")

  provider.session = SimpleNamespace(post=post)

  assert provider.build(_ctx()) is not None
  assert provider.build(_ctx()) is None
  assert not hasattr(provider, "last_valid_plan")
