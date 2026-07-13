import base64
import json
from typing import Any

import numpy as np
import zstandard as zstd


PROTOCOL_VERSION = 1
REQUEST_CONTENT_TYPE = "application/x-alpamayo-request+zstd"
RESPONSE_CONTENT_TYPE = "application/x-alpamayo-response+zstd"
FRAME_ENCODING_NV12 = "nv12"
FRAME_ENCODING_JPEG_BGR = "jpeg_bgr"

_COMPRESSOR = zstd.ZstdCompressor(level=3)
_DECOMPRESSOR = zstd.ZstdDecompressor()


def _float_list(values) -> list[float]:
  return np.asarray(values, dtype=np.float32).tolist()


def builder_to_xyzt(builder) -> dict[str, list[float]] | None:
  lengths = [len(getattr(builder, axis)) for axis in ("t", "x", "y", "z")]
  if len(set(lengths)) != 1 or lengths[0] < 2:
    return None

  return {
    "t": _float_list(builder.t),
    "x": _float_list(builder.x),
    "y": _float_list(builder.y),
    "z": _float_list(builder.z),
  }


def xyzt_to_dict(t: np.ndarray, values: np.ndarray) -> dict[str, list[float]]:
  return {
    "t": _float_list(t),
    "x": _float_list(values[:, 0]),
    "y": _float_list(values[:, 1]),
    "z": _float_list(values[:, 2]),
  }


def serialize_nv12_frame(stream: str, data: bytes, width: int, height: int, stride: int, uv_offset: int,
                         frame_id: int, timestamp_sof: int, timestamp_eof: int) -> dict[str, Any]:
  return {
    "stream": stream,
    "encoding": FRAME_ENCODING_NV12,
    "frameId": int(frame_id),
    "timestampSof": int(timestamp_sof),
    "timestampEof": int(timestamp_eof),
    "width": int(width),
    "height": int(height),
    "stride": int(stride),
    "uvOffset": int(uv_offset),
    "dataBase64": base64.b64encode(data).decode("ascii"),
  }


def serialize_jpeg_bgr_frame(stream: str, data: bytes, width: int, height: int,
                             frame_id: int, timestamp_sof: int, timestamp_eof: int) -> dict[str, Any]:
  return {
    "stream": stream,
    "encoding": FRAME_ENCODING_JPEG_BGR,
    "frameId": int(frame_id),
    "timestampSof": int(timestamp_sof),
    "timestampEof": int(timestamp_eof),
    "width": int(width),
    "height": int(height),
    "dataBase64": base64.b64encode(data).decode("ascii"),
  }


def encode_payload(payload: dict[str, Any]) -> bytes:
  return _COMPRESSOR.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def decode_payload(data: bytes) -> dict[str, Any]:
  body = data
  if body[:1] not in (b"{", b"["):
    body = _DECOMPRESSOR.decompress(body)
  return json.loads(body.decode("utf-8"))


def parse_xyzt_dict(payload: dict[str, Any], default_t) -> tuple[np.ndarray, np.ndarray] | None:
  if not isinstance(payload, dict):
    return None

  try:
    target_t = payload.get("t", default_t)
    t = np.asarray(target_t, dtype=np.float32)
    values = np.column_stack([
      np.asarray(payload["x"], dtype=np.float32),
      np.asarray(payload["y"], dtype=np.float32),
      np.asarray(payload["z"], dtype=np.float32),
    ])
  except (KeyError, TypeError, ValueError):
    return None

  if values.shape != (len(t), 3) or len(t) < 2:
    return None
  if not np.all(np.diff(t) > 0):
    return None

  return t, values
