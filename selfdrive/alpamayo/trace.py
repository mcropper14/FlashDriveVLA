import json
import os
import time
from pathlib import Path
from typing import Any


TRACE_ENV = "ALPAMAYO_TRACE_PATH"


def default_trace_path() -> Path | None:
  env_path = os.getenv(TRACE_ENV)
  if env_path:
    return Path(env_path)

  prefix = os.getenv("OPENPILOT_PREFIX")
  if prefix:
    return Path(prefix) / "alpamayo_trace.jsonl"

  data_dir = Path("/data/openpilot")
  if data_dir.exists():
    return data_dir / "alpamayo_trace.jsonl"

  return None


def _jsonable(value: Any) -> Any:
  raw = getattr(value, "raw", None)
  if raw is not None:
    return _jsonable(raw)
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  try:
    return int(value)
  except (TypeError, ValueError):
    pass
  try:
    return float(value)
  except (TypeError, ValueError):
    pass
  return str(value)


class AlpamayoTraceLogger:
  def __init__(self, path: Path | str | None = None):
    self.path = Path(path) if path is not None else default_trace_path()
    self.disabled = self.path is None

  def log(self, event: str, **fields: Any) -> None:
    if self.disabled or self.path is None:
      return

    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      payload = {
        "event": event,
        "monoTimeNs": time.monotonic_ns(),
      }
      payload.update({key: _jsonable(value) for key, value in fields.items()})
      with self.path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError:
      self.disabled = True
