import json
from types import SimpleNamespace

from openpilot.selfdrive.alpamayo.trace import AlpamayoTraceLogger


def test_trace_logger_writes_jsonl(tmp_path):
  trace_path = tmp_path / "alpamayo_trace.jsonl"
  logger = AlpamayoTraceLogger(trace_path)

  logger.log(
    "modeld_semantic_plan_decision",
    semanticSource=SimpleNamespace(raw=2),
    applied=True,
    rejectionReason="",
    safetyViolation="",
  )

  rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
  assert len(rows) == 1
  assert rows[0]["event"] == "modeld_semantic_plan_decision"
  assert rows[0]["semanticSource"] == 2
  assert rows[0]["applied"] is True
  assert rows[0]["safetyViolation"] == ""
  assert "monoTimeNs" in rows[0]
