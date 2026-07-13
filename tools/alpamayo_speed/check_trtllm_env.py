#!/usr/bin/env python3
from __future__ import annotations

import json

import modelopt
import tensorrt
import tensorrt_llm
import torch


def main() -> int:
  print(
    json.dumps(
      {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count(),
        "tensorrt": tensorrt.__version__,
        "tensorrt_has_fp4": hasattr(tensorrt.DataType, "FP4") or hasattr(tensorrt, "fp4"),
        "tensorrt_llm": tensorrt_llm.__version__,
        "modelopt": getattr(modelopt, "__version__", "unknown"),
      },
      indent=2,
      sort_keys=True,
    ),
    flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
