#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


OLD = """comm = pkl5.Intracomm(MPI.COMM_WORLD)


def set_mpi_comm(new_comm):
    global comm
    comm = new_comm
"""

NEW = """if os.environ.get("TLLM_DISABLE_MPI") == "1":
    comm = None
else:
    comm = pkl5.Intracomm(MPI.COMM_WORLD)


def set_mpi_comm(new_comm):
    global comm
    comm = new_comm
"""

OLD_LOCAL = """local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)


def local_mpi_comm():
    return local_comm
"""

NEW_LOCAL = """if os.environ.get("TLLM_DISABLE_MPI") == "1":
    local_comm = None
else:
    local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)


def local_mpi_comm():
    return local_comm
"""

OLD_WORLD = """def global_mpi_size():
    return MPI.COMM_WORLD.Get_size() if ENABLE_MULTI_DEVICE else 1


def mpi_world_size():
    return mpi_comm().Get_size() if ENABLE_MULTI_DEVICE else 1
"""

NEW_WORLD = """def global_mpi_size():
    if mpi_disabled():
        return 1
    return MPI.COMM_WORLD.Get_size() if ENABLE_MULTI_DEVICE else 1


def mpi_world_size():
    if mpi_disabled():
        return 1
    return mpi_comm().Get_size() if ENABLE_MULTI_DEVICE else 1
"""

OLD_LOCAL_SIZE = """def local_mpi_size():
    return local_comm.Get_size() if ENABLE_MULTI_DEVICE else 1
"""

NEW_LOCAL_SIZE = """def local_mpi_size():
    if mpi_disabled():
        return 1
    return local_comm.Get_size() if ENABLE_MULTI_DEVICE else 1
"""

OLD_BARRIERS = """def mpi_barrier():
    if ENABLE_MULTI_DEVICE:
        mpi_comm().Barrier()


def local_mpi_barrier():
    if ENABLE_MULTI_DEVICE:
        local_comm.Barrier()
"""

NEW_BARRIERS = """def mpi_barrier():
    if mpi_disabled():
        return
    if ENABLE_MULTI_DEVICE:
        mpi_comm().Barrier()


def local_mpi_barrier():
    if mpi_disabled():
        return
    if ENABLE_MULTI_DEVICE:
        local_comm.Barrier()
"""

OLD_ALLGATHER = """def mpi_allgather(obj):
    return mpi_comm().allgather(obj) if ENABLE_MULTI_DEVICE else obj
"""

NEW_ALLGATHER = """def mpi_allgather(obj):
    if mpi_disabled():
        return [obj]
    return mpi_comm().allgather(obj) if ENABLE_MULTI_DEVICE else obj
"""


def patch_file(path: Path) -> str:
  text = path.read_text()
  changed = False
  if NEW not in text and OLD in text:
    backup = path.with_suffix(path.suffix + ".alpamayo_mpi_patch.bak")
    if not backup.exists():
      backup.write_text(text)
    text = text.replace(OLD, NEW)
    changed = True
  if NEW_LOCAL not in text and OLD_LOCAL in text:
    backup = path.with_suffix(path.suffix + ".alpamayo_mpi_patch.bak")
    if not backup.exists():
      backup.write_text(text)
    text = text.replace(OLD_LOCAL, NEW_LOCAL)
    changed = True
  if NEW_WORLD not in text and OLD_WORLD in text:
    backup = path.with_suffix(path.suffix + ".alpamayo_mpi_patch.bak")
    if not backup.exists():
      backup.write_text(text)
    text = text.replace(OLD_WORLD, NEW_WORLD)
    changed = True
  if NEW_LOCAL_SIZE not in text and OLD_LOCAL_SIZE in text:
    backup = path.with_suffix(path.suffix + ".alpamayo_mpi_patch.bak")
    if not backup.exists():
      backup.write_text(text)
    text = text.replace(OLD_LOCAL_SIZE, NEW_LOCAL_SIZE)
    changed = True
  if NEW_BARRIERS not in text and OLD_BARRIERS in text:
    backup = path.with_suffix(path.suffix + ".alpamayo_mpi_patch.bak")
    if not backup.exists():
      backup.write_text(text)
    text = text.replace(OLD_BARRIERS, NEW_BARRIERS)
    changed = True
  if NEW_ALLGATHER not in text and OLD_ALLGATHER in text:
    backup = path.with_suffix(path.suffix + ".alpamayo_mpi_patch.bak")
    if not backup.exists():
      backup.write_text(text)
    text = text.replace(OLD_ALLGATHER, NEW_ALLGATHER)
    changed = True
  if not changed and NEW in text and NEW_LOCAL in text and NEW_WORLD in text and NEW_LOCAL_SIZE in text and NEW_BARRIERS in text and NEW_ALLGATHER in text:
    return "already_patched"
  if not changed:
    raise RuntimeError(f"expected TensorRT-LLM MPI import block not found in {path}")
  path.write_text(text)
  return "patched"


def main() -> int:
  parser = argparse.ArgumentParser(description="Patch isolated TensorRT-LLM venv so TLLM_DISABLE_MPI skips import-time MPI communicators.")
  parser.add_argument("--utils-path", required=True, type=Path, help="Path to tensorrt_llm/_utils.py inside the isolated venv.")
  args = parser.parse_args()
  status = patch_file(args.utils_path)
  print(status)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
