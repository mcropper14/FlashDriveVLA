from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import os
from pathlib import Path
from typing import Optional

try:
  from openpilot.selfdrive.controls.reasoned.pathsynth import BasePlan
except ModuleNotFoundError:
  from selfdrive.controls.reasoned.pathsynth import BasePlan


Color = tuple[int, int, int]


@dataclass
class SceneBoard:
  width: int
  height: int
  pixels: bytearray
  state_text: str
  aux_pngs: dict[str, bytes] = field(default_factory=dict)

  def set_px(self, x: int, y: int, color: Color) -> None:
    if 0 <= x < self.width and 0 <= y < self.height:
      idx = (y * self.width + x) * 3
      self.pixels[idx:idx + 3] = bytes(color)

  def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
      self.set_px(x0, y0, color)
      if x0 == x1 and y0 == y1:
        break
      e2 = 2 * err
      if e2 >= dy:
        err += dy
        x0 += sx
      if e2 <= dx:
        err += dx
        y0 += sy

  def draw_polyline(self, points: list[tuple[int, int]], color: Color) -> None:
    for a, b in zip(points, points[1:]):
      self.draw_line(a[0], a[1], b[0], b[1], color)

  def to_ppm_bytes(self) -> bytes:
    return f"P6\n{self.width} {self.height}\n255\n".encode("ascii") + bytes(self.pixels)

  def to_png_bytes(self) -> Optional[bytes]:
    try:
      from PIL import Image
    except Exception:
      return None
    image = Image.frombytes("RGB", (self.width, self.height), bytes(self.pixels))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()

  def to_jpeg_bytes(self, quality: int = 85) -> Optional[bytes]:
    try:
      from PIL import Image
    except Exception:
      return None
    image = Image.frombytes("RGB", (self.width, self.height), bytes(self.pixels))
    output = BytesIO()
    image.save(output, format="JPEG", quality=max(1, min(95, int(quality))), optimize=False)
    return output.getvalue()

  def save(self, path: Path) -> None:
    png = self.to_png_bytes()
    if png is not None:
      path.write_bytes(png)
    else:
      path.with_suffix(".ppm").write_bytes(self.to_ppm_bytes())


class SceneBoardRenderer:
  def __init__(self, width: int = 512, height: int = 384, max_s_m: float = 70.0):
    self.width = width
    self.height = height
    self.max_s_m = max_s_m

  def render(self, base_plan: BasePlan, vehicle_state: dict[str, float] | None = None) -> SceneBoard:
    pixels = bytearray([12, 16, 18] * self.width * self.height)
    state = vehicle_state or {}
    state_text = (
      f"frame={base_plan.frame_id} "
      f"v_ego={base_plan.current_speed:.1f}mps "
      f"desired speed {base_plan.desired_speed:.1f} m/s "
      f"curv={base_plan.desired_curvature:.5f} "
      f"blinkers={int(state.get('left_blinker', 0))}/{int(state.get('right_blinker', 0))}"
    )
    board = SceneBoard(self.width, self.height, pixels, state_text)
    self._draw_grid(board)
    self._draw_static_corridor(board)
    self._draw_paths(board, base_plan)
    self._draw_text_if_available(board, state_text)
    return board

  def _draw_grid(self, board: SceneBoard) -> None:
    for s_m in (10.0, 20.0, 40.0, 60.0):
      y = self._project(0.0, s_m)[1]
      board.draw_line(0, y, board.width - 1, y, (46, 55, 58))
    board.draw_line(board.width // 2, 0, board.width // 2, board.height - 1, (40, 48, 52))

  def _draw_static_corridor(self, board: SceneBoard) -> None:
    for lat_m, color in ((-1.8, (92, 110, 115)), (1.8, (92, 110, 115)), (-3.5, (58, 76, 80)), (3.5, (58, 76, 80))):
      points = [self._project(lat_m, s_m) for s_m in range(0, int(self.max_s_m), 2)]
      board.draw_polyline(points, color)

  def _draw_paths(self, board: SceneBoard, base_plan: BasePlan) -> None:
    points = [self._project(y, x) for x, y in zip(base_plan.x, base_plan.y)]
    if len(points) > 1:
      board.draw_polyline(points, (42, 180, 235))
    for offset, color in ((-0.35, (245, 195, 66)), (0.35, (149, 229, 99))):
      candidate = [self._project(y + offset, x) for x, y in zip(base_plan.x, base_plan.y)]
      if len(candidate) > 1:
        board.draw_polyline(candidate, color)

  def _draw_text_if_available(self, board: SceneBoard, text: str) -> None:
    if os.getenv("RTP_RENDER_TEXT") != "1":
      return
    try:
      from PIL import Image, ImageDraw
    except Exception:
      return
    image = Image.frombytes("RGB", (board.width, board.height), bytes(board.pixels))
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), text, fill=(230, 235, 236))
    board.pixels[:] = image.tobytes()

  def _project(self, lat_m: float, s_m: float) -> tuple[int, int]:
    px_per_m_lat = self.width / 11.0
    x = int(self.width / 2 + lat_m * px_per_m_lat)
    y = int(self.height - 28 - (max(0.0, min(self.max_s_m, s_m)) / self.max_s_m) * (self.height - 64))
    return x, y
