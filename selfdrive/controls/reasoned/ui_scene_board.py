from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np

try:
  from openpilot.selfdrive.controls.reasoned.pathsynth import BasePlan
  from openpilot.selfdrive.controls.reasoned.scene_board import SceneBoard
except ModuleNotFoundError:
  from selfdrive.controls.reasoned.pathsynth import BasePlan
  from selfdrive.controls.reasoned.scene_board import SceneBoard


@dataclass(frozen=True)
class OverlayGeometry:
  lane_width_m: float = 4.5
  camera_height_m: float = 1.22
  horizon_ratio: float = 0.44
  focal_ratio: float = 1.35
  max_draw_distance_m: float = 90.0
  planned_corridor_half_width_m: float = 1.125
  focus_corridor_extra_width_m: float = 0.50
  dim_outside_corridor: bool = True
  outside_corridor_dim_alpha: int = 145
  candidate_lateral_offset_m: float = 0.0
  draw_candidate_labels: bool = False
  draw_base_path_reference: bool = False
  draw_corridor_side_guides: bool = True
  draw_corridor_side_fill: bool = True
  draw_corridor_side_labels: bool = False
  draw_edge_insets: bool = False
  draw_candidate_obstruction_boards: bool = False
  candidate_obstruction_offset_m: float = 1.25


class UiSceneBoardRenderer:
  """Render a VLM input that visually matches the onroad UI model overlay.

  This is intentionally PIL-based so it can run in the local PC POC, sim harness,
  and tests without starting the raylib UI process.
  """

  def __init__(self, width: int = 512, height: int = 384, geometry: OverlayGeometry | None = None):
    self.width = width
    self.height = height
    self.geometry = geometry or OverlayGeometry()

  def render(self, base_plan: BasePlan, vehicle_state: dict[str, Any] | None = None) -> SceneBoard:
    try:
      from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
      raise RuntimeError("UiSceneBoardRenderer requires Pillow") from exc

    state = vehicle_state or {}
    image = self._render_overlay_image(Image, ImageDraw, ImageFont, base_plan, state, None)
    path_lateral_offset_m = float(state.get("path_lateral_offset_m", 0.0))
    aux_pngs = self._candidate_obstruction_aux_pngs(Image, ImageDraw, ImageFont, base_plan, state, path_lateral_offset_m)

    lead_text = _lead_state_text(state)
    state_text = (
      f"frame={base_plan.frame_id} "
      f"v_ego={base_plan.current_speed:.1f}mps "
      f"desired speed {base_plan.desired_speed:.1f} m/s "
      f"curv={base_plan.desired_curvature:.5f} "
      f"tracked_path_lat={path_lateral_offset_m:.2f}m "
      f"base_path_lat=0.00m "
      f"blinkers={int(bool(state.get('left_blinker', 0)))}/{int(bool(state.get('right_blinker', 0)))} "
      f"{lead_text}"
    )
    return SceneBoard(self.width, self.height, bytearray(image.convert("RGB").tobytes()), state_text, aux_pngs=aux_pngs)

  def _render_overlay_image(self, Image, ImageDraw, ImageFont, base_plan: BasePlan, state: dict[str, Any], path_lateral_offset_m: float | None):
    frame = state.get("road_frame")
    image = self._image_from_frame(frame, Image)
    path_lateral_offset_m = float(state.get("path_lateral_offset_m", 0.0) if path_lateral_offset_m is None else path_lateral_offset_m)
    if self.geometry.dim_outside_corridor:
      image = self._apply_corridor_focus_mask(image, Image, ImageDraw, base_plan, path_lateral_offset_m)

    draw = ImageDraw.Draw(image, "RGBA")
    self._draw_model_overlay(draw, base_plan, path_lateral_offset_m)
    if self.geometry.draw_edge_insets:
      self._draw_edge_insets(image, ImageDraw, ImageFont, base_plan, path_lateral_offset_m)
    if self.geometry.draw_candidate_labels and self.geometry.candidate_lateral_offset_m > 1e-3:
      self._draw_candidate_labels(draw, ImageFont, path_lateral_offset_m)
    if self.geometry.draw_corridor_side_labels:
      self._draw_corridor_side_labels(draw, ImageFont, path_lateral_offset_m)
    self._draw_metric_ticks(draw)
    self._draw_hud(draw, ImageFont, base_plan, state)
    return image

  def _candidate_obstruction_aux_pngs(self, Image, ImageDraw, ImageFont, base_plan: BasePlan, state: dict[str, Any], path_lateral_offset_m: float) -> dict[str, bytes]:
    if not self.geometry.draw_candidate_obstruction_boards:
      return {}
    offset = max(0.0, float(self.geometry.candidate_obstruction_offset_m))
    if offset <= 1e-3:
      return {}

    aux = {}
    pair_image = self._render_candidate_pair_image(Image, ImageDraw, ImageFont, base_plan, state, path_lateral_offset_m, offset)
    pair_out = BytesIO()
    pair_image.convert("RGB").save(pair_out, format="PNG")
    aux["candidate_pair"] = pair_out.getvalue()

    for name, label, candidate_offset in (
      ("candidate_left", "CANDIDATE LEFT GREEN PATH", path_lateral_offset_m + offset),
      ("candidate_right", "CANDIDATE RIGHT GREEN PATH", path_lateral_offset_m - offset),
    ):
      candidate_state = dict(state)
      candidate_state["status"] = label
      image = self._render_overlay_image(Image, ImageDraw, ImageFont, base_plan, candidate_state, candidate_offset)
      draw = ImageDraw.Draw(image, "RGBA")
      font = ImageFont.load_default()
      draw.rounded_rectangle((8, 66, 220, 87), radius=4, fill=(0, 0, 0, 150))
      draw.text((14, 72), label, font=font, fill=(245, 245, 245, 255))
      out = BytesIO()
      image.convert("RGB").save(out, format="PNG")
      aux[name] = out.getvalue()
    return aux

  def _render_candidate_pair_image(self, Image, ImageDraw, ImageFont, base_plan: BasePlan, state: dict[str, Any], path_lateral_offset_m: float, offset_m: float):
    image = self._image_from_frame(state.get("road_frame"), Image)
    draw = ImageDraw.Draw(image, "RGBA")
    base_points = self._plan_points(base_plan)
    lane_width = self.geometry.lane_width_m

    for lane_offset, alpha in zip((-1.5 * lane_width, -0.5 * lane_width, 0.5 * lane_width, 1.5 * lane_width), (70, 115, 160, 70), strict=True):
      self._draw_strip(draw, base_points, lateral_offset=lane_offset, half_width=0.035, color=(255, 255, 255, alpha))
    for lane_offset in (-2.5 * lane_width, 2.5 * lane_width):
      self._draw_strip(draw, base_points, lateral_offset=lane_offset, half_width=0.055, color=(245, 60, 50, 120))

    left_offset = path_lateral_offset_m + offset_m
    right_offset = path_lateral_offset_m - offset_m
    half_width = self.geometry.planned_corridor_half_width_m
    self._draw_strip(draw, base_points, lateral_offset=left_offset, half_width=half_width, color=(40, 230, 255, 110))
    self._draw_strip(draw, base_points, lateral_offset=right_offset, half_width=half_width, color=(255, 80, 180, 105))
    self._draw_polyline(draw, base_points, lateral_offset=left_offset, color=(40, 230, 255, 245), width=3)
    self._draw_polyline(draw, base_points, lateral_offset=right_offset, color=(255, 80, 180, 245), width=3)
    self._draw_dashed_polyline(draw, base_points, lateral_offset=0.0, color=(245, 245, 245, 210), width=2, dash_px=10.0, gap_px=7.0)

    font = ImageFont.load_default()
    draw.rounded_rectangle((8, 66, 258, 87), radius=4, fill=(0, 0, 0, 155))
    draw.text((14, 72), "CYAN = LEFT-SHIFT CANDIDATE", font=font, fill=(210, 255, 255, 255))
    draw.rounded_rectangle((8, 91, 268, 112), radius=4, fill=(0, 0, 0, 155))
    draw.text((14, 97), "MAGENTA = RIGHT-SHIFT CANDIDATE", font=font, fill=(255, 215, 245, 255))
    pair_state = dict(state)
    pair_state["status"] = "SAME-FRAME CANDIDATE COMPARISON"
    self._draw_metric_ticks(draw)
    self._draw_hud(draw, ImageFont, base_plan, pair_state)
    return image

  def _image_from_frame(self, frame: Any, Image):
    if frame is None:
      image = Image.new("RGB", (self.width, self.height), (16, 20, 22))
    elif isinstance(frame, Image.Image):
      image = frame.convert("RGB")
    else:
      arr = np.asarray(frame)
      if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"road_frame must be HxWx3, got shape {arr.shape}")
      image = Image.fromarray(arr[:, :, :3].astype(np.uint8), "RGB")

    # Match UI behavior: preserve road aspect and center-crop rather than letterboxing.
    src_w, src_h = image.size
    scale = max(self.width / src_w, self.height / src_h)
    resized = image.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.BILINEAR)
    left = max(0, (resized.width - self.width) // 2)
    top = max(0, (resized.height - self.height) // 2)
    return resized.crop((left, top, left + self.width, top + self.height))

  def _draw_model_overlay(self, draw, base_plan: BasePlan, path_lateral_offset_m: float = 0.0) -> None:
    lane_width = self.geometry.lane_width_m
    base_points = self._plan_points(base_plan)
    lane_offsets = (-1.5 * lane_width, -0.5 * lane_width, 0.5 * lane_width, 1.5 * lane_width)
    lane_alpha = (75, 120, 165, 75)

    for offset, alpha in zip(lane_offsets, lane_alpha, strict=True):
      self._draw_strip(draw, base_points, lateral_offset=offset, half_width=0.035, color=(255, 255, 255, alpha))

    for offset in (-2.5 * lane_width, 2.5 * lane_width):
      self._draw_strip(draw, base_points, lateral_offset=offset, half_width=0.055, color=(245, 60, 50, 135))

    candidate_offset = max(0.0, float(self.geometry.candidate_lateral_offset_m))
    if candidate_offset > 1e-3:
      self._draw_polyline(
        draw,
        base_points,
        lateral_offset=path_lateral_offset_m + candidate_offset,
        color=(40, 230, 255, 220),
        width=2,
      )
      self._draw_polyline(
        draw,
        base_points,
        lateral_offset=path_lateral_offset_m - candidate_offset,
        color=(255, 80, 180, 220),
        width=2,
      )

    if self.geometry.draw_base_path_reference and abs(path_lateral_offset_m) > 0.08:
      self._draw_dashed_polyline(
        draw,
        base_points,
        lateral_offset=0.0,
        color=(205, 60, 220, 210),
        width=2,
        dash_px=10.0,
        gap_px=7.0,
      )

    self._draw_strip(
      draw,
      base_points,
      lateral_offset=path_lateral_offset_m,
      half_width=self.geometry.planned_corridor_half_width_m,
      color=(35, 210, 105, 85),
    )
    if self.geometry.draw_corridor_side_fill:
      half_width = self.geometry.planned_corridor_half_width_m
      self._draw_strip(
        draw,
        base_points,
        lateral_offset=path_lateral_offset_m + half_width * 0.5,
        half_width=half_width * 0.5,
        color=(35, 155, 255, 82),
      )
      self._draw_strip(
        draw,
        base_points,
        lateral_offset=path_lateral_offset_m - half_width * 0.5,
        half_width=half_width * 0.5,
        color=(180, 95, 255, 82),
      )
    self._draw_strip(
      draw,
      base_points,
      lateral_offset=path_lateral_offset_m,
      half_width=0.24,
      color=(35, 230, 75, 245),
    )
    if self.geometry.draw_corridor_side_guides:
      half_width = self.geometry.planned_corridor_half_width_m
      self._draw_polyline(
        draw,
        base_points,
        lateral_offset=path_lateral_offset_m + half_width,
        color=(40, 155, 255, 230),
        width=3,
      )
      self._draw_polyline(
        draw,
        base_points,
        lateral_offset=path_lateral_offset_m - half_width,
        color=(170, 105, 255, 230),
        width=3,
      )
    self._draw_polyline(draw, base_points, lateral_offset=path_lateral_offset_m, color=(255, 255, 255, 225), width=2)

  def _draw_metric_ticks(self, draw) -> None:
    for s_m in (10.0, 20.0, 40.0, 60.0):
      left = self._project(s_m, 3.5)
      right = self._project(s_m, -3.5)
      if left is not None and right is not None:
        draw.line([left, right], fill=(255, 255, 255, 46), width=1)

  def _apply_corridor_focus_mask(self, image, Image, ImageDraw, base_plan: BasePlan, path_lateral_offset_m: float):
    base_points = self._plan_points(base_plan)
    focus_half = self.geometry.planned_corridor_half_width_m + max(0.0, self.geometry.focus_corridor_extra_width_m)
    alpha = int(np.clip(self.geometry.outside_corridor_dim_alpha, 0, 220))
    if alpha <= 0:
      return image
    mask = Image.new("L", (self.width, self.height), 255)
    mask_draw = ImageDraw.Draw(mask)
    self._draw_strip(
      mask_draw,
      base_points,
      lateral_offset=path_lateral_offset_m,
      half_width=focus_half,
      color=0,
    )
    dimmed = Image.blend(image, Image.new("RGB", image.size, (0, 0, 0)), alpha / 255.0)
    return Image.composite(dimmed, image, mask)

  def _draw_corridor_side_labels(self, draw, ImageFont, path_lateral_offset_m: float) -> None:
    font = ImageFont.load_default()
    for text, x, y, box_w, color in self._corridor_side_label_specs(path_lateral_offset_m):
      draw.rounded_rectangle((x - 3, y - 2, x + box_w, y + 12), radius=3, fill=(0, 0, 0, 130))
      draw.text((x, y), text, font=font, fill=color)

  def _corridor_side_label_specs(self, path_lateral_offset_m: float) -> tuple[tuple[str, float, float, int, tuple[int, int, int, int]], ...]:
    half_width = self.geometry.planned_corridor_half_width_m
    s_m = 20.0
    left_edge = self._project(s_m, path_lateral_offset_m + half_width)
    right_edge = self._project(s_m, path_lateral_offset_m - half_width)
    if left_edge is None:
      left_edge = (self.width * 0.38, self.height * 0.58)
    if right_edge is None:
      right_edge = (self.width * 0.62, self.height * 0.58)

    left_text = "BLUE LEFT"
    right_text = "PURPLE RIGHT"
    left_w = 64
    right_w = 82
    y = float(np.clip(max(left_edge[1], right_edge[1]) - 24.0, 38.0, self.height - 18.0))
    left_x = float(np.clip(left_edge[0] - left_w - 6.0, 4.0, self.width - left_w - 6.0))
    right_x = float(np.clip(right_edge[0] + 6.0, 4.0, self.width - right_w - 6.0))
    if left_x + left_w + 5.0 > right_x:
      left_x = float(np.clip(left_edge[0] - left_w - 18.0, 4.0, self.width - left_w - 6.0))
      right_x = float(np.clip(right_edge[0] + 18.0, 4.0, self.width - right_w - 6.0))
    return (
      (left_text, left_x, y, left_w, (40, 155, 255, 245)),
      (right_text, right_x, y, right_w, (170, 105, 255, 245)),
    )

  def _draw_candidate_labels(self, draw, ImageFont, path_lateral_offset_m: float) -> None:
    font = ImageFont.load_default()
    specs = (
      ("PATH A", 156.0, 58.0, 54, (40, 230, 255, 245)),
      ("PATH B", 230.0, 58.0, 54, (255, 80, 180, 245)),
    )
    for text, x, y, box_w, color in specs:
      x = float(np.clip(x, 4.0, self.width - box_w - 6.0))
      y = float(np.clip(y, 38.0, self.height - 18.0))
      draw.rounded_rectangle((x - 3, y - 2, x + box_w, y + 12), radius=3, fill=(0, 0, 0, 125))
      draw.text((x, y), text, font=font, fill=color)

  def _draw_edge_insets(self, image, ImageDraw, ImageFont, base_plan: BasePlan, path_lateral_offset_m: float) -> None:
    source = image.copy()
    panel_w = max(96, int(round(self.width * 0.381)))
    panel_h = max(52, int(round(self.height * 0.33)))
    y = max(0, self.height - panel_h - 4)
    left_panel = self._edge_inset_panel(
      source,
      ImageDraw,
      ImageFont,
      base_plan,
      path_lateral_offset_m + self.geometry.planned_corridor_half_width_m,
      panel_w,
      panel_h,
      "BLUE",
      (40, 155, 255, 255),
    )
    right_panel = self._edge_inset_panel(
      source,
      ImageDraw,
      ImageFont,
      base_plan,
      path_lateral_offset_m - self.geometry.planned_corridor_half_width_m,
      panel_w,
      panel_h,
      "PURPLE",
      (170, 105, 255, 255),
    )
    image.paste(left_panel, (4, y))
    image.paste(right_panel, (self.width - panel_w - 4, y))
    draw = ImageDraw.Draw(image, "RGBA")
    label = "edge-local Qwen evidence insets"
    font = ImageFont.load_default()
    label_w = min(self.width - 16, 174)
    label_x = max(8, int(round((self.width - label_w) * 0.5)))
    label_y = max(38, y - 18)
    draw.rounded_rectangle((label_x, label_y, label_x + label_w, label_y + 15), radius=3, fill=(0, 0, 0, 145))
    draw.text((label_x + 5, label_y + 3), label, font=font, fill=(245, 245, 245, 255))

  def _edge_inset_panel(
    self,
    image,
    ImageDraw,
    ImageFont,
    base_plan: BasePlan,
    edge_lateral_offset_m: float,
    panel_w: int,
    panel_h: int,
    label: str,
    color: tuple[int, int, int, int],
  ):
    points = []
    for s_m, y_m in self._plan_points(base_plan):
      p = self._project(s_m, y_m + edge_lateral_offset_m)
      if p is not None:
        points.append(p)
    if points:
      weights = np.asarray([max(1.0, p[1]) ** 1.5 for p in points], dtype=np.float64)
      xs = np.asarray([p[0] for p in points], dtype=np.float64)
      cx = float(np.average(xs, weights=weights))
    else:
      cx = self.width * (0.35 if label == "BLUE" else 0.65)

    crop_w = int(round(self.width * 0.48))
    crop_h = int(round(self.height * 0.62))
    y1 = self.height
    y0 = max(0, y1 - crop_h)
    x0 = int(round(cx - crop_w * 0.5))
    x0 = max(0, min(self.width - crop_w, x0))
    panel = image.crop((x0, y0, x0 + crop_w, y1)).resize((panel_w, panel_h))
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((1, 1, panel_w - 2, panel_h - 2), outline=color, width=3)
    font = ImageFont.load_default()
    box_w = 43 if label == "BLUE" else 58
    draw.rounded_rectangle((5, 4, min(panel_w - 5, box_w), 18), radius=3, fill=(0, 0, 0, 150))
    draw.text((8, 7), label, font=font, fill=color)
    return panel

  def _draw_hud(self, draw, ImageFont, base_plan: BasePlan, state: dict[str, Any]) -> None:
    font = ImageFont.load_default()
    speed_mph = base_plan.current_speed * 2.23694
    status = str(state.get("status", "SIM"))
    text = f"{speed_mph:4.0f} mph  {status}"
    draw.rounded_rectangle((8, 8, 146, 34), radius=6, fill=(0, 0, 0, 120))
    draw.text((16, 17), text, font=font, fill=(245, 245, 245, 255))
    if _as_optional_float(state.get("lead_distance_m")) is not None:
      lead_d = _as_optional_float(state.get("lead_distance_m"))
      lead_v = _as_optional_float(state.get("lead_speed_mps"))
      lead_rel = _as_optional_float(state.get("lead_rel_speed_mps"))
      lead_lat = _as_optional_float(state.get("lead_lateral_m"))
      lead_line = (
        f"lead {lead_d:.1f}m lat={lead_lat:.1f} v={lead_v:.1f} rel={lead_rel:.1f}"
        if lead_d is not None and lead_lat is not None and lead_v is not None and lead_rel is not None else "lead present"
      )
      draw.rounded_rectangle((8, 38, 286, 63), radius=6, fill=(0, 0, 0, 120))
      draw.text((16, 47), lead_line, font=font, fill=(245, 245, 245, 255))

  def _plan_points(self, base_plan: BasePlan) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in zip(base_plan.x, base_plan.y) if x >= 0.0 and x <= self.geometry.max_draw_distance_m]
    if len(pts) >= 2:
      return pts
    return [(float(s), 0.0) for s in np.linspace(0.5, self.geometry.max_draw_distance_m, 24)]

  def _draw_strip(self, draw, points: list[tuple[float, float]], lateral_offset: float, half_width: float, color: tuple[int, int, int, int]) -> None:
    left = []
    right = []
    for s_m, y_m in points:
      lp = self._project(s_m, y_m + lateral_offset + half_width)
      rp = self._project(s_m, y_m + lateral_offset - half_width)
      if lp is not None and rp is not None:
        left.append(lp)
        right.append(rp)
    if len(left) > 1 and len(right) > 1:
      draw.polygon(left + right[::-1], fill=color)

  def _draw_polyline(self, draw, points: list[tuple[float, float]], lateral_offset: float, color: tuple[int, int, int, int], width: int) -> None:
    projected = []
    for s_m, y_m in points:
      p = self._project(s_m, y_m + lateral_offset)
      if p is not None:
        projected.append(p)
    if len(projected) > 1:
      draw.line(projected, fill=color, width=width, joint="curve")

  def _draw_dashed_polyline(
    self,
    draw,
    points: list[tuple[float, float]],
    lateral_offset: float,
    color: tuple[int, int, int, int],
    width: int,
    dash_px: float,
    gap_px: float,
  ) -> None:
    projected = []
    for s_m, y_m in points:
      p = self._project(s_m, y_m + lateral_offset)
      if p is not None:
        projected.append(p)
    if len(projected) < 2:
      return

    draw_dash = True
    remaining = max(1.0, dash_px)
    for start, end in zip(projected, projected[1:], strict=False):
      x0, y0 = start
      x1, y1 = end
      seg_len = float(np.hypot(x1 - x0, y1 - y0))
      if seg_len <= 1e-3:
        continue
      consumed = 0.0
      while consumed < seg_len:
        step = min(remaining, seg_len - consumed)
        t0 = consumed / seg_len
        t1 = (consumed + step) / seg_len
        if draw_dash:
          p0 = (x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0)
          p1 = (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)
          draw.line([p0, p1], fill=color, width=width)
        consumed += step
        remaining -= step
        if remaining <= 1e-3:
          draw_dash = not draw_dash
          remaining = max(1.0, dash_px if draw_dash else gap_px)

  def _project(self, s_m: float, y_left_m: float) -> tuple[float, float] | None:
    if s_m <= 0.5:
      return None
    f = self.width * self.geometry.focal_ratio
    cx = self.width * 0.5
    horizon_y = self.height * self.geometry.horizon_ratio
    u = cx - f * y_left_m / s_m
    v = horizon_y + f * self.geometry.camera_height_m / s_m
    if u < -self.width or u > self.width * 2 or v < -self.height or v > self.height * 2:
      return None
    return (float(u), float(v))


def _as_optional_float(value: Any) -> float | None:
  if value is None:
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if np.isfinite(result) else None


def _metric_value_text(state: dict[str, Any], key: str, precision: int = 1) -> str:
  value = _as_optional_float(state.get(key))
  if value is None:
    return "none"
  return f"{value:.{precision}f}"


def _lead_state_text(state: dict[str, Any]) -> str:
  present = int(bool(state.get("lead_present", 0)))
  present_word = "yes" if present else "no"
  source = str(state.get("lead_source", "none")).replace(" ", "_")
  return " ".join((
    f"lead present {present_word};",
    f"source {source};",
    f"distance {_metric_value_text(state, 'lead_distance_m')} m;",
    f"lateral offset {_metric_value_text(state, 'lead_lateral_m')} m;",
    f"lead speed {_metric_value_text(state, 'lead_speed_mps')} m/s;",
    f"relative speed {_metric_value_text(state, 'lead_rel_speed_mps')} m/s;",
    f"closing {_metric_value_text(state, 'lead_closing_mps')} m/s;",
    f"acceleration {_metric_value_text(state, 'lead_accel_mps2')} m/s2;",
    f"lateral velocity {_metric_value_text(state, 'lead_lateral_velocity_mps')} m/s",
  ))
