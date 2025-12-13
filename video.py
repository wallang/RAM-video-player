# RAM Video Player QC
# Single-file app (PySide6 + OpenCV)
#
# v0.016
# - Frame-step buttons redesigned (frame-step icon, not scrubbing/seek)
# - Stop button removed: active play button becomes Stop while playing (forward or reverse)
# - Timeline label area reserved above ticks (no overlap), Nuke-like readable frame label
# - Color swatches: clearer selection indicator (contrast outline + check mark), toggle-off to disable drawing
# - Stroke width slider under swatches (min matches previous default)
# - Pixel inspector: removed (0-255) integer values, fixed-width stable text
# - Changelog includes all earlier versions 0.001–0.013 plus 0.014+
#
# Requirements: PySide6, opencv-python, numpy (psutil optional)

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from PySide6.QtCore import Qt, QTimer, QPoint, QRect, QEvent, QSize
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QMouseEvent,
    QKeyEvent,
    QFont,
    QFontMetrics,
    QIcon,
)
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QSlider,
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QProgressBar,
    QSizePolicy,
    QFrame,
    QToolButton,
    QDialog,
    QTextEdit,
    QDialogButtonBox,
)

__version__ = "0.016"


# -----------------------------
# Changelog (0.001–0.016)
# -----------------------------

CHANGELOG: Dict[str, Dict[str, List[str]]] = {
    "0.001": {
        "Added": [
            "Alap RAM alapú videólejátszó, amely az egész videót memóriába tölti.",
            "Lejátszás / szünet funkció, frame léptetés előre és hátra.",
            "Loop mód, amellyel a videó végtelenítve lejátszható.",
            "Betöltési progress bar és fájlméret információ.",
            "Szabad RAM kijelzése.",
        ],
        "Changed": [],
        "Fixed": [],
        "Removed": [],
    },
    "0.002": {
        "Added": [
            "Verziószám kijelzése az ablak címében és az alsó sorban.",
            "Changelog ablak, amelyben verziónként, kategóriákra bontva láthatók a változások.",
        ],
        "Changed": [],
        "Fixed": [],
        "Removed": [],
    },
    "0.003": {"Added": [], "Changed": [], "Fixed": [
        "Nagy felbontású vagy hosszú videók betöltésekor jelentkező memóriahiba (ArrayMemoryError) már nem dönti be a programot.",
        "Ha memória/olvasási hiba történik, a program jelzi a hibát, és a már betöltött rész továbbra is lejátszható.",
    ], "Removed": []},
    "0.004": {"Added": [
        "A betöltés közben folyamatosan frissül a 'Betöltve' érték, így látható, hány byte-nyi adat került már memóriába.",
    ], "Changed": [
        "Részleges betöltésnél már nem írjuk ki a teljes fájlméretet betöltöttnek, hanem csak a ténylegesen beolvasott mennyiséget becsüljük.",
    ], "Fixed": [], "Removed": []},
    "0.005": {"Added": [
        "Csatornanézetek billentyűkkel: R = Red, G = Green, B = Blue, A = Alpha (ha elérhető).",
        "A '0' billentyűvel vissza lehet térni a normál RGB nézetre.",
    ], "Changed": [], "Fixed": [], "Removed": []},
    "0.006": {"Added": [], "Changed": [
        "A csatorna nézetek (R/G/B/A) toggle módban működnek: ugyanazt a billentyűt újra megnyomva visszavált RGB nézetre.",
    ], "Fixed": [
        "Csatornanézetek használatakor fellépő 'memoryview: underlying buffer is not C-contiguous' hiba javítva.",
    ], "Removed": []},
    "0.007": {"Added": [
        "Pixel érték kijelzése az egér alatt: RGBA értékek, aspect-ratio tudatos mappinggal.",
    ], "Changed": [], "Fixed": [], "Removed": []},
    "0.008": {"Added": [
        "Light / Dark téma választható legördülő listából.",
        "Jobb oldali fix Pixel Info doboz sötét háttérrel, OS témától függetlenül.",
        "Pixel értékek Nuke-szerű színkódolással, 0.00000 formátumban + mellette 0–255 érték zárójelben.",
    ], "Changed": [], "Fixed": [], "Removed": []},
    "0.009": {"Added": [], "Changed": [
        "Stabilitási és kisebb UX finomítások a betöltés és RAM kijelzés körül.",
    ], "Fixed": [], "Removed": []},
    "0.010": {"Added": [
        "PNG mentés az aktuális képkockáról. A fájlnév: <forrásnév>_QC<frame>.png.",
        "Új 'PNG mentése' gomb a vezérlősoron.",
    ], "Changed": [], "Fixed": [], "Removed": []},
    "0.011": {"Added": [], "Changed": [
        "A PNG mentés az aktuális csatornanézetet exportálja: R/G/B/A módban az adott csatorna szürke PNG-ként kerül mentésre.",
    ], "Fixed": [], "Removed": []},
    "0.012": {"Added": [
        "Csatorna státusz label a vezérlősorban, színkóddal (RGB/R/G/B/A).",
        "A lejátszó sávban extra kis vonalkák jelzik a frame-eket.",
    ], "Changed": [
        "A PNG fájlnév jelzi az aktuális csatornát is, pl. _QC0012R, _QC0012G stb.",
        "A program neve: 'RAM Video Player v<verzió>'.",
    ], "Fixed": [], "Removed": []},
    "0.013": {"Added": [
        "Az aktuális frame szám a timeline csúszka fölött, a jelző felett mozog.",
        "Pixel értékek lejátszás közben is frissülnek, ha az egér a képen áll.",
    ], "Changed": [
        "Pixel info doboz fix szélességet kapott, így semmi nem lóg ki belőle.",
    ], "Fixed": [], "Removed": []},
    "0.014": {"Added": [
        "QC videólejátszó alap (PySide6 + OpenCV): Nuke-szerű timeline, IN/OUT, rajz overlay, annotált export.",
    ], "Changed": [], "Fixed": [], "Removed": []},
    "0.015": {"Added": [
        "Changelog ablak (verziónként kategorizálva).",
        "Timeline jelölők az annotált frame-ekhez (kék tick).",
    ], "Changed": [
        "Transport gombok letisztult Qt ikonokra cserélve, frame-ugrás Enterre.",
        "Rajz mód: szín-swatch alapú (szín kiválasztás = rajz aktív, újrakattintás = inaktív).",
        "Billentyűkezelés: aktív ablaknál nyilak frame-et léptetnek, Space play/pause.",
        "Timeline frame szám olvashatóbb háttércímkével.",
    ], "Fixed": [], "Removed": [
        "Külön 'Ugrás' gomb a frame mező mellől.",
        "Külön rajz mód BE/KI gomb.",
    ]},
    "0.016": {"Added": [
        "Vonalvastagság csúszka (min: 3) a színek alatt.",
        "Szín-swatch kijelölés pipa jelöléssel (minden színen jól látszik).",
    ], "Changed": [
        "Frame-léptetés gombok 'frame step' ikont kaptak (nem seek/tekerés).",
        "Stop gomb megszűnt: az aktív lejátszás gombja Stop ikonná vált lejátszás közben (előre vagy hátra).",
        "Timeline: külön felirat sáv a frame számnak, nem takarja a tickeket.",
        "Pixel inspector: csak 0.00000 értékek (nincs 0–255 zárójeles rész), stabilabb szöveg.",
    ], "Fixed": [], "Removed": [
        "Külön Stop gomb a transport sorból.",
    ]},
}


def _version_key(v: str) -> List[int]:
    return [int(p) for p in v.split(".") if p.isdigit()]


def render_changelog_text() -> str:
    lines: List[str] = []
    for version in sorted(CHANGELOG.keys(), key=_version_key, reverse=True):
        lines.append(f"Verzió {version}")
        lines.append("-" * (8 + len(version)))
        sections = CHANGELOG[version]
        for label, title in [
            ("Added", "Új funkciók"),
            ("Changed", "Módosítások"),
            ("Fixed", "Javítások"),
            ("Removed", "Eltávolítva"),
        ]:
            items = sections.get(label, [])
            if not items:
                continue
            lines.append(f"\n{title}:")
            for item in items:
                lines.append(f"  • {item}")
        lines.append("\n")
    return "\n".join(lines).strip()


def format_bytes(num: int) -> str:
    val = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if val < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} PB"


# -----------------------------
# Icons (custom, clean)
# -----------------------------

def _icon_from_painter(draw_fn, size: int = 18) -> QIcon:
    def make_pix(color: QColor, alpha: int) -> QPixmap:
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        draw_fn(p, QColor(color.red(), color.green(), color.blue(), alpha), size)
        p.end()
        return pm

    icon = QIcon()
    icon.addPixmap(make_pix(QColor(230, 230, 230), 255), QIcon.Normal, QIcon.Off)
    icon.addPixmap(make_pix(QColor(140, 140, 140), 180), QIcon.Disabled, QIcon.Off)
    return icon


def _draw_triangle(p: QPainter, color: QColor, size: int, direction: int) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    pad = 4
    if direction >= 0:
        pts = [QPoint(pad, pad), QPoint(size - pad, size // 2), QPoint(pad, size - pad)]
    else:
        pts = [QPoint(size - pad, pad), QPoint(pad, size // 2), QPoint(size - pad, size - pad)]
    p.drawPolygon(pts)


def _draw_stop(p: QPainter, color: QColor, size: int) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    pad = 5
    p.drawRect(pad, pad, size - 2 * pad, size - 2 * pad)


def _draw_frame_step(p: QPainter, color: QColor, size: int, direction: int) -> None:
    # Triangle + vertical bar = single-frame step
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    pad = 4
    bar_w = 3
    if direction >= 0:
        p.drawRect(size - pad - bar_w, pad, bar_w, size - 2 * pad)
        pts = [QPoint(pad, pad), QPoint(size - pad - bar_w - 1, size // 2), QPoint(pad, size - pad)]
    else:
        p.drawRect(pad, pad, bar_w, size - 2 * pad)
        pts = [QPoint(size - pad, pad), QPoint(pad + bar_w + 1, size // 2), QPoint(size - pad, size - pad)]
    p.drawPolygon(pts)


def play_icon(direction: int) -> QIcon:
    return _icon_from_painter(lambda p, c, s: _draw_triangle(p, c, s, direction))


def stop_icon() -> QIcon:
    return _icon_from_painter(lambda p, c, s: _draw_stop(p, c, s))


def frame_step_icon(direction: int) -> QIcon:
    return _icon_from_painter(lambda p, c, s: _draw_frame_step(p, c, s, direction))


# -----------------------------
# Annotations
# -----------------------------

@dataclass
class Stroke:
    color: QColor
    width: int
    points: List[Tuple[float, float]]  # frame pixel coords


class AnnotationStore:
    def __init__(self) -> None:
        self._data: Dict[int, List[Stroke]] = {}
        self._annotated: Set[int] = set()

    def clear_all(self) -> None:
        self._data.clear()
        self._annotated.clear()

    def strokes_for(self, frame_idx: int) -> List[Stroke]:
        return self._data.get(frame_idx, [])

    def annotated_sorted(self) -> List[int]:
        return sorted(self._annotated)

    def annotated_set(self) -> Set[int]:
        return set(self._annotated)

    def add_stroke(self, frame_idx: int, stroke: Stroke) -> None:
        self._data.setdefault(frame_idx, []).append(stroke)
        self._annotated.add(frame_idx)

    def undo_last(self, frame_idx: int) -> bool:
        lst = self._data.get(frame_idx)
        if not lst:
            return False
        lst.pop()
        if not lst:
            self._data.pop(frame_idx, None)
            self._annotated.discard(frame_idx)
        return True

    def clear_frame(self, frame_idx: int) -> bool:
        if frame_idx not in self._data:
            return False
        self._data.pop(frame_idx, None)
        self._annotated.discard(frame_idx)
        return True


# -----------------------------
# Nuke-like slider (label lane above ticks)
# -----------------------------

class FrameSlider(QSlider):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_point: Optional[int] = None
        self.out_point: Optional[int] = None
        self.annotated_frames: Set[int] = set()

    def set_in_out(self, i: Optional[int], o: Optional[int]) -> None:
        self.in_point = i
        self.out_point = o
        self.update()

    def set_annotated_frames(self, frames: Set[int]) -> None:
        self.annotated_frames = set(frames)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        frames = self.maximum() + 1
        if frames <= 1:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        w = self.width()
        h = self.height()
        left_margin = 8
        right_margin = 8
        start_x = left_margin
        end_x = max(left_margin, w - right_margin)
        span = max(1, end_x - start_x)

        # Frame label box sizing
        font = QFont(self.font())
        font.setPointSize(max(10, font.pointSize() + 2))
        font.setBold(True)
        painter.setFont(font)
        fm = QFontMetrics(font)
        val = self.value()
        txt = str(val + 1)
        tw = fm.horizontalAdvance(txt)
        th = fm.height()
        pad_x, pad_y = 8, 3
        box_w = tw + pad_x * 2
        box_h = th + pad_y * 2

        label_lane_h = min(box_h + 8, h - 10)
        tick_top = label_lane_h
        tick_h = max(10, h - tick_top)

        y1 = tick_top + tick_h // 2 + 2
        y2 = y1 + 5

        # IN-OUT range background
        if self.in_point is not None and self.out_point is not None:
            a = min(self.in_point, self.out_point)
            b = max(self.in_point, self.out_point)
            x1 = start_x + int(a / (frames - 1) * span)
            x2 = start_x + int(b / (frames - 1) * span)
            painter.fillRect(QRect(min(x1, x2), 0, max(1, abs(x2 - x1)), h), QColor(255, 230, 0, 26))

        # ticks
        max_ticks = 500
        step = max(1, frames // max_ticks)
        painter.setPen(QPen(self.palette().mid().color()))
        for i in range(0, frames, step):
            x = start_x + int((i / (frames - 1)) * span)
            if i % 50 == 0:
                painter.drawLine(x, y1 - 4, x, y2 + 4)
            else:
                painter.drawLine(x, y1, x, y2)

        # annotated markers (blue)
        if self.annotated_frames:
            painter.setPen(QPen(QColor(80, 144, 255)))
            for i in self.annotated_frames:
                if 0 <= i <= frames - 1:
                    x = start_x + int((i / (frames - 1)) * span)
                    painter.drawLine(x, tick_top + 2, x, tick_top + 10)

        # IN/OUT markers
        for point, color in ((self.in_point, QColor(80, 255, 80)), (self.out_point, QColor(255, 80, 80))):
            if point is None:
                continue
            x = start_x + int((point / (frames - 1)) * span)
            painter.setPen(QPen(color))
            painter.drawLine(x, 0, x, h)

        # playhead
        x = start_x + int((val / (frames - 1)) * span)
        painter.setPen(QPen(QColor(255, 230, 0)))
        painter.drawLine(x, 0, x, h)

        # frame label (in label lane)
        bx = int(x - box_w / 2)
        bx = max(2, min(self.width() - box_w - 2, bx))
        by = 2
        painter.fillRect(QRect(bx, by, box_w, box_h), QColor(0, 0, 0, 180))
        painter.setPen(QPen(QColor(255, 230, 0)))
        painter.drawRect(QRect(bx, by, box_w, box_h))
        painter.drawText(QRect(bx, by, box_w, box_h), Qt.AlignCenter, txt)

        painter.end()


# -----------------------------
# Swatch (contrast outline + checkmark)
# -----------------------------

class ColorSwatchButton(QToolButton):
    def __init__(self, color: QColor) -> None:
        super().__init__()
        self.color = QColor(color)
        self.setCheckable(True)
        self.setEnabled(False)
        self.setAutoRaise(True)
        self.setFixedSize(22, 22)

    def _luma(self) -> float:
        c = self.color
        return 0.2126 * c.red() + 0.7152 * c.green() + 0.0722 * c.blue()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        r = self.rect().adjusted(1, 1, -1, -1)
        fill = self.color if self.isEnabled() else self.color.darker(180)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(r, 4, 4)

        # base border
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(40, 40, 40), 1))
        p.drawRoundedRect(r, 4, 4)

        if self.isChecked():
            # high-contrast double outline
            p.setPen(QPen(QColor(245, 245, 245), 2))
            p.drawRoundedRect(r, 4, 4)
            p.setPen(QPen(QColor(0, 0, 0), 1))
            p.drawRoundedRect(r.adjusted(2, 2, -2, -2), 3, 3)

            # checkmark
            check_color = QColor(0, 0, 0) if self._luma() > 140 else QColor(255, 255, 255)
            p.setPen(QPen(check_color, 2))
            x1, y1 = r.left() + 5, r.center().y()
            x2, y2 = r.left() + 9, r.bottom() - 6
            x3, y3 = r.right() - 5, r.top() + 6
            p.drawLine(x1, y1, x2, y2)
            p.drawLine(x2, y2, x3, y3)

        p.end()


# -----------------------------
# Video canvas + overlay drawing
# -----------------------------

class VideoCanvas(QWidget):
    def __init__(self, player: "RamVideoPlayer") -> None:
        super().__init__()
        self.player = player
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)

        self._pixmap: Optional[QPixmap] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._scale: float = 1.0
        self._offx: float = 0.0
        self._offy: float = 0.0

        self._drawing_active: bool = False
        self._current_stroke_pts: List[Tuple[float, float]] = []

    def set_frame_pixmap(self, pixmap: QPixmap, frame_w: int, frame_h: int) -> None:
        self._pixmap = pixmap
        self._frame_size = (frame_w, frame_h)
        self.update()

    def _compute_transform(self) -> None:
        if not self._pixmap or not self._frame_size:
            self._scale = 1.0
            self._offx = 0.0
            self._offy = 0.0
            return

        fw, fh = self._frame_size
        lw, lh = self.width(), self.height()
        scale = min(lw / fw, lh / fh)
        offx = (lw - fw * scale) / 2.0
        offy = (lh - fh * scale) / 2.0
        self._scale = scale
        self._offx = offx
        self._offy = offy

        self.player.display_info = {
            "frame_width": fw,
            "frame_height": fh,
            "scale": scale,
            "offset_x": offx,
            "offset_y": offy,
            "disp_w": fw * scale,
            "disp_h": fh * scale,
        }

    def _pos_to_frame_xy(self, pos: QPoint) -> Optional[Tuple[float, float]]:
        if not self._frame_size:
            return None
        self._compute_transform()
        fw, fh = self._frame_size
        x = float(pos.x())
        y = float(pos.y())
        if (x < self._offx or x >= self._offx + fw * self._scale or
                y < self._offy or y >= self._offy + fh * self._scale):
            return None
        fx = (x - self._offx) / self._scale
        fy = (y - self._offy) / self._scale
        return max(0.0, min(fw - 1.0, fx)), max(0.0, min(fh - 1.0, fy))

    def _frame_xy_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        return self._offx + x * self._scale, self._offy + y * self._scale

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if not self._pixmap or not self._frame_size:
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(self.rect(), Qt.AlignCenter, "Nincs betöltött videó")
            painter.end()
            return

        self._compute_transform()
        fw, fh = self._frame_size
        target = QRect(int(self._offx), int(self._offy), int(fw * self._scale), int(fh * self._scale))
        painter.drawPixmap(target, self._pixmap)

        if self.player.qc_visible:
            strokes = self.player.annotations.strokes_for(self.player.current_index)
            for st in strokes:
                pen = QPen(st.color)
                pen.setWidth(max(1, int(st.width * self._scale)))
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(pen)
                pts = st.points
                for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                    sx1, sy1 = self._frame_xy_to_screen(x1, y1)
                    sx2, sy2 = self._frame_xy_to_screen(x2, y2)
                    painter.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))

            if self._drawing_active and self._current_stroke_pts and self.player.active_draw_color is not None:
                pen = QPen(self.player.active_draw_color)
                pen.setWidth(max(1, int(self.player.stroke_width * self._scale)))
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(pen)
                pts = self._current_stroke_pts
                for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                    sx1, sy1 = self._frame_xy_to_screen(x1, y1)
                    sx2, sy2 = self._frame_xy_to_screen(x2, y2)
                    painter.drawLine(int(sx1), int(sy1), int(sx2), int(sy2))

        painter.end()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.player.on_video_mouse_move(event.pos())
        if self.player.active_draw_color is not None and self.player.qc_visible and self._drawing_active:
            xy = self._pos_to_frame_xy(event.pos())
            if xy is not None:
                self._current_stroke_pts.append(xy)
                self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self.player.active_draw_color is not None and self.player.qc_visible and self._pixmap:
            xy = self._pos_to_frame_xy(event.pos())
            if xy is not None:
                self._drawing_active = True
                self._current_stroke_pts = [xy]
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drawing_active:
            self._drawing_active = False
            if len(self._current_stroke_pts) >= 2 and self.player.active_draw_color is not None:
                st = Stroke(
                    color=QColor(self.player.active_draw_color),
                    width=self.player.stroke_width,
                    points=list(self._current_stroke_pts),
                )
                self.player.annotations.add_stroke(self.player.current_index, st)
                self.player._refresh_annotated_ui()
            self._current_stroke_pts = []
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self.player.on_video_leave()
        super().leaveEvent(event)


# -----------------------------
# Main player
# -----------------------------

class RamVideoPlayer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"RAM Video Player QC v{__version__}")
        self.setFocusPolicy(Qt.StrongFocus)

        self.frames: List[np.ndarray] = []
        self.current_index = 0
        self.total_frames = 0
        self.fps = 25.0

        self.source_path: Optional[str] = None
        self.source_basename: Optional[str] = None
        self.file_size_bytes: int = 0
        self.estimated_frame_count: int = 0

        self.display_info: Optional[dict] = None
        self.last_mouse_pos: Optional[QPoint] = None

        self.in_point: Optional[int] = None
        self.out_point: Optional[int] = None

        self.loop_enabled = True
        self.play_direction = 1  # 1 forward, -1 reverse
        self._playing_button: Optional[str] = None  # "fwd" / "rev" / None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.annotations = AnnotationStore()
        self.qc_visible = True

        self.stroke_width_min = 3
        self.stroke_width_max = 18
        self.stroke_width = self.stroke_width_min
        self.active_draw_color: Optional[QColor] = None

        self.video_canvas = VideoCanvas(self)

        self.slider = FrameSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTickPosition(QSlider.NoTicks)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.setMaximumHeight(44)

        self.load_progress = QProgressBar()
        self.load_progress.setRange(0, 100)
        self.load_progress.setValue(0)
        self.load_progress.setTextVisible(True)
        self.load_progress.setFormat("Nincs aktív betöltés")
        self.load_progress.setMaximumHeight(18)
        self._set_progress_color(QColor(58, 123, 213))

        self.load_info_label = QLabel("Fájl méret: - | Betöltve: -")
        self.ram_info_label = QLabel("Szabad RAM: -")
        self.info_label = QLabel("Frame: - / -")

        self.pixel_info_label = QLabel("R: -  G: -  B: -")
        self.pixel_info_label.setTextFormat(Qt.RichText)
        self.pixel_info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.pixel_info_frame = QFrame()
        self.pixel_info_frame.setFixedWidth(280)
        pf = QHBoxLayout()
        pf.setContentsMargins(8, 4, 8, 4)
        pf.addWidget(self.pixel_info_label)
        self.pixel_info_frame.setLayout(pf)

        self.open_button = QPushButton("Videó megnyitása")
        self.open_button.clicked.connect(self.open_video)

        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setChecked(True)
        self.loop_checkbox.stateChanged.connect(self._on_loop_changed)

        self.version_label = QLabel(f"Verzió: v{__version__}")
        self.version_label.setStyleSheet("color: #888;")
        self.changelog_button = QPushButton("Változások…")
        self.changelog_button.setMaximumWidth(120)
        self.changelog_button.clicked.connect(self.show_changelog)

        # Transport buttons
        self.step_back_button = self._tool_button("Frame step back", frame_step_icon(-1))
        self.play_back_button = self._tool_button("Play reverse", play_icon(-1))
        self.play_fwd_button = self._tool_button("Play forward", play_icon(+1))
        self.step_fwd_button = self._tool_button("Frame step forward", frame_step_icon(+1))

        self.step_back_button.clicked.connect(lambda: self.step_frame(-1))
        self.step_fwd_button.clicked.connect(lambda: self.step_frame(+1))
        self.play_back_button.clicked.connect(lambda: self._toggle_play_button("rev"))
        self.play_fwd_button.clicked.connect(lambda: self._toggle_play_button("fwd"))

        for b in (self.step_back_button, self.play_back_button, self.play_fwd_button, self.step_fwd_button):
            b.setEnabled(False)

        self.frame_edit = QLineEdit()
        self.frame_edit.setFixedWidth(90)
        self.frame_edit.setEnabled(False)
        self.frame_edit.returnPressed.connect(self._jump_to_frame_from_edit)

        self.prev_annot_button = QPushButton("⟵ Annot")
        self.next_annot_button = QPushButton("Annot ⟶")
        self.prev_annot_button.setEnabled(False)
        self.next_annot_button.setEnabled(False)
        self.prev_annot_button.clicked.connect(lambda: self.jump_annotated(-1))
        self.next_annot_button.clicked.connect(lambda: self.jump_annotated(+1))

        self.in_button = QPushButton("IN (I)")
        self.out_button = QPushButton("OUT (O)")
        self.clear_io_button = QPushButton("IN/OUT törlés")
        for b in (self.in_button, self.out_button, self.clear_io_button):
            b.setEnabled(False)
        self.in_button.clicked.connect(self.toggle_in)
        self.out_button.clicked.connect(self.toggle_out)
        self.clear_io_button.clicked.connect(self.clear_in_out)

        self.qc_checkbox = QCheckBox("QC réteg")
        self.qc_checkbox.setChecked(True)
        self.qc_checkbox.setEnabled(False)
        self.qc_checkbox.stateChanged.connect(self._on_qc_visible_changed)

        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.clear_draw_button = QPushButton("Clear frame")
        self.undo_button.setEnabled(False)
        self.clear_draw_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_stroke)
        self.clear_draw_button.clicked.connect(self.clear_current_frame_drawings)

        self.color_y = ColorSwatchButton(QColor(255, 230, 0))
        self.color_r = ColorSwatchButton(QColor(255, 80, 80))
        self.color_b = ColorSwatchButton(QColor(80, 144, 255))
        self.color_g = ColorSwatchButton(QColor(80, 255, 80))
        for sw in (self.color_y, self.color_r, self.color_b, self.color_g):
            sw.clicked.connect(lambda _=False, b=sw: self.toggle_color(b))

        self.stroke_slider = QSlider(Qt.Horizontal)
        self.stroke_slider.setEnabled(False)
        self.stroke_slider.setMinimum(self.stroke_width_min)
        self.stroke_slider.setMaximum(self.stroke_width_max)
        self.stroke_slider.setValue(self.stroke_width)
        self.stroke_slider.setFixedWidth(110)
        self.stroke_slider.valueChanged.connect(self._on_stroke_width_changed)
        self.stroke_slider.setToolTip("Vonalvastagság")

        self.save_png_button = QPushButton("PNG (aktuális)")
        self.export_annot_button = QPushButton("Export annotált PNG-k…")
        self.save_png_button.setEnabled(False)
        self.export_annot_button.setEnabled(False)
        self.save_png_button.clicked.connect(self.save_current_frame_png_with_overlay)
        self.export_annot_button.clicked.connect(self.export_annotated_frames_png)

        self._build_layout()
        self._apply_dark_theme()
        self._style_pixel_info_box()
        self.update_ram_info()

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self._refresh_transport_icons()

    # ----- UI helpers -----
    def _tool_button(self, tooltip: str, icon: QIcon) -> QToolButton:
        b = QToolButton()
        b.setToolTip(tooltip)
        b.setIcon(icon)
        b.setAutoRaise(True)
        b.setIconSize(QSize(18, 18))
        b.setFixedSize(28, 28)
        return b

    def _build_layout(self) -> None:
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.load_info_label)
        info_layout.addStretch()
        info_layout.addWidget(self.ram_info_label)

        transport = QHBoxLayout()
        transport.setAlignment(Qt.AlignHCenter)
        transport.addWidget(self.step_back_button)
        transport.addWidget(self.play_back_button)
        transport.addWidget(self.play_fwd_button)
        transport.addWidget(self.step_fwd_button)
        transport.addSpacing(12)
        transport.addWidget(self.frame_edit)
        transport.addSpacing(12)
        transport.addWidget(self.prev_annot_button)
        transport.addWidget(self.next_annot_button)

        sw_row = QHBoxLayout()
        sw_row.setSpacing(6)
        sw_row.addWidget(self.color_y)
        sw_row.addWidget(self.color_r)
        sw_row.addWidget(self.color_b)
        sw_row.addWidget(self.color_g)

        sw_block = QVBoxLayout()
        sw_block.setSpacing(4)
        sw_block.addLayout(sw_row)
        sw_block.addWidget(self.stroke_slider, alignment=Qt.AlignLeft)

        sw_widget = QWidget()
        sw_widget.setLayout(sw_block)

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.loop_checkbox)
        controls.addWidget(self.info_label)
        controls.addStretch()
        controls.addWidget(self.in_button)
        controls.addWidget(self.out_button)
        controls.addWidget(self.clear_io_button)
        controls.addSpacing(12)
        controls.addWidget(self.qc_checkbox)
        controls.addWidget(sw_widget)
        controls.addWidget(self.undo_button)
        controls.addWidget(self.clear_draw_button)
        controls.addSpacing(12)
        controls.addWidget(self.save_png_button)
        controls.addWidget(self.export_annot_button)
        controls.addSpacing(12)
        controls.addWidget(self.pixel_info_frame)
        controls.addSpacing(12)
        controls.addWidget(self.version_label)
        controls.addWidget(self.changelog_button)

        main = QVBoxLayout()
        main.addWidget(self.video_canvas)
        main.addWidget(self.slider)
        main.addWidget(self.load_progress)
        main.addLayout(info_layout)
        main.addLayout(transport)
        main.addLayout(controls)
        self.setLayout(main)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background-color: #2b2b2b; color: #f0f0f0; }
            QPushButton {
                background-color: #3c3c3c; color: #f0f0f0;
                border: 1px solid #555555; border-radius: 4px; padding: 4px 8px;
            }
            QPushButton:disabled { background-color: #555555; color: #888888; }
            QToolButton { background: transparent; border: 1px solid #555555; border-radius: 4px; padding: 2px; }
            QToolButton:disabled { border: 1px solid #444444; }
            QLineEdit { background-color: #1f1f1f; border: 1px solid #555555; border-radius: 4px; padding: 4px 6px; }
            QSlider::groove:horizontal { background: #555555; height: 4px; }
            QSlider::handle:horizontal { background: #dddddd; width: 10px; margin: -5px 0; border-radius: 5px; }
            """
        )

    def _style_pixel_info_box(self) -> None:
        self.pixel_info_frame.setStyleSheet(
            """
            QFrame { background-color: #111111; border: 1px solid #555555; border-radius: 4px; }
            QLabel { background: transparent; color: #f0f0f0; }
            """
        )

    def _set_progress_color(self, color: QColor) -> None:
        self.load_progress.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid #444;
                border-radius: 4px;
                text-align: center;
                color: white;
                background-color: #222;
            }}
            QProgressBar::chunk {{
                background-color: {color.name()};
            }}
            """
        )

    def _enable_controls(self, enabled: bool) -> None:
        for w in (
            self.slider,
            self.step_back_button, self.play_back_button, self.play_fwd_button, self.step_fwd_button,
            self.frame_edit,
            self.prev_annot_button, self.next_annot_button,
            self.in_button, self.out_button, self.clear_io_button,
            self.qc_checkbox,
            self.color_y, self.color_r, self.color_b, self.color_g,
            self.stroke_slider,
            self.undo_button, self.clear_draw_button,
            self.save_png_button, self.export_annot_button,
        ):
            w.setEnabled(enabled)

    # ----- Changelog -----
    def show_changelog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Változások – RAM Video Player QC v{__version__}")
        dlg.resize(760, 560)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(render_changelog_text())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)

        layout = QVBoxLayout()
        layout.addWidget(text)
        layout.addWidget(buttons)
        dlg.setLayout(layout)
        dlg.exec()

    # ----- Video loading -----
    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Válassz videót",
            "",
            "Video Files (*.mov *.mp4 *.m4v *.avi);;All Files (*.*)"
        )
        if not path:
            return

        self.source_path = path
        self.source_basename = os.path.splitext(os.path.basename(path))[0]
        try:
            self.file_size_bytes = os.path.getsize(path)
        except OSError:
            self.file_size_bytes = 0

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Hiba", "Nem sikerült megnyitni a videót.")
            return

        fps_read = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps_read if fps_read and fps_read > 0 else 25.0
        self.estimated_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        self.stop()
        self.frames.clear()
        self.annotations.clear_all()
        self.current_index = 0
        self.total_frames = 0
        self.in_point = None
        self.out_point = None
        self.slider.set_in_out(None, None)
        self.slider.set_annotated_frames(set())
        self.qc_visible = True
        self.qc_checkbox.setChecked(True)
        self._clear_color_selection()
        self.stroke_width = self.stroke_width_min
        self.stroke_slider.setValue(self.stroke_width)

        file_size_str = format_bytes(self.file_size_bytes) if self.file_size_bytes else "ismeretlen"
        self.load_progress.setValue(0)
        self.load_progress.setFormat("Betöltés: %p%")
        self._set_progress_color(QColor(58, 123, 213))
        self.load_info_label.setText(f"Fájl méret: {file_size_str} | Betöltve: -")

        loaded_frames = 0
        error_message: Optional[str] = None

        while True:
            try:
                ret, frame_bgr = cap.read()
            except Exception as e:
                error_message = str(e)
                break
            if not ret:
                break

            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                error_message = f"Színkonverziós hiba: {e}"
                break

            self.frames.append(frame_rgb)
            loaded_frames += 1

            if self.estimated_frame_count > 0:
                progress = int(loaded_frames * 100 / self.estimated_frame_count)
                self.load_progress.setValue(min(100, progress))
                if self.file_size_bytes:
                    est_bytes = int(self.file_size_bytes * loaded_frames / max(1, self.estimated_frame_count))
                    self.load_info_label.setText(f"Fájl méret: {file_size_str} | Betöltve: {format_bytes(est_bytes)}")

            if loaded_frames % 30 == 0:
                self.update_ram_info()
                QApplication.processEvents()

        cap.release()

        self.total_frames = len(self.frames)
        if self.total_frames == 0:
            self.load_progress.setFormat("Betöltés sikertelen")
            QMessageBox.warning(self, "Üres videó", "Nem sikerült képkockákat beolvasni.")
            return

        self.load_progress.setValue(100)
        self.load_progress.setFormat("Betöltés kész")
        self._set_progress_color(QColor(39, 174, 96))
        if self.file_size_bytes:
            self.load_info_label.setText(
                f"Fájl méret: {file_size_str} | Betöltve: {format_bytes(self.file_size_bytes)}"
            )

        if error_message is not None:
            QMessageBox.warning(
                self,
                "Figyelem",
                f"A videó betöltése közben hiba történt:\n\n{error_message}\n\n"
                f"A már betöltött {self.total_frames} képkocka lejátszható, de a videó nem teljes."
            )

        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.setValue(0)

        self._enable_controls(True)
        self.update_ram_info()
        self.update_frame()

    def update_ram_info(self) -> None:
        if psutil is None:
            self.ram_info_label.setText("Szabad RAM: (psutil nincs telepítve)")
            return
        vm = psutil.virtual_memory()
        self.ram_info_label.setText(
            f"Szabad RAM: {format_bytes(int(vm.available))} / összes: {format_bytes(int(vm.total))}"
        )

    # ----- Frame display -----
    def _make_qimage_for_display(self, frame: np.ndarray) -> QImage:
        rgb = np.ascontiguousarray(frame)
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)

    def update_frame(self) -> None:
        if not self.frames:
            return
        frame = self.frames[self.current_index]
        h, w = frame.shape[:2]
        qimg = self._make_qimage_for_display(frame)
        self.video_canvas.set_frame_pixmap(QPixmap.fromImage(qimg), w, h)

        self.info_label.setText(f"Frame: {self.current_index + 1} / {self.total_frames}")

        self.slider.set_in_out(self.in_point, self.out_point)
        self.slider.set_annotated_frames(self.annotations.annotated_set())

        if not self.frame_edit.hasFocus():
            self.frame_edit.blockSignals(True)
            self.frame_edit.setText(str(self.current_index + 1))
            self.frame_edit.blockSignals(False)

        if self.last_mouse_pos is not None:
            self.on_video_mouse_move(self.last_mouse_pos)

        self._refresh_annotated_ui()

    def _on_slider_changed(self, value: int) -> None:
        if not self.frames:
            return
        self.current_index = int(value)
        self.update_frame()

    def _set_index(self, idx: int) -> None:
        if not self.frames:
            return
        idx = max(0, min(self.total_frames - 1, int(idx)))
        self.current_index = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.update_frame()

    # ----- Playback -----
    def _on_loop_changed(self) -> None:
        self.loop_enabled = self.loop_checkbox.isChecked()

    def _effective_range(self) -> Tuple[int, int]:
        if self.in_point is None or self.out_point is None:
            return 0, max(0, self.total_frames - 1)
        a = max(0, min(self.total_frames - 1, min(self.in_point, self.out_point)))
        b = max(0, min(self.total_frames - 1, max(self.in_point, self.out_point)))
        return a, b

    def _tick(self) -> None:
        if not self.frames:
            return
        start, end = self._effective_range()
        idx = self.current_index + self.play_direction

        if idx > end:
            if self.loop_enabled:
                idx = start
            else:
                self.stop()
                return
        elif idx < start:
            if self.loop_enabled:
                idx = end
            else:
                self.stop()
                return
        self._set_index(idx)

    def _refresh_transport_icons(self) -> None:
        # While playing: active play button becomes Stop icon
        if self.timer.isActive() and self._playing_button == "fwd":
            self.play_fwd_button.setIcon(stop_icon())
            self.play_back_button.setIcon(play_icon(-1))
        elif self.timer.isActive() and self._playing_button == "rev":
            self.play_back_button.setIcon(stop_icon())
            self.play_fwd_button.setIcon(play_icon(+1))
        else:
            self.play_back_button.setIcon(play_icon(-1))
            self.play_fwd_button.setIcon(play_icon(+1))

        self.step_back_button.setIcon(frame_step_icon(-1))
        self.step_fwd_button.setIcon(frame_step_icon(+1))

    def play(self, direction: int) -> None:
        if not self.frames:
            return
        self.play_direction = 1 if direction >= 0 else -1
        interval_ms = max(1, int(1000 / max(1.0, self.fps)))
        self.timer.start(interval_ms)
        self._refresh_transport_icons()

    def stop(self) -> None:
        self.timer.stop()
        self._playing_button = None
        self._refresh_transport_icons()

    def _toggle_play_pause(self) -> None:
        if not self.frames:
            return
        if self.timer.isActive():
            self.stop()
        else:
            # resume in last direction (default forward)
            which = "fwd" if self.play_direction >= 0 else "rev"
            self._toggle_play_button(which)

    def _toggle_play_button(self, which: str) -> None:
        # which: "fwd" or "rev"
        if not self.frames:
            return

        if self.timer.isActive() and self._playing_button == which:
            self.stop()
            return

        self._playing_button = which
        self.play_direction = 1 if which == "fwd" else -1
        self.play(self.play_direction)

    def step_frame(self, delta: int) -> None:
        if not self.frames:
            return
        self.stop()

        start, end = self._effective_range()
        idx = self.current_index + int(delta)

        if self.in_point is not None and self.out_point is not None:
            if idx > end:
                idx = start if self.loop_enabled else end
            if idx < start:
                idx = end if self.loop_enabled else start
        else:
            if idx >= self.total_frames:
                idx = 0 if self.loop_enabled else self.total_frames - 1
            if idx < 0:
                idx = self.total_frames - 1 if self.loop_enabled else 0

        self._set_index(idx)

    # ----- IN/OUT -----
    def toggle_in(self) -> None:
        if self.in_point == self.current_index:
            self.in_point = None
        else:
            self.in_point = self.current_index
            if self.out_point is not None and self.in_point > self.out_point:
                self.in_point, self.out_point = self.out_point, self.in_point
        self.update_frame()

    def toggle_out(self) -> None:
        if self.out_point == self.current_index:
            self.out_point = None
        else:
            self.out_point = self.current_index
            if self.in_point is not None and self.in_point > self.out_point:
                self.in_point, self.out_point = self.out_point, self.in_point
        self.update_frame()

    def clear_in_out(self) -> None:
        self.in_point = None
        self.out_point = None
        self.update_frame()

    # ----- Frame jump -----
    def _jump_to_frame_from_edit(self) -> None:
        if not self.frames:
            return
        txt = self.frame_edit.text().strip()
        if not txt:
            return
        try:
            n = int(txt)
        except ValueError:
            QMessageBox.warning(self, "Hibás frame", "Kérlek számot adj meg (1-től indul).")
            self.update_frame()
            return
        n = max(1, min(self.total_frames, n))
        self._set_index(n - 1)

    # ----- Annotated navigation -----
    def _refresh_annotated_ui(self) -> None:
        ann = self.annotations.annotated_sorted()
        has_any = bool(ann)
        self.prev_annot_button.setEnabled(has_any)
        self.next_annot_button.setEnabled(has_any)
        self.export_annot_button.setEnabled(has_any)

    def jump_annotated(self, direction: int) -> None:
        ann = self.annotations.annotated_sorted()
        if not ann:
            return
        idx = self.current_index
        if direction >= 0:
            for a in ann:
                if a > idx:
                    self._set_index(a)
                    return
            self._set_index(ann[0])
        else:
            for a in reversed(ann):
                if a < idx:
                    self._set_index(a)
                    return
            self._set_index(ann[-1])

    # ----- QC drawing -----
    def _clear_color_selection(self) -> None:
        for b in (self.color_y, self.color_r, self.color_b, self.color_g):
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)
        self.active_draw_color = None

    def toggle_color(self, btn: ColorSwatchButton) -> None:
        if btn.isChecked():
            for b in (self.color_y, self.color_r, self.color_b, self.color_g):
                if b is not btn:
                    b.blockSignals(True)
                    b.setChecked(False)
                    b.blockSignals(False)
            self.active_draw_color = QColor(btn.color)
        else:
            self.active_draw_color = None

    def _on_stroke_width_changed(self, value: int) -> None:
        self.stroke_width = int(value)

    def _on_qc_visible_changed(self) -> None:
        self.qc_visible = self.qc_checkbox.isChecked()
        self.video_canvas.update()

    def undo_stroke(self) -> None:
        if self.annotations.undo_last(self.current_index):
            self.video_canvas.update()
            self.slider.set_annotated_frames(self.annotations.annotated_set())
            self._refresh_annotated_ui()

    def clear_current_frame_drawings(self) -> None:
        if self.annotations.clear_frame(self.current_index):
            self.video_canvas.update()
            self.slider.set_annotated_frames(self.annotations.annotated_set())
            self._refresh_annotated_ui()

    # ----- Pixel inspector (normalized only) -----
    def on_video_mouse_move(self, pos: QPoint) -> None:
        if not self.frames or self.display_info is None:
            self.last_mouse_pos = None
            return

        di = self.display_info
        x = float(pos.x())
        y = float(pos.y())

        if (x < di["offset_x"] or x >= di["offset_x"] + di["disp_w"] or
                y < di["offset_y"] or y >= di["offset_y"] + di["disp_h"]):
            self.last_mouse_pos = None
            return

        self.last_mouse_pos = QPoint(pos)

        fx = int((x - di["offset_x"]) / di["scale"])
        fy = int((y - di["offset_y"]) / di["scale"])
        fw = int(di["frame_width"])
        fh = int(di["frame_height"])
        fx = max(0, min(fw - 1, fx))
        fy = max(0, min(fh - 1, fy))

        frame = self.frames[self.current_index]
        r = int(frame[fy, fx, 0])
        g = int(frame[fy, fx, 1])
        b = int(frame[fy, fx, 2])

        rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
        html = (
            f'<span style="color:#ff5555">R: {rn:0.5f}</span>  '
            f'<span style="color:#55ff55">G: {gn:0.5f}</span>  '
            f'<span style="color:#5590ff">B: {bn:0.5f}</span>'
        )
        self.pixel_info_label.setText(html)

    def on_video_leave(self) -> None:
        self.last_mouse_pos = None
        self.pixel_info_label.setText("R: -  G: -  B: -")

    # ----- Export -----
    def _compose_qimage_with_overlay(self, frame_idx: int) -> QImage:
        frame = self.frames[frame_idx]
        qimg = self._make_qimage_for_display(frame).copy().convertToFormat(QImage.Format_ARGB32)

        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.Antialiasing, True)
        for st in self.annotations.strokes_for(frame_idx):
            pen = QPen(st.color)
            pen.setWidth(max(1, int(st.width)))
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            pts = st.points
            for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.end()
        return qimg

    def save_current_frame_png_with_overlay(self) -> None:
        if not self.frames:
            return
        frame_number = self.current_index + 1
        suffix = f"QC{frame_number:04d}"
        default_name = f"{self.source_basename}_{suffix}.png" if self.source_basename else f"frame_{suffix}.png"
        initial_dir = os.path.dirname(self.source_path) if self.source_path else os.path.expanduser("~")
        default_path = os.path.join(initial_dir or os.path.expanduser("~"), default_name)

        save_path, _ = QFileDialog.getSaveFileName(self, "PNG mentése (rajzokkal)", default_path, "PNG képek (*.png)")
        if not save_path:
            return
        if not self._compose_qimage_with_overlay(self.current_index).save(save_path, "PNG"):
            QMessageBox.warning(self, "PNG mentési hiba", "Nem sikerült a PNG fájl mentése.")

    def export_annotated_frames_png(self) -> None:
        ann = self.annotations.annotated_sorted()
        if not ann:
            QMessageBox.information(self, "Nincs annotáció", "Nincs annotált frame.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Válassz export könyvtárat", os.path.expanduser("~"))
        if not out_dir:
            return

        base = self.source_basename or "clip"
        failures: List[str] = []

        for idx in ann:
            fname = f"{base}_QC{idx + 1:04d}.png"
            fpath = os.path.join(out_dir, fname)
            if not self._compose_qimage_with_overlay(idx).save(fpath, "PNG"):
                failures.append(fname)
            QApplication.processEvents()

        if failures:
            QMessageBox.warning(self, "Export kész (hibákkal)", "Nem sikerült menteni:\n" + "\n".join(failures[:20]))
        else:
            QMessageBox.information(self, "Export kész", f"Sikeres export: {len(ann)} PNG\n\n{out_dir}")

    # ----- Global key handling -----
    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.KeyPress and self.isActiveWindow() and self.frames:
            if isinstance(event, QKeyEvent):
                key = event.key()
                mods = event.modifiers()

                if (mods & Qt.ControlModifier) and key == Qt.Key_Z:
                    self.undo_stroke()
                    return True
                if mods & Qt.ControlModifier:
                    return False

                if key == Qt.Key_Space:
                    self._toggle_play_pause()
                    return True
                if key == Qt.Key_Left:
                    self.step_frame(-1)
                    return True
                if key == Qt.Key_Right:
                    self.step_frame(+1)
                    return True
                if key == Qt.Key_I:
                    self.toggle_in()
                    return True
                if key == Qt.Key_O:
                    self.toggle_out()
                    return True
                if key == Qt.Key_V:
                    self.qc_checkbox.toggle()
                    self._on_qc_visible_changed()
                    return True
                if key == Qt.Key_PageUp:
                    self.jump_annotated(-1)
                    return True
                if key == Qt.Key_PageDown:
                    self.jump_annotated(+1)
                    return True

        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.frames:
            self.video_canvas.update()


def main() -> None:
    app = QApplication(sys.argv)
    player = RamVideoPlayer()
    player.resize(1500, 850)
    player.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
