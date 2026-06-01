#!/usr/bin/env python3
"""
Full-screen Expo welcome splash: staggered text + particle field (PyQt5 only).
"""

import math
import random

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGraphicsOpacityEffect,
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup, pyqtSignal, QPointF,
)
from PyQt5.QtGui import QPainter, QColor, QPen, QFont


class _ParticleField(QWidget):
    """Breathing grid + drift particles + corner brackets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._t = 0.0
        self._dots = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def _tick(self):
        self._t += 0.04
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = max(self.width(), 1), max(self.height(), 1)
        self._dots = []
        rng = random.Random(42)
        for _ in range(140):
            self._dots.append({
                "x": rng.random() * w,
                "y": rng.random() * h,
                "vx": (rng.random() - 0.5) * 0.35,
                "vy": (rng.random() - 0.5) * 0.35,
                "r": 0.6 + rng.random() * 1.8,
                "p": rng.random() * 6.28,
            })

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            p.end()
            return

        p.fillRect(0, 0, w, h, QColor(12, 14, 20))

        for i in range(0, 80, 4):
            a = max(0, 8 - i // 10)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, a))
            p.drawRect(i, i, w - 2 * i, h - 2 * i)

        g_alpha = 18 + int(10 * math.sin(self._t * 0.6))
        pen = QPen(QColor(201, 162, 39, g_alpha))
        pen.setWidth(0)
        p.setPen(pen)
        step = 48
        off = int((self._t * 12) % step)
        for x in range(-step, w + step, step):
            p.drawLine(x + off, 0, x + off, h)
        for y in range(-step, h + step, step):
            p.drawLine(0, y + off, w, y + off)

        for d in self._dots:
            d["x"] = (d["x"] + d["vx"]) % w
            d["y"] = (d["y"] + d["vy"]) % h
            alpha = 40 + int(40 * math.sin(self._t * 1.2 + d["p"]))
            alpha = max(0, min(120, alpha))
            col = QColor(201, 162, 39, alpha)
            p.setBrush(col)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(d["x"], d["y"]), d["r"], d["r"])

        p.setPen(QPen(QColor(201, 162, 39, 100), 2))
        m = 24
        p.drawLine(m, m, m + 80, m)
        p.drawLine(m, m, m, m + 80)
        p.drawLine(w - m, h - m, w - m - 80, h - m)
        p.drawLine(w - m, h - m, w - m, h - m - 80)

        p.end()


class ExpoWelcomeOverlay(QWidget):
    """Staggered fade-in; Skip / Continue; Escape or Enter dismisses."""

    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._seq = None
        self._bg = _ParticleField(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(48, 64, 48, 48)
        root.addStretch(1)

        self._title = QLabel("Formula-Tron")
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setFont(QFont("Segoe UI", 36, QFont.Bold))
        self._title.setStyleSheet("color: #e8d4a8; background: transparent; letter-spacing: 2px;")

        self._rule1 = QFrame()
        self._rule1.setFixedHeight(2)
        self._rule1.setStyleSheet("background:#c9a227; max-height:2px; border:none;")

        self._sub = QLabel("Vision-based autonomous racing — McMaster Mechatronics Capstone")
        self._sub.setAlignment(Qt.AlignCenter)
        self._sub.setWordWrap(True)
        self._sub.setFont(QFont("Segoe UI", 14))
        self._sub.setStyleSheet("color: #aeb8c9; background: transparent;")

        self._b1 = QLabel("• Real-time lane perception (HSV / polynomial lookahead)")
        self._b2 = QLabel("• Model predictive control & safety interlocks")
        self._b3 = QLabel("• ROS 2 Foxy — F1TENTH platform integration")
        ff = QFont("Consolas", 12)
        if not ff.exactMatch():
            ff = QFont("Segoe UI", 12)
        for b in (self._b1, self._b2, self._b3):
            b.setAlignment(Qt.AlignCenter)
            b.setFont(ff)
            b.setStyleSheet("color: #e0e6f0; background: transparent;")

        self._rule2 = QFrame()
        self._rule2.setFixedHeight(1)
        self._rule2.setStyleSheet("background:#3d4a5c; max-height:1px; border:none;")

        for w in (self._title, self._rule1, self._sub, self._b1, self._b2, self._b3, self._rule2):
            root.addWidget(w)
        root.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        skip = QPushButton("Skip")
        skip.setCursor(Qt.PointingHandCursor)
        skip.setStyleSheet(
            "QPushButton { color: #888; background: transparent; border: 1px solid #555; "
            "border-radius: 6px; padding: 10px 28px; font-size: 13px; }"
            "QPushButton:hover { color: #fff; border-color: #c9a227; }"
        )
        go = QPushButton("Continue to Expo")
        go.setDefault(True)
        go.setCursor(Qt.PointingHandCursor)
        go.setStyleSheet(
            "QPushButton { color: #0d0e12; background: #c9a227; border: none; "
            "border-radius: 8px; padding: 12px 32px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background: #e0c45c; }"
        )
        skip.clicked.connect(self._dismiss)
        go.clicked.connect(self._dismiss)
        btn_row.addWidget(skip)
        btn_row.addSpacing(16)
        btn_row.addWidget(go)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self._fade_widgets = [
            self._title, self._rule1, self._sub, self._b1, self._b2, self._b3, self._rule2,
        ]
        for w in self._fade_widgets:
            eff = QGraphicsOpacityEffect(w)
            eff.setOpacity(0.0)
            w.setGraphicsEffect(eff)

        self.setFocusPolicy(Qt.StrongFocus)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._bg.setGeometry(self.rect())

    def showEvent(self, event):
        super().showEvent(event)
        self._bg.lower()
        self._bg.setGeometry(self.rect())
        self.setFocus(Qt.OtherFocusReason)
        self._start_animation()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter):
            self._dismiss()
        else:
            super().keyPressEvent(event)

    def _start_animation(self):
        if self._seq is not None:
            self._seq.stop()
            self._seq.deleteLater()
        self._seq = QSequentialAnimationGroup(self)
        for w in self._fade_widgets:
            eff = w.graphicsEffect()
            if eff is None:
                continue
            anim = QPropertyAnimation(eff, b"opacity")
            anim.setDuration(420)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            self._seq.addAnimation(anim)
        self._seq.start()

    def _dismiss(self):
        if self._seq is not None:
            self._seq.stop()
            self._seq.deleteLater()
            self._seq = None
        self.finished.emit()
        self.hide()
        self.deleteLater()
