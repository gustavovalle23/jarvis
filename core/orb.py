from __future__ import annotations

import math
import random
import time
from datetime import datetime
from typing import Callable

import numpy as np
import pygame

from .state import Mode, State

LOOKS = {
    Mode.BOOT:      ((70, 110, 255), 0.35, 0.35, 1.0),
    Mode.IDLE:      ((0, 140, 255),  0.22, 0.10, 0.7),
    Mode.LISTENING: ((0, 255, 200),  0.45, 0.28, 1.0),
    Mode.THINKING:  ((165, 85, 255), 1.80, 0.45, 1.6),
    Mode.SPEAKING:  ((40, 205, 255), 0.55, 0.22, 1.4),
    Mode.ERROR:     ((255, 55, 70),  0.08, 0.55, 2.2),
}

N_POINTS = 760
N_BUCKETS = 12
N_BARS = 128
CAMERA = 3.4


def _glow(radius: float, intensity: float, falloff: float = 2.2, core: float = 0.0) -> pygame.Surface:
    r = max(1, int(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y) / r
    g = np.clip(1 - d, 0, 1) ** falloff
    if core:
        g += core * np.clip(1 - d * 4, 0, 1)
    g = (np.clip(g * intensity, 0, 1) * 255).astype(np.uint8)
    return pygame.surfarray.make_surface(np.dstack([g, g, g]))


def _tint(surf: pygame.Surface, color) -> pygame.Surface:
    s = surf.copy()
    s.fill(color, special_flags=pygame.BLEND_MULT)
    return s


def _rgb(color, scale: float = 1.0) -> tuple[int, int, int]:
    return tuple(int(max(0, min(255, c * scale))) for c in color)


def _arc(surf, color, center, r, a0, a1, width=1) -> None:
    steps = max(6, int(abs(a1 - a0) * r / 5))
    a = np.linspace(a0, a1, steps)
    pts = np.stack([center[0] + np.cos(a) * r, center[1] + np.sin(a) * r], 1).tolist()
    if width <= 1:
        pygame.draw.aalines(surf, color, False, pts)
    else:
        pygame.draw.lines(surf, color, False, pts, width)


class Orb:
    def __init__(self) -> None:
        i = np.arange(N_POINTS) + 0.5
        phi = np.arccos(1 - 2 * i / N_POINTS)
        theta = math.pi * (1 + 5 ** 0.5) * i
        self.base = np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)
        look = LOOKS[Mode.BOOT]
        self.color = np.array(look[0], float)
        self.spin, self.energy, self.turb = look[1:]
        self.level = 0.0
        self.rot = 0.0
        self.ring_phase = 0.0
        self.orbits = [[random.uniform(0, 6.28), tilt, speed, yaw]
                       for tilt, speed, yaw in ((0.45, 1.0, 0.0), (1.15, -1.35, 1.9), (2.25, 0.8, 3.6))]
        self.bars = np.zeros(N_BARS)
        self.bar_angles = np.linspace(0, 2 * math.pi, N_BARS, endpoint=False)
        self.born = time.perf_counter()
        self.R = 0
        self._tint_key = None

    def _build(self, R: int) -> None:
        self.R = R
        k = R / 250
        self.point_base = [_glow((2.5 + 6 * b / (N_BUCKETS - 1)) * k, 0.3 + 0.9 * b / (N_BUCKETS - 1),
                                 falloff=1.6, core=0.7) for b in range(N_BUCKETS)]
        self.spark_base = _glow(10 * k, 1.0, falloff=1.5, core=1.0)
        self.halo_base = _glow(R * 1.9, 0.45, falloff=2.8)
        self.core_base = _glow(R * 0.6, 1.0, falloff=1.7)
        self._tint_key = None

    def update(self, dt: float, mode: Mode, level: float) -> None:
        color, spin, energy, turb = LOOKS.get(mode, LOOKS[Mode.IDLE])
        k = 1 - math.exp(-dt * 3.5)
        self.color += (np.array(color, float) - self.color) * k
        self.spin += (spin - self.spin) * k
        self.energy += (energy - self.energy) * k
        self.turb += (turb - self.turb) * k
        attack = 1 - math.exp(-dt * (30 if level > self.level else 6))
        self.level += (level - self.level) * attack
        self.rot += self.spin * dt
        self.ring_phase += (0.3 + self.spin) * dt
        for orbit in self.orbits:
            orbit[0] += orbit[2] * (0.6 + self.spin) * dt

    def _project(self, p: np.ndarray, cx: float, cy: float):
        s = CAMERA / (CAMERA - p[:, 2])
        return cx + p[:, 0] * self.R * s, cy + p[:, 1] * self.R * s

    def _rotate(self, p: np.ndarray, t: float) -> np.ndarray:
        ay, ax, az = self.rot, 0.35 + 0.15 * math.sin(t * 0.3), self.rot * 0.25
        cy_, sy = math.cos(ay), math.sin(ay)
        cx_, sx = math.cos(ax), math.sin(ax)
        cz, sz = math.cos(az), math.sin(az)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        x, z = x * cy_ + z * sy, -x * sy + z * cy_
        y, z = y * cx_ - z * sx, y * sx + z * cx_
        x, y = x * cz - y * sz, x * sz + y * cz
        return np.stack([x, y, z], 1)

    def draw(self, surf: pygame.Surface, cx: float, cy: float, R: int, t: float) -> None:
        if R != self.R:
            self._build(R)
        intro = min(1.0, (time.perf_counter() - self.born) / 1.8)
        intro = 1 - (1 - intro) ** 3
        e = self.energy + self.level * 0.9
        color = self.color

        key = (tuple((color / 4).astype(int)), round(min(e, 1.5), 1))
        if key != self._tint_key:
            self._tint_key = key
            self.points = [_tint(s, _rgb(color)) for s in self.point_base]
            self.halo = _tint(self.halo_base, _rgb(color, 0.45 + 0.4 * min(1, e)))
            self.spark = _tint(self.spark_base, _rgb(color * 0.6 + 100))
        core_col = _rgb(color * 0.7 + 80, 0.25 + 0.75 * min(1, e * 1.2) * (0.85 + 0.15 * math.sin(t * 3)))
        core = _tint(self.core_base, core_col)

        self._draw_rings(surf, cx, cy, t)
        add = pygame.BLEND_ADD
        surf.blit(self.halo, self.halo.get_rect(center=(cx, cy)), special_flags=add)

        x, y, z = self.base.T
        n = (np.sin(x * 3.1 + t * 1.9) * np.cos(y * 2.7 - t * 1.3)
             + 0.6 * np.sin(z * 5.3 + t * 2.9 + x * 2.0)) / 1.6
        breathe = 0.02 * math.sin(t * 1.4)
        r = (1 + breathe + n * (0.03 + e * 0.24 * self.turb) + self.level * 0.06) * intro
        p = self._rotate(self.base * r[:, None], t)
        sx, sy = self._project(p, cx, cy)
        bright = np.clip((p[:, 2] + 1) / 2 * 0.85 + np.clip(n, 0, None) * 0.35 * min(1, e * 2) + 0.05, 0, 1)
        idx = (bright * (N_BUCKETS - 1)).astype(int)
        half = [s.get_width() // 2 for s in self.points]
        surf.blits([(self.points[b], (int(px) - half[b], int(py) - half[b]), None, add)
                    for b, px, py in zip(idx.tolist(), sx.tolist(), sy.tolist())], doreturn=False)

        surf.blit(core, core.get_rect(center=(cx, cy)), special_flags=add)
        self._draw_orbits(surf, cx, cy, t, intro)
        self._draw_bars(surf, cx, cy, t)

    def _draw_orbits(self, surf, cx, cy, t, intro) -> None:
        add = pygame.BLEND_ADD
        for phase, tilt, _, yaw in self.orbits:
            a = phase - np.arange(14) * 0.07
            ring = np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], 1) * 1.18 * intro
            ct, st = math.cos(tilt), math.sin(tilt)
            y, z = ring[:, 1] * ct - ring[:, 2] * st, ring[:, 1] * st + ring[:, 2] * ct
            cyw, syw = math.cos(yaw + self.rot * 0.5), math.sin(yaw + self.rot * 0.5)
            x, z = ring[:, 0] * cyw + z * syw, -ring[:, 0] * syw + z * cyw
            sx, sy = self._project(np.stack([x, y, z], 1), cx, cy)
            for j in range(len(a) - 1, 0, -1):
                s = self.points[max(0, N_BUCKETS - 1 - j)]
                surf.blit(s, (int(sx[j]) - s.get_width() // 2, int(sy[j]) - s.get_height() // 2), special_flags=add)
            surf.blit(self.spark, self.spark.get_rect(center=(int(sx[0]), int(sy[0]))), special_flags=add)

    def _draw_bars(self, surf, cx, cy, t) -> None:
        a = self.bar_angles
        shape = 0.3 + 0.7 * np.abs(np.sin(a * 5 + t * 3) * np.cos(a * 3 - t * 2.2))
        target = self.level * shape * (0.7 + 0.3 * np.random.rand(N_BARS)) + 0.012
        self.bars += (target - self.bars) * 0.35
        r0 = self.R * 1.24
        for ang, v in zip(a.tolist(), self.bars.tolist()):
            if v < 0.02:
                continue
            c, s = math.cos(ang - math.pi / 2), math.sin(ang - math.pi / 2)
            r1 = r0 + v * self.R * 0.5
            pygame.draw.line(surf, _rgb(self.color, 0.35 + 0.65 * min(1, v * 3)),
                             (cx + c * r0, cy + s * r0), (cx + c * r1, cy + s * r1), 2)

    def _draw_rings(self, surf, cx, cy, t) -> None:
        R, ph, col = self.R, self.ring_phase, self.color
        c = (cx, cy)
        for k in range(3):
            a0 = ph * 0.6 + k * 2 * math.pi / 3
            _arc(surf, _rgb(col, 0.85), c, R * 1.42, a0, a0 + 0.9, width=3)
        _arc(surf, _rgb(col, 0.22), c, R * 1.50, 0, 2 * math.pi)
        for k in range(72):
            a = -ph * 0.15 + k * math.pi / 36
            long_tick = k % 6 == 0
            r0, r1 = R * 1.50, R * (1.56 if long_tick else 1.53)
            pygame.draw.aaline(surf, _rgb(col, 0.55 if long_tick else 0.25),
                               (cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                               (cx + math.cos(a) * r1, cy + math.sin(a) * r1))
        for k in range(2):
            a0 = -ph * 1.1 + k * math.pi
            _arc(surf, _rgb(col, 0.5), c, R * 1.66, a0, a0 + 2.1)
            for a in (a0, a0 + 2.1):
                pygame.draw.circle(surf, _rgb(col, 0.9), (cx + math.cos(a) * R * 1.66, cy + math.sin(a) * R * 1.66), 3)
        for k in range(48):
            a0 = ph * 0.08 + k * math.pi / 24
            _arc(surf, _rgb(col, 0.28), c, R * 1.80, a0, a0 + 0.05)


class Hud:
    def __init__(self) -> None:
        self.fonts: dict = {}
        self._hex_lines: list[str] = []
        self._hex_at = 0.0

    def font(self, size: int, mono: bool = False) -> pygame.font.Font:
        key = (size, mono)
        if key not in self.fonts:
            names = "consolas,couriernew" if mono else "bahnschrift,segoeui,arial"
            self.fonts[key] = pygame.font.SysFont(names, size)
        return self.fonts[key]

    def spaced(self, font, text, color, spacing) -> pygame.Surface:
        glyphs = [font.render(ch, True, color) for ch in text]
        w = sum(g.get_width() for g in glyphs) + spacing * max(0, len(glyphs) - 1)
        surf = pygame.Surface((max(1, w), font.get_height()), pygame.SRCALPHA)
        x = 0
        for g in glyphs:
            surf.blit(g, (x, 0))
            x += g.get_width() + spacing
        return surf

    @staticmethod
    def wrap(font, text: str, width: int, max_lines: int) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if font.size(trial)[0] <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
            lines[0] = "… " + lines[0]
        return lines

    def draw(self, surf, state: State, color, cx, cy, R, t) -> None:
        W, H = surf.get_size()
        u = H / 860
        pad = int(28 * u)
        bright, dim, faint = _rgb(color), _rgb(color, 0.55), _rgb(color, 0.3)
        white = (225, 240, 255)

        L = int(36 * u)
        for (x, y, dx, dy) in ((pad // 2, pad // 2, 1, 1), (W - pad // 2, pad // 2, -1, 1),
                               (pad // 2, H - pad // 2, 1, -1), (W - pad // 2, H - pad // 2, -1, -1)):
            pygame.draw.lines(surf, dim, False, [(x + dx * L, y), (x, y), (x, y + dy * L)], 2)

        surf.blit(self.spaced(self.font(int(30 * u)), "J.A.R.V.I.S", bright, int(6 * u)), (pad + 8, pad))
        surf.blit(self.spaced(self.font(int(12 * u)), "JUST A RATHER VERY INTELLIGENT SYSTEM", faint, int(2 * u)),
                  (pad + 10, pad + int(38 * u)))

        now = datetime.now()
        clock = self.font(int(30 * u), mono=True).render(now.strftime("%H:%M:%S"), True, bright)
        surf.blit(clock, (W - pad - 8 - clock.get_width(), pad))
        date = self.spaced(self.font(int(12 * u)), now.strftime("%a %d %b %Y").upper(), faint, int(2 * u))
        surf.blit(date, (W - pad - 8 - date.get_width(), pad + int(38 * u)))

        mono = self.font(int(12 * u), mono=True)
        y0 = int(H * 0.36)
        for i, (label, value) in enumerate((("MIC", state.mic_level), ("VOX", state.out_level))):
            y = y0 + i * int(34 * u)
            surf.blit(mono.render(label, True, dim), (pad + 8, y))
            for k in range(14):
                lit = value * 14 > k
                pygame.draw.rect(surf, bright if lit else faint,
                                 (pad + 8 + int(36 * u) + k * int(8 * u), y + 2, int(5 * u), int(12 * u)),
                                 0 if lit else 1)
        surf.blit(mono.render(f"CORE  {state.mode.value.upper()}", True, dim), (pad + 8, y0 + int(76 * u)))

        if t - self._hex_at > 0.12:
            self._hex_at = t
            self._hex_lines = (self._hex_lines + [" ".join(f"{random.getrandbits(16):04X}" for _ in range(3))])[-9:]
        for i, line in enumerate(self._hex_lines):
            img = mono.render(line, True, _rgb(color, 0.15 + 0.35 * i / 9))
            surf.blit(img, (W - pad - 8 - img.get_width(), y0 + i * int(16 * u)))

        label = self.spaced(self.font(int(22 * u)), state.mode.value.upper(), bright, int(10 * u))
        ly = int(cy + R * 1.95)
        surf.blit(label, label.get_rect(midtop=(cx, ly)))
        status = self.font(int(14 * u)).render(state.status, True, dim)
        surf.blit(status, status.get_rect(midtop=(cx, ly + int(32 * u))))

        body = self.font(int(17 * u))
        width = int(W * 0.8)
        y = ly + int(64 * u)
        for prefix, text, col, lines in (("YOU", state.user_text, white, 2), ("JARVIS", state.reply_text, bright, 3)):
            if not text:
                continue
            tag = self.spaced(self.font(int(11 * u)), prefix, faint, int(3 * u))
            surf.blit(tag, tag.get_rect(midtop=(cx, y)))
            y += tag.get_height() + 2
            for line in self.wrap(body, text, width, lines):
                img = body.render(line, True, col)
                surf.blit(img, img.get_rect(midtop=(cx, y)))
                y += body.get_linesize()
            y += int(8 * u)

        hint = mono.render("SPACE talk   ESC stop   F11 fullscreen", True, faint)
        surf.blit(hint, hint.get_rect(bottomright=(W - pad - 8, H - pad)))


def _background(size: tuple[int, int]) -> pygame.Surface:
    W, H = size
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / W) ** 2 + ((y - H * 0.45) / H) ** 2) * 2
    v = np.clip(1 - d, 0, 1) ** 1.6
    rgb = np.stack([2 + 8 * v, 5 + 16 * v, 10 + 30 * v], -1)
    grid = ((x.astype(int) % 32 == 0) & (y.astype(int) % 32 == 0))
    rgb[grid] += (10, 22, 36)
    rgb[(y.astype(int) % 3) == 0] *= 0.85
    return pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 0, 2))


class OrbUI:
    def __init__(self, cfg, state: State, on_talk: Callable[[], None] | None = None,
                 on_interrupt: Callable[[], None] | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("J.A.R.V.I.S")
        self.cfg, self.state = cfg, state
        self.on_talk, self.on_interrupt = on_talk, on_interrupt
        self.windowed_size = cfg.window_size
        self.screen = pygame.display.set_mode(cfg.window_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.orb, self.hud = Orb(), Hud()
        self.bg = None
        self.t0 = time.perf_counter()

    def _toggle_fullscreen(self) -> None:
        if self.screen.get_flags() & pygame.FULLSCREEN:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        else:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.on_talk:
                    self.on_talk()
                elif event.key == pygame.K_ESCAPE and self.on_interrupt:
                    self.on_interrupt()
                elif event.key == pygame.K_F11:
                    self._toggle_fullscreen()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.on_talk:
                self.on_talk()

    def frame(self, dt: float) -> None:
        st = self.state
        W, H = self.screen.get_size()
        if self.bg is None or self.bg.get_size() != (W, H):
            self.bg = _background((W, H))
        t = time.perf_counter() - self.t0
        level = st.mic_level if st.mode == Mode.LISTENING else st.out_level if st.mode == Mode.SPEAKING else 0.0
        cx, cy, R = W / 2, H * 0.42, int(min(W, H) * 0.17)

        self.screen.blit(self.bg, (0, 0))
        self.orb.update(dt, st.mode, level)
        self.orb.draw(self.screen, cx, cy, R, t)
        self.hud.draw(self.screen, st, self.orb.color, cx, cy, R, t)

    def run(self) -> None:
        while self.state.running:
            dt = self.clock.tick(self.cfg.fps) / 1000
            self.handle_events()
            self.frame(dt)
            pygame.display.flip()
        pygame.quit()
