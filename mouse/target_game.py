"""
target_game.py

A structured mouse task to run alongside the free-form movement
recording in mouse_tracker.py: circles appear one at a time on a
Canvas and must be clicked before they disappear. Unlike "just use
your mouse normally", this gives a fixed, repeatable task, so reaction
time, hit rate, and click precision are directly comparable across
sessions and stress conditions -- three signals passive tracking alone
doesn't give you.

Used by mouse_gui.py when session type is "game". Doesn't touch the
global pynput listener in mouse_tracker.py at all -- that keeps
recording every move/click system-wide exactly as before, so the usual
velocity/jerk/pause features are still computed on top of this.

Usage (wired up by mouse_gui.py):
    game = ClickTargetGame(canvas, on_finish=my_callback)
    game.start(duration_sec=45)
    ...
    game.stop()          # early abort
    game.summary()        # hits / misses / reaction time, any time
"""

import random
import time
from statistics import mean, pstdev


class ClickTargetGame:
    """Spawns one circular target at a time on `canvas`. A target is a
    HIT if clicked within its radius before its lifespan runs out, a
    MISS if its lifespan runs out unclicked. Clicks that land outside
    the current target are counted separately as stray clicks (they
    don't resolve the target -- it's still live)."""

    def __init__(self, canvas, on_finish=None,
                 radius=26, min_lifespan_ms=650, max_lifespan_ms=1300,
                 spawn_gap_ms=250,
                 target_color="#3a8fd9", hit_flash_color="#2ecc71",
                 miss_flash_color="#e0574a"):
        self.canvas = canvas
        self.on_finish = on_finish
        self.radius = radius
        self.min_lifespan_ms = min_lifespan_ms
        self.max_lifespan_ms = max_lifespan_ms
        self.spawn_gap_ms = spawn_gap_ms
        self.target_color = target_color
        self.hit_flash_color = hit_flash_color
        self.miss_flash_color = miss_flash_color

        self.active = False
        self.current_target = None
        self.results = []
        self.stray_clicks = 0
        self.t_start = None
        self.end_at = None

        self._deadline_job = None
        self._spawn_job = None
        self._click_binding = self.canvas.bind("<Button-1>", self._on_canvas_click)

    # -- lifecycle -----------------------------------------------------
    def start(self, duration_sec):
        self.active = True
        self.t_start = time.perf_counter()
        self.end_at = self.t_start + duration_sec
        self._spawn_next()

    def stop(self):
        """Early abort (or natural end): cancels pending timers and
        clears whatever target is on screen. Safe to call more than
        once. Does NOT fire on_finish -- call summary() yourself if you
        need the final numbers after stopping manually."""
        self.active = False
        if self._deadline_job is not None:
            self.canvas.after_cancel(self._deadline_job)
            self._deadline_job = None
        if self._spawn_job is not None:
            self.canvas.after_cancel(self._spawn_job)
            self._spawn_job = None
        if self.current_target is not None:
            self.canvas.delete(self.current_target["oval_id"])
            self.current_target = None

    def time_remaining(self) -> float:
        if not self.active or self.end_at is None:
            return 0.0
        return max(self.end_at - time.perf_counter(), 0.0)

    # -- spawning --------------------------------------------------
    def _spawn_next(self):
        if not self.active:
            return
        if time.perf_counter() >= self.end_at:
            self._finish()
            return

        w = max(int(self.canvas.winfo_width()), 200)
        h = max(int(self.canvas.winfo_height()), 200)
        pad = self.radius + 12
        x = random.randint(pad, max(pad + 1, w - pad))
        y = random.randint(pad, max(pad + 1, h - pad))

        oval_id = self.canvas.create_oval(
            x - self.radius, y - self.radius, x + self.radius, y + self.radius,
            fill=self.target_color, outline="", tags="target",
        )
        lifespan_ms = random.randint(self.min_lifespan_ms, self.max_lifespan_ms)
        self.current_target = {
            "index": len(self.results) + 1,
            "oval_id": oval_id,
            "x": x, "y": y,
            "appear_t": time.perf_counter(),
            "lifespan_ms": lifespan_ms,
        }
        self._deadline_job = self.canvas.after(lifespan_ms, self._on_timeout, oval_id)

    def _on_timeout(self, oval_id):
        t = self.current_target
        if t is None or t["oval_id"] != oval_id:
            return  # already resolved by a click this same tick

        self._flash_and_remove(oval_id, self.miss_flash_color)
        self.results.append({
            "index": t["index"], "outcome": "miss",
            "x": t["x"], "y": t["y"], "lifespan_ms": t["lifespan_ms"],
            "reaction_time_ms": None, "click_offset_px": None,
        })
        self.current_target = None
        self._deadline_job = None
        self._spawn_job = self.canvas.after(self.spawn_gap_ms, self._spawn_next)

    def _on_canvas_click(self, event):
        if not self.active or self.current_target is None:
            return
        t = self.current_target
        dist = ((event.x - t["x"]) ** 2 + (event.y - t["y"]) ** 2) ** 0.5

        if dist > self.radius:
            self.stray_clicks += 1
            return

        reaction_ms = (time.perf_counter() - t["appear_t"]) * 1000
        if self._deadline_job is not None:
            self.canvas.after_cancel(self._deadline_job)
            self._deadline_job = None
        self._flash_and_remove(t["oval_id"], self.hit_flash_color)
        self.results.append({
            "index": t["index"], "outcome": "hit",
            "x": t["x"], "y": t["y"], "lifespan_ms": t["lifespan_ms"],
            "reaction_time_ms": round(reaction_ms, 1), "click_offset_px": round(dist, 1),
        })
        self.current_target = None
        self._spawn_job = self.canvas.after(self.spawn_gap_ms, self._spawn_next)

    def _flash_and_remove(self, oval_id, color):
        """Brief color flash so a hit/miss is visible even though the
        circle disappears immediately after -- purely cosmetic, doesn't
        affect timing or scoring."""
        try:
            self.canvas.itemconfig(oval_id, fill=color)
            self.canvas.after(90, lambda: self.canvas.delete(oval_id))
        except Exception:
            self.canvas.delete(oval_id)

    def _finish(self):
        self.active = False
        if self.current_target is not None:
            # session ended mid-target -- not clicked, but also never
            # timed out, so it's unresolved rather than a miss
            self.canvas.delete(self.current_target["oval_id"])
            self.current_target = None
        if self.on_finish:
            self.on_finish(self.summary())

    # -- results ---------------------------------------------------
    def summary(self) -> dict:
        hits = [r for r in self.results if r["outcome"] == "hit"]
        misses = [r for r in self.results if r["outcome"] == "miss"]
        total = len(self.results)
        reaction_times = [r["reaction_time_ms"] for r in hits]
        offsets = [r["click_offset_px"] for r in hits]
        return {
            "game_targets_shown": total,
            "game_hits": len(hits),
            "game_misses": len(misses),
            "game_hit_rate": round(len(hits) / total, 3) if total else 0.0,
            "game_stray_clicks": self.stray_clicks,
            "game_avg_reaction_time_ms": round(mean(reaction_times), 1) if reaction_times else 0.0,
            "game_reaction_time_std_ms": round(pstdev(reaction_times), 1) if len(reaction_times) > 1 else 0.0,
            "game_avg_click_offset_px": round(mean(offsets), 1) if offsets else 0.0,
            "results": self.results,
        }
