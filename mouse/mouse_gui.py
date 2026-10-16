"""
mouse_gui.py

GUI version of the mouse-tracking data collector, styled to match
keystroke_gui.py and voice_stress_collection.py.

Run standalone:
    python mouse_gui.py

Run pre-filled (e.g. from run_combined_session.py) so this session's
row shares an ID with a keystroke/voice session collected for the
same task instance:
    python mouse_gui.py --participant P001 --session-id 20260719_101500

Unlike the CLI version, this stays non-blocking: recording runs in the
background via pynput's global listeners while a Tkinter window shows
live status, so you can watch it update instead of watching a
terminal. It still tracks mouse activity system-wide (not just inside
this window) -- that's intentional, matching the original tool's
"work normally while it listens" design.
"""

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mouse_tracker import MouseSessionRecorder, compute_features, save_row, DEFAULT_MOUSE_CSV
from target_game import ClickTargetGame
from ecg.ecg_link import start_linked_ecg, stop_linked_ecg


class MouseGUI:
    def __init__(self, root, prefill_participant=None, prefill_session=None):
        self.root = root
        self.root.title("Mouse Stress Tracking - Data Collection")
        self.root.geometry("700x600")
        self.root.configure(bg="#2c3e50")

        self.recorder = None
        self.session_active = False
        self.poll_job = None
        self.duration = None
        self.end_at = None
        self.game = None

        FG = "#ecf0f1"
        BG = "#2c3e50"
        ACCENT = "#3498db"

        tk.Label(root, text="Mouse Stress Tracking - Data Collection",
                 font=("Helvetica", 16, "bold"), bg=BG, fg=FG).pack(pady=(15, 5))

        info_frame = tk.Frame(root, bg=BG)
        info_frame.pack(pady=5)

        tk.Label(info_frame, text="Participant ID:", bg=BG, fg=FG).grid(row=0, column=0, padx=5, pady=3, sticky="e")
        self.participant_entry = tk.Entry(info_frame, width=18)
        if prefill_participant:
            self.participant_entry.insert(0, prefill_participant)
        self.participant_entry.grid(row=0, column=1, padx=5, pady=3)

        tk.Label(info_frame, text="Session ID:", bg=BG, fg=FG).grid(row=0, column=2, padx=5, pady=3, sticky="e")
        self.session_entry = tk.Entry(info_frame, width=22)
        self.session_entry.insert(0, prefill_session or "")
        self.session_entry.grid(row=0, column=3, padx=5, pady=3)

        tk.Label(info_frame, text="Game length (s):", bg=BG, fg=FG).grid(row=1, column=0, padx=5, pady=3, sticky="e")
        self.duration_entry = tk.Entry(info_frame, width=8)
        self.duration_entry.insert(0, "60")
        self.duration_entry.grid(row=1, column=1, padx=5, pady=3, sticky="w")

        self.start_btn = tk.Button(info_frame, text="Start Recording", bg=ACCENT, fg="white",
                                    font=("Helvetica", 11, "bold"), command=self.start_session)
        self.start_btn.grid(row=2, column=0, padx=5, pady=8)

        self.stop_btn = tk.Button(info_frame, text="Stop Recording", bg="#e74c3c", fg="white",
                                   font=("Helvetica", 11, "bold"), command=self.finish_session,
                                   state="disabled")
        self.stop_btn.grid(row=2, column=1, padx=5, pady=8)

        self.status_label = tk.Label(root, text="Status: waiting to start", bg=BG, fg="#f1c40f",
                                      font=("Helvetica", 11, "bold"))
        self.status_label.pack(pady=(10, 5))

        self.instructions_label = tk.Label(
            root,
            text="Circles will appear one at a time on the board below --\n"
                 "click each one before it disappears.",
            bg=BG, fg=FG, justify="center")
        self.instructions_label.pack(pady=5)

        # Circles pop up here and must be clicked before they vanish.
        self.game_canvas = tk.Canvas(root, width=640, height=300, bg="#1b2530",
                                      highlightthickness=1, highlightbackground="#34495e")
        self.game_canvas.pack(pady=8, padx=20)
        self.game_status_label = tk.Label(root, text="", bg=BG, fg="#2ecc71",
                                           font=("Consolas", 11, "bold"))
        self.game_status_label.pack(pady=(0, 5))

        tk.Label(root, text="Session summary:", bg=BG, fg=FG).pack(pady=(15, 2))
        self.result_text = tk.Text(root, height=16, width=80, bg="#34495e", fg=FG,
                                    font=("Consolas", 10))
        self.result_text.config(state="disabled")
        self.result_text.pack(pady=5, padx=20)

    # -- session lifecycle ---------------------------------------------
    def start_session(self):
        if self.session_active:
            return

        participant_id = self.participant_entry.get().strip()
        if not participant_id:
            messagebox.showerror("Missing participant ID", "Enter a participant ID first.")
            return

        session_id = self.session_entry.get().strip() or (
            f"P-{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        self.session_entry.delete(0, tk.END)
        self.session_entry.insert(0, session_id)

        self.current_participant_id = participant_id
        self.current_session_id = session_id
        self.current_session_type = "game"

        try:
            self.duration = float(self.duration_entry.get().strip() or 60)
        except ValueError:
            self.duration = 60.0

        self.recorder = MouseSessionRecorder(pause_threshold=0.3)
        self.recorder.start()
        self.session_active = True

        # ECG is recorded for exactly the duration of this mouse-tracking
        # session -- started the instant recording starts, stopped the
        # instant it stops (see finish_session below).
        self._ecg_rec = start_linked_ecg()

        # The target game reads clicks straight off this canvas via its
        # own binding -- it doesn't touch the global pynput listener
        # above, so the usual velocity/jerk/click features are still
        # computed on top of it as normal.
        self.game = ClickTargetGame(self.game_canvas, on_finish=self._on_game_finish)
        self.game_canvas.delete("all")
        self.game_status_label.config(text="targets: 0    hits: 0    misses: 0")
        self.root.after(50, lambda: self.game.start(duration_sec=self.duration))

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self._poll()

    def _on_game_finish(self, summary):
        """Fired by ClickTargetGame itself once its timer runs out
        naturally (not on early Stop) -- just routes into the same
        finish_session() path so both endings are handled identically."""
        self._game_summary = summary
        self.finish_session()

    def _poll(self):
        if not self.session_active:
            return

        n_events = len(self.recorder.events)
        remaining = self.game.time_remaining() if self.game else 0.0
        hits = sum(1 for r in self.game.results if r["outcome"] == "hit") if self.game else 0
        misses = sum(1 for r in self.game.results if r["outcome"] == "miss") if self.game else 0
        self.status_label.config(
            text=f"Status: RECORDING - {remaining:.0f}s left - {n_events} mouse events"
        )
        self.game_status_label.config(
            text=f"targets: {hits + misses}    hits: {hits}    misses: {misses}"
        )

        self.poll_job = self.root.after(200, self._poll)

    def finish_session(self):
        if not self.session_active:
            return
        self.session_active = False

        if self.poll_job is not None:
            self.root.after_cancel(self.poll_job)
            self.poll_job = None

        game_summary = None
        if self.game is not None:
            if self.game.active:
                self.game.stop()  # manual Stop button pressed mid-game
            game_summary = getattr(self, "_game_summary", None) or self.game.summary()

        self.recorder.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        row, breakdown = compute_features(
            self.recorder, self.current_participant_id, self.current_session_type,
            session_id=self.current_session_id, game_summary=game_summary,
        )
        save_row(row, str(DEFAULT_MOUSE_CSV))

        stop_linked_ecg(getattr(self, "_ecg_rec", None), participant_id=self.current_participant_id,
                         session_id=self.current_session_id, session_type=self.current_session_type,
                         stress_level=row["stress_level"])

        self.status_label.config(text=f"Status: session saved (session_id={self.current_session_id})")

        lines = [f"Session ID:        {self.current_session_id}",
                 f"Inferred stress level: {row['stress_level'].upper()} "
                 f"(composite score {row['mouse_stress_score']})", ""]
        if game_summary:
            lines.append(f"Game: {game_summary['game_hits']} hits / "
                         f"{game_summary['game_misses']} misses "
                         f"({game_summary['game_hit_rate']:.0%} hit rate), "
                         f"{game_summary['game_stray_clicks']} stray clicks, "
                         f"avg reaction {game_summary['game_avg_reaction_time_ms']:.0f} ms")
            lines.append("")
        for k, v in row.items():
            if k in ("session_id", "participant_id"):
                continue
            lines.append(f"{k:35s} {v}")
        lines.append("")
        lines.append("Indicator breakdown (0=calm pattern, 1=highly stressed pattern):")
        for k, v in breakdown.items():
            lines.append(f"  {k:25s} {v:.2f}")
        lines.append("")
        lines.append(f"Appended to: {DEFAULT_MOUSE_CSV}")
        lines.append("ECG: recorded automatically for this session (see data/ecg/ecg_stress_sessions.csv)")

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", "\n".join(lines))
        self.result_text.config(state="disabled")

        self.session_entry.delete(0, tk.END)
        self._game_summary = None


def _parse_args():
    p = argparse.ArgumentParser(description="Mouse Stress Tracking - Data Collection GUI")
    p.add_argument("--participant", help="Pre-fill Participant ID (e.g. from a combined session)")
    p.add_argument("--session-id", help="Pre-fill Session ID so it matches your keystroke/voice session")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = tk.Tk()
    app = MouseGUI(root, prefill_participant=args.participant, prefill_session=args.session_id)
    root.mainloop()
