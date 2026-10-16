"""
keystroke_gui.py

GUI version of the keystroke typing-task data collector, styled to
match voice_stress_collection.py so all three modalities feel like one
family of tools.

Run standalone:
    python keystroke_gui.py

Run pre-filled (e.g. from run_combined_session.py) so this session's
rows share an ID with a mouse/voice session collected for the same
task instance:
    python keystroke_gui.py --participant P001 --session-id 20260719_101500

Unlike the original CLI version (which used a global pynput keyboard
hook), this GUI captures key press/release timestamps directly from
the Tkinter text widget the participant types into. That's more
reliable for this use case -- timing only exists while the widget has
focus, and there's no OS-level Accessibility/Input Monitoring
permission to grant on macOS.
"""

import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keystroke_research import config, participant, storage
from keystroke_research.models import KeyEvent
from keystroke_research.features import extract_features
from ecg.ecg_link import start_linked_ecg, stop_linked_ecg


def _normalize_keysym(keysym: str) -> str:
    """Match capture.py's naming convention closely enough for
    features.py's logic (which only special-cases "backspace")."""
    return keysym.lower()


class KeystrokeGUI:
    def __init__(self, root, prefill_participant=None, prefill_session=None):
        self.root = root
        self.root.title("Keystroke Dynamics - Data Collection")
        self.root.geometry("760x640")
        self.root.configure(bg="#2c3e50")

        self.events = []
        self.session_active = False
        self.timer_job = None
        self.time_remaining = None
        self.current_session_info = None
        self.current_session_id = None

        FG = "#ecf0f1"
        BG = "#2c3e50"
        ACCENT = "#3498db"

        header = tk.Label(root, text="Keystroke Dynamics - Data Collection",
                           font=("Helvetica", 16, "bold"), bg=BG, fg=FG)
        header.pack(pady=(15, 5))

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

        tk.Label(info_frame, text="Condition:", bg=BG, fg=FG).grid(row=1, column=0, padx=5, pady=3, sticky="e")
        self.condition_var = tk.StringVar(value="1")
        condition_labels = [f"{k}. {v['label']}" for k, v in config.SESSION_TYPES.items()]
        self.condition_combo = ttk.Combobox(info_frame, values=condition_labels, state="readonly", width=20)
        self.condition_combo.current(0)
        self.condition_combo.grid(row=1, column=1, columnspan=2, padx=5, pady=3, sticky="w")

        self.start_btn = tk.Button(info_frame, text="Start Session", bg=ACCENT, fg="white",
                                    font=("Helvetica", 11, "bold"), command=self.start_session)
        self.start_btn.grid(row=1, column=3, padx=5, pady=3)

        self.status_label = tk.Label(root, text="Status: waiting to start", bg=BG, fg="#f1c40f",
                                      font=("Helvetica", 11, "bold"))
        self.status_label.pack(pady=(10, 5))

        tk.Label(root, text="Type the paragraph below exactly as shown:",
                 bg=BG, fg=FG, font=("Helvetica", 10, "italic")).pack(pady=(10, 2))

        self.prompt_text = tk.Text(root, height=5, width=80, wrap="word",
                                    bg="#34495e", fg=FG, font=("Helvetica", 11))
        self.prompt_text.insert("1.0", config.SAMPLE_PARAGRAPH)
        self.prompt_text.config(state="disabled")
        self.prompt_text.pack(pady=5, padx=20)

        tk.Label(root, text="Type here:", bg=BG, fg=FG).pack(pady=(10, 2))
        self.typing_widget = tk.Text(root, height=5, width=80, wrap="word",
                                      bg="white", fg="black", font=("Helvetica", 12))
        self.typing_widget.config(state="disabled")
        self.typing_widget.pack(pady=5, padx=20)
        self.typing_widget.bind("<KeyPress>", self._on_key_press)
        self.typing_widget.bind("<KeyRelease>", self._on_key_release)
        # Enter/Return finishes the session instead of inserting a newline
        self.typing_widget.bind("<Return>", self._on_return_pressed)

        self.finish_btn = tk.Button(root, text="Finish Session", bg="#e74c3c", fg="white",
                                     font=("Helvetica", 11, "bold"), command=self.finish_session,
                                     state="disabled")
        self.finish_btn.pack(pady=8)

        tk.Label(root, text="Session summary:", bg=BG, fg=FG).pack(pady=(10, 2))
        self.result_text = tk.Text(root, height=8, width=80, bg="#34495e", fg=FG,
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
        if not participant.participant_exists(participant_id):
            participant.create_participant(participant_id)

        condition_idx = self.condition_combo.current()
        condition_key = list(config.SESSION_TYPES.keys())[condition_idx]
        session_info = config.SESSION_TYPES[condition_key]

        session_id = self.session_entry.get().strip() or (
            f"{participant_id}_{datetime.now().strftime(config.TIMESTAMP_FORMAT)}_{uuid.uuid4().hex[:6]}"
        )
        self.session_entry.delete(0, tk.END)
        self.session_entry.insert(0, session_id)

        self.current_participant_id = participant_id
        self.current_session_id = session_id
        self.current_session_info = session_info
        self.events = []

        self.typing_widget.config(state="normal")
        self.typing_widget.delete("1.0", tk.END)
        self.typing_widget.focus_set()

        self.start_btn.config(state="disabled")
        self.finish_btn.config(state="normal")
        self.session_active = True

        # ECG is recorded for exactly the duration of this typing task --
        # started the instant the session starts, stopped the instant it
        # finishes (see finish_session below) -- so heart-rate/HRV can be
        # read directly against this specific condition.
        self._ecg_rec = start_linked_ecg()

        time_limit = session_info["time_limit_sec"]
        if time_limit:
            self.time_remaining = time_limit
            self.status_label.config(
                text=f"Status: RECORDING ({session_info['label']}) - {self.time_remaining}s left"
            )
            self.timer_job = self.root.after(1000, self._tick)
        else:
            self.status_label.config(text=f"Status: RECORDING ({session_info['label']}) - no time limit")

    def _tick(self):
        self.time_remaining -= 1
        if self.time_remaining <= 0:
            self.status_label.config(text="Status: TIME'S UP - finishing session...")
            self.finish_session()
            return
        self.status_label.config(
            text=f"Status: RECORDING ({self.current_session_info['label']}) - {self.time_remaining}s left"
        )
        self.timer_job = self.root.after(1000, self._tick)

    def _on_key_press(self, event):
        if not self.session_active:
            return
        self.events.append(KeyEvent(
            session_id=self.current_session_id,
            participant_id=self.current_participant_id,
            key=_normalize_keysym(event.keysym),
            event_type="press",
            timestamp=time.perf_counter(),
        ))

    def _on_key_release(self, event):
        if not self.session_active:
            return
        self.events.append(KeyEvent(
            session_id=self.current_session_id,
            participant_id=self.current_participant_id,
            key=_normalize_keysym(event.keysym),
            event_type="release",
            timestamp=time.perf_counter(),
        ))

    def _on_return_pressed(self, event):
        self.finish_session()
        return "break"  # prevent inserting a newline into the text widget

    def finish_session(self):
        if not self.session_active:
            return
        self.session_active = False

        if self.timer_job is not None:
            self.root.after_cancel(self.timer_job)
            self.timer_job = None

        typed_text = self.typing_widget.get("1.0", "end-1c")
        self.typing_widget.config(state="disabled")
        self.start_btn.config(state="normal")
        self.finish_btn.config(state="disabled")

        if not self.events:
            self.status_label.config(text="Status: no keystrokes recorded - session discarded")
            stop_linked_ecg(getattr(self, "_ecg_rec", None), participant_id=self.current_participant_id,
                             session_id=self.current_session_id,
                             session_type=self.current_session_info["key"],
                             stress_level=self.current_session_info["stress_level"], save=False)
            messagebox.showwarning("No data", "No keystrokes were recorded; session discarded.")
            return

        session_info = self.current_session_info
        stop_linked_ecg(getattr(self, "_ecg_rec", None), participant_id=self.current_participant_id,
                         session_id=self.current_session_id,
                         session_type=session_info["key"], stress_level=session_info["stress_level"])
        raw_path = storage.save_raw_events(self.events)
        features = extract_features(
            session_id=self.current_session_id,
            participant_id=self.current_participant_id,
            events=self.events,
            typed_text=typed_text,
            session_type=session_info["key"],
            stress_level=session_info["stress_level"],
        )
        features_path = storage.save_features(features)

        self.status_label.config(text=f"Status: session saved (session_id={self.current_session_id})")

        summary = (
            f"Session type:      {session_info['label']} (stress: {session_info['stress_level']})\n"
            f"Session ID:        {self.current_session_id}\n"
            f"Total typing time: {features.total_typing_time_sec:.2f} s\n"
            f"Words per minute:  {features.words_per_minute:.2f}\n"
            f"Backspaces:        {features.backspace_count}\n"
            f"Avg dwell time:    {features.avg_dwell_time_ms:.2f} ms\n"
            f"Avg flight time:   {features.avg_flight_time_ms:.2f} ms\n"
            f"Pauses (>{config.PAUSE_THRESHOLD_SECONDS:.1f}s): {features.pause_count}\n"
            f"\nRaw keystrokes:    {raw_path}\n"
            f"Features file:     {features_path}\n"
            f"ECG:               recorded automatically for this session "
            f"(see data/ecg/ecg_stress_sessions.csv)\n"
        )
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", summary)
        self.result_text.config(state="disabled")

        # Fresh session_id next round so repeated runs don't collide
        self.session_entry.delete(0, tk.END)


def _parse_args():
    p = argparse.ArgumentParser(description="Keystroke Dynamics - Data Collection GUI")
    p.add_argument("--participant", help="Pre-fill Participant ID (e.g. from a combined session)")
    p.add_argument("--session-id", help="Pre-fill Session ID so it matches your mouse/voice session")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = tk.Tk()
    app = KeystrokeGUI(root, prefill_participant=args.participant, prefill_session=args.session_id)
    root.mainloop()
