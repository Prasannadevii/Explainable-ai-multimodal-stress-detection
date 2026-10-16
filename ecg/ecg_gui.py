"""
ecg_gui.py

GUI version of the ECG data collector, styled to match keystroke_gui.py,
mouse_gui.py, and voice_stress_collection.py so all four modalities feel
like one family of tools.

Run standalone:
    python ecg_gui.py

Run pre-filled (e.g. from run_combined_session.py) so this session's row
shares an ID with a keystroke/mouse/voice session collected for the same
task instance:
    python ecg_gui.py --participant P001 --session-id 20260719_101500

If no ESP32 is attached, this falls back to a simulated ECG waveform
(see ecg_recorder.py) instead of failing, so the rest of the pipeline
can still be exercised end-to-end.
"""

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg import config, ecg_features
from ecg.ecg_recorder import EcgSessionRecorder, save_raw_samples, save_feature_row

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


class EcgGUI:
    def __init__(self, root, prefill_participant=None, prefill_session=None):
        self.root = root
        self.root.title("ECG - Data Collection")
        self.root.geometry("780x700")
        self.root.configure(bg="#2c3e50")

        self.recorder = None
        self.session_active = False
        self.poll_job = None
        self.duration = None

        FG = "#ecf0f1"
        BG = "#2c3e50"
        ACCENT = "#3498db"

        tk.Label(root, text="ECG - Data Collection", font=("Helvetica", 16, "bold"),
                 bg=BG, fg=FG).pack(pady=(15, 5))

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
        condition_labels = [f"{k}. {v['label']}" for k, v in config.CONDITIONS.items()]
        self.condition_combo = ttk.Combobox(info_frame, values=condition_labels, state="readonly", width=20)
        self.condition_combo.current(0)
        self.condition_combo.grid(row=1, column=1, padx=5, pady=3, sticky="w")
        self.condition_combo.bind("<<ComboboxSelected>>", self._on_condition_change)

        tk.Label(info_frame, text="Duration (s):", bg=BG, fg=FG).grid(row=1, column=2, padx=5, pady=3, sticky="e")
        self.duration_entry = tk.Entry(info_frame, width=8)
        self.duration_entry.insert(0, str(config.CONDITIONS["1"]["default_duration_sec"]))
        self.duration_entry.grid(row=1, column=3, padx=5, pady=3, sticky="w")

        self.start_btn = tk.Button(info_frame, text="Start Recording", bg=ACCENT, fg="white",
                                    font=("Helvetica", 11, "bold"), command=self.start_session)
        self.start_btn.grid(row=2, column=1, padx=5, pady=8)

        self.stop_btn = tk.Button(info_frame, text="Stop Recording", bg="#e74c3c", fg="white",
                                   font=("Helvetica", 11, "bold"), command=self.finish_session,
                                   state="disabled")
        self.stop_btn.grid(row=2, column=2, padx=5, pady=8)

        self.status_label = tk.Label(root, text="Status: waiting to start", bg=BG, fg="#f1c40f",
                                      font=("Helvetica", 11, "bold"))
        self.status_label.pack(pady=(10, 5))

        # -- live waveform --------------------------------------------------
        plot_frame = tk.Frame(root, bg=BG)
        plot_frame.pack(pady=5, padx=20, fill="both", expand=False)

        if _HAVE_MPL:
            self.fig = Figure(figsize=(7.2, 2.6), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("ECG waveform (live)", fontsize=10)
            self.ax.set_xlabel("samples")
            self.ax.set_ylabel("ADC value")
            (self.line,) = self.ax.plot([], [])
            self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            tk.Label(plot_frame, text="(install matplotlib for a live waveform plot)",
                     bg=BG, fg=FG).pack()

        tk.Label(root, text="Once started, keep the ECG leads attached and stay still.",
                 bg=BG, fg=FG, justify="center").pack(pady=5)

        tk.Label(root, text="Session summary:", bg=BG, fg=FG).pack(pady=(10, 2))
        self.result_text = tk.Text(root, height=12, width=90, bg="#34495e", fg=FG,
                                    font=("Consolas", 9))
        self.result_text.config(state="disabled")
        self.result_text.pack(pady=5, padx=20)

    def _on_condition_change(self, event=None):
        idx = self.condition_combo.current()
        key = list(config.CONDITIONS.keys())[idx]
        self.duration_entry.delete(0, tk.END)
        self.duration_entry.insert(0, str(config.CONDITIONS[key]["default_duration_sec"]))

    # -- session lifecycle ---------------------------------------------
    def start_session(self):
        if self.session_active:
            return

        participant_id = self.participant_entry.get().strip()
        if not participant_id:
            messagebox.showerror("Missing participant ID", "Enter a participant ID first.")
            return

        idx = self.condition_combo.current()
        condition_key = list(config.CONDITIONS.keys())[idx]
        self.current_condition_info = config.CONDITIONS[condition_key]

        session_id = self.session_entry.get().strip() or (
            f"ECG-{participant_id}_{datetime.now().strftime(config.TIMESTAMP_FORMAT)}_{uuid.uuid4().hex[:6]}"
        )
        self.session_entry.delete(0, tk.END)
        self.session_entry.insert(0, session_id)

        try:
            self.duration = float(self.duration_entry.get().strip() or self.current_condition_info["default_duration_sec"])
        except ValueError:
            self.duration = self.current_condition_info["default_duration_sec"]

        self.current_participant_id = participant_id
        self.current_session_id = session_id

        self.recorder = EcgSessionRecorder()
        self.recorder.start()
        self.session_active = True

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._poll()

    def _poll(self):
        if not self.session_active:
            return

        rec = self.recorder
        n = len(rec.samples)
        import time as _time
        elapsed = _time.perf_counter() - rec.t_start if rec.t_start else 0.0
        remaining = max(self.duration - elapsed, 0) if self.duration else None

        source_note = "" if rec.source == "hardware" else "  [SIMULATED - no ESP32 detected]"
        if remaining is not None:
            self.status_label.config(
                text=f"Status: RECORDING - {remaining:.0f}s left - {n} samples captured{source_note}"
            )
        else:
            self.status_label.config(
                text=f"Status: RECORDING - {n} samples captured{source_note}"
            )

        if _HAVE_MPL and n > 0:
            window = rec.samples[-1000:]
            ys = [s["adc_value"] for s in window]
            xs = list(range(len(ys)))
            self.line.set_data(xs, ys)
            self.ax.set_xlim(0, max(len(ys), 1))
            if ys:
                pad = max(20, (max(ys) - min(ys)) * 0.2)
                self.ax.set_ylim(min(ys) - pad, max(ys) + pad)
            self.canvas.draw_idle()

        if remaining is not None and remaining <= 0:
            self.finish_session()
            return

        self.poll_job = self.root.after(150, self._poll)

    def finish_session(self):
        if not self.session_active:
            return
        self.session_active = False

        if self.poll_job is not None:
            self.root.after_cancel(self.poll_job)
            self.poll_job = None

        self.recorder.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        rec = self.recorder
        adc_values = [s["adc_value"] for s in rec.samples]
        actual_duration = (rec.t_end - rec.t_start) if (rec.t_start and rec.t_end) else 0.0
        expected_samples = int(rec.sample_rate * self.duration) if self.duration else None

        features = ecg_features.extract_ecg_features(adc_values, rec.sample_rate)

        if expected_samples and len(rec.samples) >= expected_samples * 0.98:
            quality = "GOOD"
        elif expected_samples and len(rec.samples) >= expected_samples * 0.90:
            quality = "WARNING"
        elif expected_samples:
            quality = "POOR"
        else:
            quality = "GOOD" if len(rec.samples) > rec.sample_rate * 5 else "WARNING"

        row = {
            "session_id": self.current_session_id,
            "participant_id": self.current_participant_id,
            "session_type": self.current_condition_info["key"],
            "stress_level": self.current_condition_info["stress_level"],
            "ecg_source": rec.source,
            "recording_duration_sec": round(actual_duration, 3),
            "sample_rate_hz": rec.sample_rate,
            "total_samples": len(rec.samples),
            "expected_samples": expected_samples if expected_samples is not None else "",
            "sequence_gaps": rec.sequence_gaps,
            "largest_gap": rec.largest_gap,
            "invalid_lines": rec.invalid_lines,
            "data_quality": quality,
            **features,
        }

        raw_path = save_raw_samples(rec, self.current_participant_id, self.current_session_id)
        features_path = save_feature_row(row)

        self.status_label.config(text=f"Status: session saved (session_id={self.current_session_id})")

        lines = [f"Session ID:        {self.current_session_id}",
                 f"Source:            {rec.source.upper()}", ""]
        for k, v in row.items():
            if k in ("session_id", "participant_id"):
                continue
            lines.append(f"{k:28s} {v}")
        lines.append("")
        lines.append(f"Raw samples:    {raw_path}")
        lines.append(f"Feature row appended to: {features_path}")

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", "\n".join(lines))
        self.result_text.config(state="disabled")

        self.session_entry.delete(0, tk.END)


def _parse_args():
    p = argparse.ArgumentParser(description="ECG - Data Collection GUI")
    p.add_argument("--participant", help="Pre-fill Participant ID (e.g. from a combined session)")
    p.add_argument("--session-id", help="Pre-fill Session ID so it matches your keystroke/mouse/voice session")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = tk.Tk()
    app = EcgGUI(root, prefill_participant=args.participant, prefill_session=args.session_id)
    root.mainloop()
