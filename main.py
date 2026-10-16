"""
main.py

GUI control panel for the whole pipeline -- collect sessions, merge,
train, predict, and generate the consolidated report, all from one
window. Nothing here talks to the terminal: every status line, and
every print() that merge_datasets.py / train_tabnet_model.py /
decision_fusion.py / predict.py / consolidated_report.py already make,
gets redirected into the log panel in this window instead of stdout.

Run:
    python main.py
"""

import io
import os
import queue
import subprocess
import sys
import threading
import uuid
import warnings
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

# Library warnings (pytorch_tabnet's "no early stopping", "device used:
# cpu", sklearn convergence notices, etc.) are noise for this GUI's log
# panel -- they're not something the user can act on, and they used to
# get redirected into the log alongside real status lines. Silence them
# everywhere in this process...
warnings.filterwarnings("ignore")
# ...and in every subprocess this GUI launches (the keystroke/mouse/
# voice/ECG collection windows), via the same mechanism Python itself
# reads on startup.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "fusion"))
sys.path.insert(0, str(BASE_DIR / "voice"))

KEYSTROKE_SCRIPT = BASE_DIR / "keystroke_research" / "keystroke_gui.py"
MOUSE_SCRIPT = BASE_DIR / "mouse" / "mouse_gui.py"
VOICE_SCRIPT = BASE_DIR / "voice" / "voice_stress_collection.py"
ECG_SCRIPT = BASE_DIR / "ecg" / "ecg_gui.py"

FG, BG, PANEL, ACCENT = "#e7ede8", "#1b2226", "#222b2f", "#3a8fd9"


def _make_session_id(participant_id: str) -> str:
    return f"{participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


class _QueueWriter(io.TextIOBase):
    """File-like object that pushes writes onto a queue instead of stdout,
    so print() calls inside the fusion scripts land in the GUI's log
    panel instead of the terminal they were launched from."""

    def __init__(self, q: "queue.Queue"):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(s)
        return len(s)

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Stress Fusion -- Control Panel")
        self.root.geometry("880x680")
        self.root.configure(bg=BG)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.busy = False
        self.participant_id = tk.StringVar()

        tk.Label(root, text="Stress Fusion -- Control Panel", font=("Helvetica", 16, "bold"),
                 bg=BG, fg=FG).pack(pady=(14, 4))

        top = tk.Frame(root, bg=BG)
        top.pack(pady=4)
        tk.Label(top, text="Participant ID:", bg=BG, fg=FG).pack(side="left", padx=(0, 6))
        tk.Entry(top, textvariable=self.participant_id, width=22).pack(side="left")

        # -- collect a session -------------------------------------------
        collect = tk.LabelFrame(root, text="Collect a session", bg=PANEL, fg=FG,
                                 padx=12, pady=10, font=("Helvetica", 10, "bold"))
        collect.pack(fill="x", padx=16, pady=(12, 6))

        self.ks_var = tk.BooleanVar(value=True)
        self.mouse_var = tk.BooleanVar(value=True)
        self.voice_var = tk.BooleanVar(value=True)
        check_row = tk.Frame(collect, bg=PANEL)
        check_row.pack(anchor="w")
        for text, var in [("Keystroke", self.ks_var), ("Mouse", self.mouse_var), ("Voice", self.voice_var)]:
            tk.Checkbutton(check_row, text=text, variable=var, bg=PANEL, fg=FG,
                           selectcolor=PANEL, activebackground=PANEL, activeforeground=FG
                           ).pack(side="left", padx=4)

        tk.Label(collect, text="ECG tags along automatically with whichever of these you run.",
                 bg=PANEL, fg="#8a9a96", font=("Helvetica", 9)).pack(anchor="w", pady=(4, 8))

        btn_row = tk.Frame(collect, bg=PANEL)
        btn_row.pack(anchor="w")
        self.btn_run_session = tk.Button(btn_row, text="Run selected", bg=ACCENT, fg="white",
                                          activebackground=ACCENT, command=self.run_session)
        self.btn_run_session.pack(side="left", padx=(0, 8))
        self.btn_ecg_standalone = tk.Button(btn_row, text="ECG on its own (no other modality)",
                                             bg="#5c6c69", fg="white", command=self.run_ecg_standalone)
        self.btn_ecg_standalone.pack(side="left")

        # -- fusion & modeling ---------------------------------------------
        model = tk.LabelFrame(root, text="Fusion & modeling", bg=PANEL, fg=FG,
                               padx=12, pady=10, font=("Helvetica", 10, "bold"))
        model.pack(fill="x", padx=16, pady=6)

        mrow = tk.Frame(model, bg=PANEL)
        mrow.pack(anchor="w")
        self.model_buttons = []
        for text, cmd in [
            ("Rebuild merged dataset", self.run_merge),
            ("Train feature fusion (TabNet)", self.run_train_tabnet),
            ("Train decision fusion", self.run_train_decision),
            ("Predict latest session", self.run_predict_latest),
            ("Full report on latest", self.run_consolidated_report),
        ]:
            b = tk.Button(mrow, text=text, bg="#2f3b3f", fg=FG, activebackground="#2f3b3f",
                          activeforeground=FG, command=cmd)
            b.pack(side="left", padx=4, pady=2)
            self.model_buttons.append(b)

        # -- status + log -----------------------------------------------------
        self.status_label = tk.Label(root, text="ready", bg=BG, fg="#f1c40f", font=("Helvetica", 10, "bold"))
        self.status_label.pack(pady=(10, 2))

        log_frame = tk.Frame(root, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(4, 14))
        self.log_text = tk.Text(log_frame, bg="#10161a", fg="#c7d1cd", font=("Consolas", 10),
                                 wrap="word", state="disabled")
        scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._log("ready -- pick an action above. everything that used to print "
                   "to the terminal shows up here instead.")
        self._poll_log()

    # -- logging plumbing --------------------------------------------------
    def _poll_log(self):
        try:
            while True:
                chunk = self.log_queue.get_nowait()
                self.log_text.config(state="normal")
                self.log_text.insert("end", chunk)
                self.log_text.see("end")
                self.log_text.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log)

    def _log(self, line: str):
        self.log_queue.put(line.rstrip("\n") + "\n")

    def _all_buttons(self):
        return [self.btn_run_session, self.btn_ecg_standalone] + self.model_buttons

    def _set_busy(self, busy: bool, label: str = ""):
        self.busy = busy
        for b in self._all_buttons():
            b.config(state="disabled" if busy else "normal")
        self.status_label.config(text=(f"working: {label} ..." if busy else "ready"))

    def _run_in_thread(self, label: str, fn, *args):
        """Runs fn(*args) on a background thread so the window stays
        responsive, with its stdout/stderr (i.e. every print() it makes)
        redirected into the log panel instead of the terminal."""
        if self.busy:
            return
        self._set_busy(True, label)

        def worker():
            writer = _QueueWriter(self.log_queue)
            try:
                with redirect_stdout(writer), redirect_stderr(writer):
                    fn(*args)
            except Exception as e:
                self.log_queue.put(f"\n[error] {type(e).__name__}: {e}\n")
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _get_participant(self) -> str:
        pid = self.participant_id.get().strip()
        if not pid:
            pid = "unknown"
            self.participant_id.set(pid)
        return pid

    # -- actions -------------------------------------------------------
    def run_session(self):
        picks = []
        if self.ks_var.get():
            picks.append(("Keystroke", KEYSTROKE_SCRIPT))
        if self.mouse_var.get():
            picks.append(("Mouse", MOUSE_SCRIPT))
        if self.voice_var.get():
            picks.append(("Voice", VOICE_SCRIPT))
        if not picks:
            messagebox.showinfo("Nothing selected", "Check at least one of keystroke/mouse/voice.")
            return

        pid = self._get_participant()
        session_id = _make_session_id(pid)

        def job():
            self._log(f"session id: {session_id}")
            for label, script in picks:
                self._log(f"opening {label.lower()} -- close its window when you're done with it")
                self._log(f"[ecg] recording alongside {label.lower()} for this session's duration")
                subprocess.run([sys.executable, str(script), "--participant", pid, "--session-id", session_id])
            self._log(f"done -- {session_id}")

        self._run_in_thread("running session", job)

    def run_ecg_standalone(self):
        pid = self._get_participant()
        session_id = _make_session_id(pid)

        def job():
            self._log("opening ecg -- close its window when you're done with it")
            subprocess.run([sys.executable, str(ECG_SCRIPT), "--participant", pid, "--session-id", session_id])

        self._run_in_thread("running ecg", job)

    def run_merge(self):
        import merge_datasets  # type: ignore
        import extract_voice_features  # type: ignore

        def job():
            # Recorded .wav files sit under data/voice/voice_stress_data/
            # until something turns them into voice_features_dataset.csv --
            # that conversion never used to run anywhere in this GUI, which
            # is why merge always reported "voice file not found" even
            # after successful voice sessions. Do it here, every time,
            # before merging, so any newly recorded audio is picked up.
            voice_data_dir = merge_datasets.BASE_DIR / "data" / "voice" / "voice_stress_data"
            voice_out_csv = merge_datasets.BASE_DIR / "data" / "voice" / "voice_features_dataset.csv"
            if voice_data_dir.exists() and any(voice_data_dir.glob("*/*/*.wav")):
                print("[voice] extracting acoustic features from recorded sessions...")
                extractor = extract_voice_features.VoiceFeatureExtractor()
                extractor.process_dataset(data_dir=str(voice_data_dir), output_csv=str(voice_out_csv))
            else:
                print("[voice] no recorded .wav files found yet -- skipping feature extraction.")

            merge_datasets.merge(merge_datasets.BASE_DIR / "fusion" / "merged_dataset.csv")

        self._run_in_thread("rebuilding merged dataset", job)

    def run_train_tabnet(self):
        import train_tabnet_model  # type: ignore
        self._run_in_thread(
            "training tabnet",
            train_tabnet_model.train, train_tabnet_model.MERGED_CSV, train_tabnet_model.ARTIFACTS_DIR,
        )

    def run_train_decision(self):
        import decision_fusion  # type: ignore
        self._run_in_thread(
            "training decision fusion",
            decision_fusion.train, decision_fusion.MERGED_CSV, decision_fusion.ARTIFACTS_DIR,
        )

    def run_predict_latest(self):
        import predict  # type: ignore
        self._run_in_thread("predicting latest session", predict.predict_latest, predict.TABNET_ARTIFACTS_DIR)

    def run_consolidated_report(self):
        import consolidated_report  # type: ignore
        import predict  # type: ignore

        def job():
            row = predict.get_latest_row()
            report_text = consolidated_report.generate_report(row)
            print("\n" + report_text)
            saved_path = consolidated_report.save_report(report_text, row.get("session_id", "session"))
            print(f"\nsaved -> {saved_path}")

        self._run_in_thread("generating consolidated report", job)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
