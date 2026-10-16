import argparse
import pyaudio
import wave
import os
import sys
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import sounddevice as sd
from scipy.io import wavfile

# Max recording length per stress level, in seconds. This is only a
# safety-net upper bound now -- actual recording is captured live via
# an InputStream and stopped the instant the user clicks Stop, so a
# level's real, spoken audio is exactly what gets saved (see
# start_recording/_start_stream/stop_recording below). Previously this
# project used sd.rec(), which pre-allocates a fixed-duration buffer
# that keeps recording in the background no matter when Stop is
# clicked -- so an early stop saved a mostly-silent buffer, and the
# leftover background recording could collide with the next prompt's
# recording on the same input device. That was the root cause of
# "my voice isn't being recorded."
MAX_RECORD_SECONDS = {1: 30, 2: 20, 3: 15}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecg.ecg_link import start_linked_ecg, stop_linked_ecg

_STRESS_LEVEL_BY_LEVEL = {1: "low", 2: "medium", 3: "high"}

class VoiceStressDataCollector:
    def __init__(self, root, prefill_participant=None, prefill_session=None):
        self._prefill_participant = prefill_participant
        self._prefill_session = prefill_session
        self.root = root
        self.root.title("Voice Stress Detection - Data Collection")
        self.root.geometry("700x600")
        self.root.configure(bg="#2c3e50")
        
        # Data storage
        self.participant_id = None
        self.session_id = None
        self.current_level = None
        self.is_recording = False
        self.audio_data = []
        self.audio_frames = []   # chunks collected live by the InputStream callback
        self._stream = None
        self._auto_stop_id = None
        self.fs = 44100  # Sample rate
        
        # Create data directory
        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        self.data_dir = PROJECT_ROOT / "data" / "voice" / "voice_stress_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Prompts for each stress level
        self.prompts = {
            1: [
                "Please read this text naturally: The quick brown fox jumps over the lazy dog.",
                "Describe your favorite hobby in a calm manner.",
                "Talk about what you did today in a relaxed tone.",
                "Share a happy memory from your past."
            ],
            2: [
                "Read this quickly: The quick brown fox jumps over the lazy dog. You have 10 seconds!",
                "Describe your hobby quickly! You have limited time.",
                "Tell me about today rapidly! Speak faster!",
                "Recall a happy memory and share it quickly!"
            ],
            3: [
                "URGENT: Read this ASAP: The quick brown fox jumps over the lazy dog. You have 5 seconds! GO!",
                "Quickly describe your hobby! Limited time! HURRY!",
                "Tell me about today RIGHT NOW! Very fast!",
                "Recall and share a memory IMMEDIATELY! Quick!"
            ]
        }
        
        self.prompt_index = {1: 0, 2: 0, 3: 0}
        
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="🎤 Voice Stress Detection",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        title_label.pack(pady=20)
        
        # Participant Info Frame
        info_frame = tk.LabelFrame(
            self.root,
            text="Participant Information",
            bg="#34495e",
            fg="#ecf0f1",
            font=("Arial", 10, "bold")
        )
        info_frame.pack(padx=20, pady=10, fill="x")
        
        # Participant ID
        tk.Label(info_frame, text="Participant ID:", bg="#34495e", fg="#ecf0f1").pack(side="left", padx=10, pady=10)
        self.participant_entry = tk.Entry(info_frame, width=15)
        if self._prefill_participant:
            self.participant_entry.insert(0, self._prefill_participant)
        self.participant_entry.pack(side="left", padx=5)
        
        tk.Label(info_frame, text="Session:", bg="#34495e", fg="#ecf0f1").pack(side="left", padx=10)
        self.session_entry = tk.Entry(info_frame, width=15)
        self.session_entry.insert(0, self._prefill_session or datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.session_entry.pack(side="left", padx=5)
        
        start_btn = tk.Button(
            info_frame,
            text="Start Session",
            command=self.start_session,
            bg="#27ae60",
            fg="white",
            padx=10
        )
        start_btn.pack(side="right", padx=10)
        
        # Stress Level Selection
        level_frame = tk.LabelFrame(
            self.root,
            text="Select Stress Level",
            bg="#34495e",
            fg="#ecf0f1",
            font=("Arial", 10, "bold")
        )
        level_frame.pack(padx=20, pady=10, fill="x")
        
        self.level_var = tk.IntVar(value=1)
        
        levels_info = [
            (1, "Normal - Read naturally"),
            (2, "Medium - Read with time pressure (10s)"),
            (3, "High - Read urgently (5s)")
        ]
        
        for level, description in levels_info:
            btn = tk.Radiobutton(
                level_frame,
                text=f"Level {level}: {description}",
                variable=self.level_var,
                value=level,
                bg="#34495e",
                fg="#ecf0f1",
                selectcolor="#27ae60",
                font=("Arial", 10)
            )
            btn.pack(anchor="w", padx=20, pady=8)
        
        # Prompt Display
        prompt_frame = tk.LabelFrame(
            self.root,
            text="Current Prompt",
            bg="#34495e",
            fg="#ecf0f1",
            font=("Arial", 10, "bold")
        )
        prompt_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        self.prompt_text = tk.Text(
            prompt_frame,
            height=6,
            bg="#2c3e50",
            fg="#ecf0f1",
            font=("Arial", 11),
            wrap="word"
        )
        self.prompt_text.pack(padx=10, pady=10, fill="both", expand=True)
        self.prompt_text.config(state="disabled")
        
        # Recording Controls
        control_frame = tk.Frame(self.root, bg="#2c3e50")
        control_frame.pack(pady=15)
        
        self.record_btn = tk.Button(
            control_frame,
            text="🔴 Start Recording",
            command=self.start_recording,
            bg="#e74c3c",
            fg="white",
            width=20,
            height=2,
            font=("Arial", 10, "bold"),
            state="disabled"
        )
        self.record_btn.pack(side="left", padx=5)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="⏹️  Stop Recording",
            command=self.stop_recording,
            bg="#95a5a6",
            fg="white",
            width=20,
            height=2,
            font=("Arial", 10, "bold"),
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="Status: Awaiting session start",
            bg="#2c3e50",
            fg="#f39c12",
            font=("Arial", 9)
        )
        self.status_label.pack(pady=5)
    
    def start_session(self):
        self.participant_id = self.participant_entry.get().strip()
        self.session_id = self.session_entry.get().strip()
        
        if not self.participant_id:
            messagebox.showerror("Error", "Please enter Participant ID")
            return
        
        # Create participant directory
        self.participant_dir = self.data_dir / self.participant_id / self.session_id
        self.participant_dir.mkdir(parents=True, exist_ok=True)
        
        self.record_btn.config(state="normal")
        self.status_label.config(text=f"Status: Session started for {self.participant_id}")
        self.participant_entry.config(state="disabled")
        self.session_entry.config(state="disabled")
        
        # Load first prompt
        self.load_prompt()
    
    def load_prompt(self):
        level = self.level_var.get()
        self.current_level = level
        
        prompt_list = self.prompts[level]
        idx = self.prompt_index[level]
        
        if idx >= len(prompt_list):
            messagebox.showinfo("Info", "All prompts for this level completed!")
            return
        
        prompt = prompt_list[idx]
        
        self.prompt_text.config(state="normal")
        self.prompt_text.delete(1.0, "end")
        
        if level == 1:
            self.prompt_text.insert("end", f"📖 Prompt (Read naturally):\n\n{prompt}")
        elif level == 2:
            self.prompt_text.insert("end", f"⏱️  Prompt (Read quickly - 10 seconds):\n\n{prompt}")
        else:
            self.prompt_text.insert("end", f"🚨 Prompt (Read URGENTLY - 5 seconds):\n\n{prompt}")
        
        self.prompt_text.config(state="disabled")
    
    def start_recording(self):
        if self.current_level != self.level_var.get():
            self.load_prompt()
        
        self.is_recording = True
        self.audio_data = []
        self.audio_frames = []
        self.record_btn.config(state="disabled", bg="#95a5a6")
        self.stop_btn.config(state="normal", bg="#e74c3c")
        
        # ECG is recorded for exactly the duration of THIS prompt recording
        # -- started the instant recording starts, stopped the instant it
        # stops (see stop_recording/save_audio below) -- so each stress
        # level's physiological reading is tied precisely to that specific
        # reading, not the whole multi-prompt session.
        self._ecg_rec = start_linked_ecg()
        
        prompt_idx = self.prompt_index[self.current_level]
        self.status_label.config(
            text=f"🔴 Recording Level {self.current_level} | Prompt {prompt_idx + 1}...",
            fg="#e74c3c"
        )

        try:
            self._stream = sd.InputStream(
                samplerate=self.fs, channels=1, dtype='float32',
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as e:
            self.is_recording = False
            self._stream = None
            messagebox.showerror("Recording Error", f"Could not open the microphone: {str(e)}")
            self.record_btn.config(state="normal", bg="#e74c3c")
            self.stop_btn.config(state="disabled", bg="#95a5a6")
            return

        # Safety-net auto-stop in case the user never clicks Stop, so a
        # session can't record forever -- but the actual audio length is
        # whatever was captured between Start and Stop (or this timeout),
        # not a padded fixed-size buffer.
        max_seconds = MAX_RECORD_SECONDS.get(self.current_level, 30)
        self._auto_stop_id = self.root.after(max_seconds * 1000, self._auto_stop)

    def _audio_callback(self, indata, frames, time_info, status):
        # Runs on sounddevice's own audio thread; just append the chunk.
        if self.is_recording:
            self.audio_frames.append(indata.copy())

    def _auto_stop(self):
        self._auto_stop_id = None
        if self.is_recording:
            self.stop_recording()

    def _stop_stream(self):
        if self._auto_stop_id is not None:
            try:
                self.root.after_cancel(self._auto_stop_id)
            except Exception:
                pass
            self._auto_stop_id = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
    
    def stop_recording(self):
        self.is_recording = False
        self.stop_btn.config(state="disabled", bg="#95a5a6")
        self._stop_stream()

        self.audio_data = (
            np.concatenate(self.audio_frames, axis=0) if self.audio_frames else np.array([])
        )
        
        if len(self.audio_data) == 0:
            stop_linked_ecg(getattr(self, "_ecg_rec", None), participant_id=self.participant_id,
                             session_id=self.session_id, session_type=f"voice_level{self.current_level}",
                             stress_level=_STRESS_LEVEL_BY_LEVEL.get(self.current_level, "unknown"),
                             save=False)
            messagebox.showerror("Error", "No audio recorded!")
            self.record_btn.config(state="normal", bg="#e74c3c")
            return
        
        # Save audio file
        self.save_audio()
    
    def save_audio(self):
        level = self.current_level
        prompt_idx = self.prompt_index[level]
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"level{level}_prompt{prompt_idx + 1}_{timestamp}.wav"
        filepath = self.participant_dir / filename

        # Stop + save the ECG recording that ran for exactly this prompt's
        # duration, tagged with a per-prompt id (distinct from the shared
        # voice session_id, since one voice session covers several prompts
        # across several levels) and the level's ground-truth stress label.
        ecg_session_id = f"{self.session_id}_L{level}P{prompt_idx + 1}_{timestamp}"
        stop_linked_ecg(getattr(self, "_ecg_rec", None), participant_id=self.participant_id,
                         session_id=ecg_session_id, session_type=f"voice_level{level}",
                         stress_level=_STRESS_LEVEL_BY_LEVEL.get(level, "unknown"))
        
        # Save audio file
        try:
            wavfile.write(str(filepath), self.fs, self.audio_data)
            
            # Save metadata
            metadata = {
                "participant_id": self.participant_id,
                "session_id": self.session_id,
                "level": level,
                "prompt_index": prompt_idx,
                "timestamp": timestamp,
                "prompt_text": self.prompts[level][prompt_idx],
                "duration": len(self.audio_data) / self.fs,
                "sample_rate": self.fs
            }
            
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update status
            self.prompt_index[level] += 1
            self.status_label.config(
                text=f"✅ Saved: {filename}",
                fg="#27ae60"
            )
            
            # Auto-load next prompt or show completion
            if self.prompt_index[level] < len(self.prompts[level]):
                self.load_prompt()
                self.record_btn.config(state="normal", bg="#e74c3c")
            else:
                messagebox.showinfo(
                    "Completed",
                    f"All prompts for Level {level} completed!\nSelect next level or end session."
                )
                self.record_btn.config(state="normal", bg="#e74c3c")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving audio: {str(e)}")
            self.record_btn.config(state="normal", bg="#e74c3c")


def _parse_args():
    p = argparse.ArgumentParser(description="Voice Stress Detection - Data Collection GUI")
    p.add_argument("--participant", help="Pre-fill Participant ID (e.g. from a combined session)")
    p.add_argument("--session-id", help="Pre-fill Session ID so it matches your keystroke/mouse session")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = tk.Tk()
    app = VoiceStressDataCollector(
        root,
        prefill_participant=args.participant,
        prefill_session=args.session_id,
    )
    root.mainloop()
