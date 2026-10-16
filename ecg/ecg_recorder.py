"""
ecg_recorder.py

Companion instrument to the keystroke/mouse/voice stress-detection
collectors. Reads raw ADC samples streamed over serial from an ESP32
("sample_number,esp32_time_us,adc_value" lines, same protocol as the
original standalone recorder script), and exposes them the same way
mouse_tracker.py exposes mouse events:

    - start() / stop()   non-blocking, for GUI-driven or background use
    - record(duration)   blocking, for CLI parity with the original tool

If no ESP32 is attached (serial port can't be opened), this module
falls back to generating a synthetic ECG-like waveform at the same
sample rate instead of crashing. Every saved row is tagged with
`ecg_source` = "hardware" or "simulated" so downstream analysis can
tell real recordings apart from placeholders collected while testing
the pipeline without hardware. Install pyserial to use real hardware:

    pip install pyserial

Run standalone (matches the original recorder's interactive flow):
    python ecg_recorder.py

Run non-interactively (used by ecg_gui.py and run_combined_session.py):
    python ecg_recorder.py --participant P001 --condition baseline --duration 300
"""

import argparse
import csv
import math
import random
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import config

try:
    import serial  # pyserial
except ImportError:
    serial = None


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class EcgSessionRecorder:
    """Threaded ECG sample reader, mirroring MouseSessionRecorder's shape."""

    def __init__(self, port=None, baud=None, sample_rate=None):
        self.port = port or config.PORT
        self.baud = baud or config.BAUD
        self.sample_rate = sample_rate or config.SAMPLE_RATE

        self.samples = []           # list of dicts: sample_number/esp32_time_us/pc_time_sec/adc_value
        self.invalid_lines = 0
        self.sequence_gaps = 0
        self.largest_gap = 0
        self._first_sample_number = None
        self._last_sample_number = None

        self.source = None          # "hardware" or "simulated", set once start() runs
        self.t_start = None
        self.t_end = None

        self._ser = None
        self._stop_event = None
        self._thread = None
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------
    def start(self):
        """Start reading in the background. Never blocks. Call stop() later."""
        self._stop_event = threading.Event()
        self.t_start = time.perf_counter()

        opened = self._try_open_serial()
        if opened:
            self.source = "hardware"
            self._thread = threading.Thread(target=self._hardware_loop, daemon=True)
        else:
            self.source = "simulated"
            self._thread = threading.Thread(target=self._simulated_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop reading. Safe to call once."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.close()
            except Exception:
                pass
        self.t_end = time.perf_counter()

    def record(self, duration=None, verbose=True):
        """Blocking convenience wrapper: start(), wait `duration` seconds
        (or until stopped externally / KeyboardInterrupt if duration is
        None), then stop()."""
        self.start()
        if verbose:
            print(f"\n\u25cf Recording ECG ({self.source})... " +
                  (f"for {duration:.0f}s" if duration else "no time limit, Ctrl+C to stop"))
        try:
            if duration:
                end_at = self.t_start + duration
                last_print = -1
                while time.perf_counter() < end_at:
                    remaining = end_at - time.perf_counter()
                    if int(remaining) != last_print and int(remaining) % 5 == 0:
                        if verbose:
                            print(f"  ...{int(remaining):>3d}s remaining "
                                  f"({len(self.samples)} samples captured)")
                        last_print = int(remaining)
                    time.sleep(0.1)
            else:
                while True:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        if verbose:
            print(f"\u25a0 ECG recording stopped. {len(self.samples)} samples captured "
                  f"over {self.t_end - self.t_start:.1f}s ({self.source}).\n")

    # -- serial setup ----------------------------------------------------
    def _try_open_serial(self) -> bool:
        if serial is None:
            print("[ecg] pyserial not installed -- falling back to a simulated "
                  "ECG signal. Install it with `pip install pyserial` to use "
                  "real hardware.")
            return False
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
        except Exception as e:
            print(f"[ecg] Could not open serial port {self.port} ({e}) -- "
                  "falling back to a simulated ECG signal.")
            self._ser = None
            return False
        time.sleep(config.SERIAL_WARMUP_SEC)
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass
        print(f"[ecg] Serial connected: {self.port} @ {self.baud} baud")
        return True

    # -- reading loops -----------------------------------------------------
    def _hardware_loop(self):
        while not self._stop_event.is_set():
            try:
                raw = self._ser.readline()
            except Exception:
                break
            if not raw:
                continue
            try:
                text = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                self.invalid_lines += 1
                continue

            parts = text.split(",")
            if len(parts) != 3:
                self.invalid_lines += 1
                continue
            try:
                esp32_sample = int(parts[0])
                esp32_time = int(parts[1])
                adc_value = int(parts[2])
            except ValueError:
                self.invalid_lines += 1
                continue

            if not (config.ADC_MIN <= adc_value <= config.ADC_MAX):
                self.invalid_lines += 1
                continue

            self._record_sample(esp32_sample, esp32_time, adc_value)

    def _simulated_loop(self):
        """Generate a plausible synthetic ECG-like waveform (a repeating
        QRS-ish pulse train with noise) at self.sample_rate so the rest of
        the pipeline (feature extraction, fusion, model training) can be
        exercised end-to-end without an ESP32 attached."""
        interval = 1.0 / self.sample_rate
        sample_number = 0
        # A gently randomized "heart rate" so repeated simulated sessions
        # don't all look identical.
        bpm = random.uniform(65, 95)
        beat_period = 60.0 / bpm
        baseline = 2048  # mid-scale 12-bit ADC
        t0 = time.perf_counter()

        while not self._stop_event.is_set():
            t_rel = time.perf_counter() - t0
            phase = (t_rel % beat_period) / beat_period
            # crude QRS-like spike near the start of each beat, smoother
            # P/T waves elsewhere
            spike = 900 * math.exp(-((phase - 0.05) ** 2) / (2 * 0.0025))
            wave = 80 * math.sin(2 * math.pi * phase) + spike
            noise = random.gauss(0, 15)
            adc_value = int(max(config.ADC_MIN, min(config.ADC_MAX, baseline + wave + noise)))

            esp32_time_us = int(t_rel * 1_000_000)
            self._record_sample(sample_number, esp32_time_us, adc_value)
            sample_number += 1

            # keep pace with the configured sample rate
            next_t = t0 + sample_number * interval
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _record_sample(self, sample_number, esp32_time_us, adc_value):
        with self._lock:
            if self._first_sample_number is None:
                self._first_sample_number = sample_number
            if self._last_sample_number is not None:
                gap = sample_number - self._last_sample_number - 1
                if gap > 0:
                    self.sequence_gaps += gap
                    self.largest_gap = max(self.largest_gap, gap)
            self._last_sample_number = sample_number

            elapsed = time.perf_counter() - self.t_start if self.t_start else 0.0
            self.samples.append({
                "sample_number": sample_number,
                "esp32_time_us": esp32_time_us,
                "pc_time_sec": round(elapsed, 6),
                "adc_value": adc_value,
            })


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _make_session_id(participant_id):
    ts = datetime.now().strftime(config.TIMESTAMP_FORMAT)
    rand = uuid.uuid4().hex[:6]
    return f"ECG-{participant_id}_{ts}_{rand}"


def save_raw_samples(rec: EcgSessionRecorder, participant_id, session_id):
    """Write every raw ADC sample for one session to its own CSV file,
    mirroring keystroke_research.storage.save_raw_events."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"{session_id}_raw.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_Number", "ESP32_Time_us", "PC_Time_seconds", "ADC_Value"])
        for s in rec.samples:
            writer.writerow([s["sample_number"], s["esp32_time_us"], s["pc_time_sec"], s["adc_value"]])
    return str(path)


def save_feature_row(row: dict, output_path=None):
    output_path = Path(output_path or config.FEATURES_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    fieldnames = list(row.keys())

    # Self-heal like keystroke_research.storage: if the schema grew,
    # rewrite with the union of columns rather than appending under a
    # stale header.
    rows = []
    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            for r in reader:
                rows.append(r)
        fieldnames = list(dict.fromkeys(existing_fields + fieldnames))
        for r in rows:
            for fkey in fieldnames:
                r.setdefault(fkey, "unknown")

    rows.append({fkey: row.get(fkey, "unknown") for fkey in fieldnames})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(output_path)


# ---------------------------------------------------------------------------
# Programmatic entry point used by ecg_gui.py and run_combined_session.py
# ---------------------------------------------------------------------------
def run_ecg_session(participant_id, condition_key="baseline", stress_level="low",
                     duration=None, session_id=None, output_path=None,
                     port=None, baud=None, sample_rate=None, verbose=True):
    """
    Records one ECG session, extracts HRV/morphology features, and
    appends the feature row to output_path (default:
    data/ecg/ecg_stress_sessions.csv). Returns (row, raw_csv_path).
    """
    from . import ecg_features  # local import: keeps numpy/scipy optional at module import time

    session_id = session_id or _make_session_id(participant_id)
    rec = EcgSessionRecorder(port=port, baud=baud, sample_rate=sample_rate)
    rec.record(duration=duration, verbose=verbose)

    raw_path = save_raw_samples(rec, participant_id, session_id)
    adc_values = [s["adc_value"] for s in rec.samples]
    actual_duration = (rec.t_end - rec.t_start) if (rec.t_start and rec.t_end) else 0.0
    expected_samples = int((rec.sample_rate) * duration) if duration else None

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
        "session_id": session_id,
        "participant_id": participant_id,
        "session_type": condition_key,
        "stress_level": stress_level,
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
    output_path = save_feature_row(row, output_path)

    if verbose:
        print("ECG features:")
        for k, v in row.items():
            print(f"  {k:30s} {v}")
        print(f"\nRaw samples:  {raw_path}")
        print(f"Feature row appended to: {output_path}")

    return row, raw_path


# ---------------------------------------------------------------------------
# CLI (kept close to the original standalone recorder's interactive flow)
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="ECG session recorder.")
    p.add_argument("--participant", help="Participant ID")
    p.add_argument("--condition", choices=["baseline", "monkeytype", "recovery"],
                    help="Recording condition")
    p.add_argument("--duration", type=float, help="Duration in seconds (default 300)")
    p.add_argument("--session-id", help="Reuse an existing session_id to force an exact join")
    p.add_argument("--output", default=str(config.FEATURES_CSV), help="Feature CSV to append to")
    p.add_argument("--port", default=None, help=f"Serial port (default {config.PORT})")
    p.add_argument("--baud", type=int, default=None, help=f"Baud rate (default {config.BAUD})")
    return p.parse_args()


def main():
    args = _parse_args()

    print("=" * 60)
    print("             ECG DATA RECORDER")
    print("=" * 60)

    participant_id = args.participant or input("Participant ID (example: 1002): ").strip() or "unknown"

    condition_key = args.condition
    if not condition_key:
        print("\nSelect recording condition:\n")
        print("1 - Baseline (Rest)")
        print("2 - Timed Monkeytype")
        print("3 - Recovery")
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice not in config.CONDITIONS:
            print("Invalid choice.")
            sys.exit(1)
        condition_info = config.CONDITIONS[choice]
    else:
        condition_info = next(c for c in config.CONDITIONS.values() if c["key"] == condition_key)

    duration = args.duration or condition_info["default_duration_sec"]

    print()
    print("=" * 60)
    print("Recording Information")
    print("=" * 60)
    print("Participant :", participant_id)
    print("Condition   :", condition_info["label"])
    print("Duration    :", duration, "seconds")
    print("Sampling    :", config.SAMPLE_RATE, "Hz")
    print("=" * 60)
    print()
    print(condition_info["intro"])
    print()
    input("Press ENTER when ready...")

    run_ecg_session(
        participant_id=participant_id,
        condition_key=condition_info["key"],
        stress_level=condition_info["stress_level"],
        duration=duration,
        session_id=args.session_id,
        output_path=args.output,
        port=args.port,
        baud=args.baud,
    )


if __name__ == "__main__":
    main()
