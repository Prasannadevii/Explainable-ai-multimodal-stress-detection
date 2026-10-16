"""
ecg_link.py

Thin helper that keystroke_gui.py, mouse_gui.py, and
voice_stress_collection.py each call directly from their OWN
start/stop lifecycle (Start Session -> Finish Session, or Start
Recording -> Stop Recording). That's what makes ECG capture automatic
no matter how a modality's GUI is launched -- standalone, or through
run_combined_session.py -- rather than only when wrapped by an outer
script: whenever a keystroke session is happening, ECG is recording;
whenever a mouse session is happening, ECG is recording; whenever a
voice prompt is being recorded, ECG is recording. That gives a real
physiological reading for the exact window each task was underway, so
you can tell whether the person was actually stressed during that
window instead of only assuming it from the assigned condition.

Usage inside a GUI's own lifecycle methods:

    from ecg.ecg_link import start_linked_ecg, stop_linked_ecg

    def start_session(self):
        ...
        self._ecg_rec = start_linked_ecg()

    def finish_session(self):
        ...
        stop_linked_ecg(
            self._ecg_rec,
            participant_id=self.current_participant_id,
            session_id=self.current_session_id,
            session_type=session_info["key"],
            stress_level=session_info["stress_level"],
        )

Never raises. If pyserial/numpy/scipy aren't installed, no ESP32 is
attached and the simulated fallback also fails for some reason, or
anything else goes wrong, this prints a warning and returns None --
a broken/missing ECG setup should never take down keystroke, mouse,
or voice data collection.
"""

import os

ENABLE = os.environ.get("ECG_AUTO_CAPTURE", "1") != "0"


def start_linked_ecg():
    """Call at the exact moment a task/recording begins. Returns an
    EcgSessionRecorder (already recording) or None if ECG capture is
    disabled/unavailable."""
    if not ENABLE:
        return None
    try:
        from .ecg_recorder import EcgSessionRecorder
        rec = EcgSessionRecorder()
        rec.start()
        print(f"[ecg] Linked ECG capture started (source={rec.source}).")
        return rec
    except Exception as e:
        print(f"[ecg] Could not start linked ECG capture, continuing without it: {e}")
        return None


def stop_linked_ecg(rec, participant_id, session_id, session_type, stress_level,
                     save=True):
    """Call at the exact moment that same task/recording ends. Stops the
    recorder, extracts HR/HRV features, and appends a row to
    data/ecg/ecg_stress_sessions.csv tagged with the SAME session_id and
    the real condition/stress_level the caller already knows -- no
    after-the-fact lookup needed. Pass save=False to discard (e.g. the
    caller's own session was discarded/had no data). Returns the saved
    feature row dict, or None if nothing was saved."""
    if rec is None:
        return None
    try:
        rec.stop()
        if not save or not rec.samples:
            if not rec.samples:
                print("[ecg] No ECG samples captured -- nothing saved.")
            return None

        from . import ecg_features
        from .ecg_recorder import save_raw_samples, save_feature_row

        adc_values = [s["adc_value"] for s in rec.samples]
        actual_duration = (rec.t_end - rec.t_start) if (rec.t_start and rec.t_end) else 0.0
        features = ecg_features.extract_ecg_features(adc_values, rec.sample_rate)
        quality = "GOOD" if len(rec.samples) > rec.sample_rate * 3 else "WARNING"

        row = {
            "session_id": session_id,
            "participant_id": participant_id,
            "session_type": session_type,
            "stress_level": stress_level,
            "ecg_source": rec.source,
            "recording_duration_sec": round(actual_duration, 3),
            "sample_rate_hz": rec.sample_rate,
            "total_samples": len(rec.samples),
            "expected_samples": "",
            "sequence_gaps": rec.sequence_gaps,
            "largest_gap": rec.largest_gap,
            "invalid_lines": rec.invalid_lines,
            "data_quality": quality,
            **features,
        }
        raw_path = save_raw_samples(rec, participant_id, session_id)
        features_path = save_feature_row(row)
        print(f"[ecg] Saved {len(rec.samples)} samples ({rec.source}, "
              f"{actual_duration:.1f}s) for {session_type}/{stress_level} "
              f"(session_id={session_id}) -> {features_path}")
        print(f"[ecg] Raw samples -> {raw_path}")
        return row
    except Exception as e:
        print(f"[ecg] Could not finalize linked ECG capture: {e}")
        return None
