"""
ecg_features.py

Turns a raw ADC sample stream into heart-rate / HRV features indicative
of stress, the same role extract_voice_features.py plays for audio.

Deliberately built on numpy + scipy only (both already required by the
project for the voice pipeline) rather than a dedicated ECG library, so
no new heavy dependency is introduced. The R-peak detector is a
simplified Pan-Tompkins style pipeline: bandpass filter -> derivative
-> squaring -> moving-window integration -> adaptive peak picking. It's
adequate for consumer-grade single-lead ECG at 250 Hz; swap in a more
rigorous detector later if needed without changing anything downstream
(this module's public function signature is the contract).
"""

import numpy as np

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None


def _bandpass(adc, sample_rate, low=0.5, high=40.0):
    if scipy_signal is None or len(adc) < 10:
        return adc
    nyq = sample_rate / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.99)
    if low_n >= high_n:
        return adc
    b, a = scipy_signal.butter(2, [low_n, high_n], btype="band")
    try:
        return scipy_signal.filtfilt(b, a, adc)
    except Exception:
        return adc


def _detect_r_peaks(adc, sample_rate):
    """Simplified Pan-Tompkins R-peak detector. Returns peak sample indices."""
    adc = np.asarray(adc, dtype=float)
    if len(adc) < sample_rate:  # need at least ~1s of signal
        return np.array([], dtype=int)

    filtered = _bandpass(adc, sample_rate)
    derivative = np.diff(filtered, prepend=filtered[0])
    squared = derivative ** 2

    window = max(1, int(0.15 * sample_rate))  # ~150ms integration window
    kernel = np.ones(window) / window
    integrated = np.convolve(squared, kernel, mode="same")

    if scipy_signal is not None:
        min_distance = max(1, int(0.25 * sample_rate))  # refractory ~250ms (max ~240bpm)
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        peaks, _ = scipy_signal.find_peaks(integrated, distance=min_distance, height=threshold)
        return peaks

    # Fallback without scipy: naive local-maxima scan
    peaks = []
    min_distance = max(1, int(0.25 * sample_rate))
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    last_peak = -min_distance
    for i in range(1, len(integrated) - 1):
        if integrated[i] > threshold and integrated[i] >= integrated[i - 1] and integrated[i] >= integrated[i + 1]:
            if i - last_peak >= min_distance:
                peaks.append(i)
                last_peak = i
    return np.array(peaks, dtype=int)


def extract_ecg_features(adc_values, sample_rate):
    """
    adc_values: iterable of raw ADC ints/floats for one session.
    sample_rate: samples per second.

    Returns a flat dict of features. Every value defaults to 0.0 (not
    NaN) when there isn't enough signal to compute it, so downstream
    CSV writing / merging never chokes on missing columns -- callers
    that want to distinguish "no signal" from "calm signal" should
    check total_samples / data_quality alongside these.
    """
    adc = np.asarray(list(adc_values), dtype=float)
    n = len(adc)

    features = {
        "adc_mean": round(float(np.mean(adc)), 3) if n else 0.0,
        "adc_std": round(float(np.std(adc)), 3) if n else 0.0,
        "adc_min": round(float(np.min(adc)), 3) if n else 0.0,
        "adc_max": round(float(np.max(adc)), 3) if n else 0.0,
        "adc_range": round(float(np.max(adc) - np.min(adc)), 3) if n else 0.0,
        "r_peak_count": 0,
        "mean_hr_bpm": 0.0,
        "min_hr_bpm": 0.0,
        "max_hr_bpm": 0.0,
        "mean_rr_ms": 0.0,
        "sdnn_ms": 0.0,        # overall RR variability -- lower under acute stress
        "rmssd_ms": 0.0,       # short-term (parasympathetic) variability -- lower under stress
        "pnn50_pct": 0.0,      # % of successive RR diffs > 50ms -- lower under stress
        "cvnn_pct": 0.0,       # SDNN normalized by mean RR
    }

    if n < sample_rate * 2:  # need at least ~2s of signal to say anything about HR
        return features

    peaks = _detect_r_peaks(adc, sample_rate)
    features["r_peak_count"] = int(len(peaks))

    if len(peaks) < 3:
        return features

    rr_samples = np.diff(peaks)
    rr_ms = (rr_samples / sample_rate) * 1000.0

    # discard physiologically implausible RR intervals (< 300ms / >240bpm,
    # > 2000ms / <30bpm) as detector artifacts rather than real beats
    valid_mask = (rr_ms >= 300) & (rr_ms <= 2000)
    rr_ms = rr_ms[valid_mask]

    if len(rr_ms) < 2:
        return features

    hr_bpm = 60000.0 / rr_ms
    diffs = np.diff(rr_ms)

    mean_rr = float(np.mean(rr_ms))
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
    pnn50 = float(np.mean(np.abs(diffs) > 50) * 100) if len(diffs) > 0 else 0.0
    cvnn = float((sdnn / mean_rr) * 100) if mean_rr > 0 else 0.0

    features.update({
        "mean_hr_bpm": round(float(np.mean(hr_bpm)), 2),
        "min_hr_bpm": round(float(np.min(hr_bpm)), 2),
        "max_hr_bpm": round(float(np.max(hr_bpm)), 2),
        "mean_rr_ms": round(mean_rr, 2),
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2),
        "pnn50_pct": round(pnn50, 2),
        "cvnn_pct": round(cvnn, 2),
    })
    return features
