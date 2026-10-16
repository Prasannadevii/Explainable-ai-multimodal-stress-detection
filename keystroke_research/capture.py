"""
Keystroke capture using a keyboard hook (pynput).

RESEARCH ETHICS / INFORMED CONSENT NOTE:
pynput's Listener observes key events at the OS level for as long as it
is running. This application only starts the listener immediately
before the typing task and stops it immediately after (see
session_runner.py), so timing data is only ever recorded during the
consented task window. Participants should be informed, as part of the
study protocol, that keystroke timing is captured while they type the
displayed paragraph.

Platform notes:
- Windows: works out of the box.
- macOS: the terminal app needs "Accessibility" / "Input Monitoring"
  permission granted in System Settings.
- Linux: requires an X11 session (not all Wayland setups support
  global hooks) and, on some distros, elevated permissions.
"""

import threading
import time
from typing import List, Optional

from pynput import keyboard

from .models import KeyEvent


def _normalize_key(key) -> str:
    """Convert a pynput key object into a stable, readable string name."""
    char = getattr(key, "char", None)
    if char is not None:
        return char
    # Special keys (space, backspace, enter, shift, etc.)
    return str(key).replace("Key.", "")


class KeystrokeCapture:
    """Records key press/release timestamps for a single typing session."""

    def __init__(self, session_id: str, participant_id: str):
        self.session_id = session_id
        self.participant_id = participant_id
        self.events: List[KeyEvent] = []
        self._listener: Optional[keyboard.Listener] = None
        self._lock = threading.Lock()

    def _record(self, key, event_type: str) -> None:
        ts = time.perf_counter()
        with self._lock:
            self.events.append(
                KeyEvent(
                    session_id=self.session_id,
                    participant_id=self.participant_id,
                    key=_normalize_key(key),
                    event_type=event_type,
                    timestamp=ts,
                )
            )

    def _on_press(self, key) -> None:
        self._record(key, "press")

    def _on_release(self, key) -> None:
        self._record(key, "release")

    def start(self) -> None:
        """Begin listening for key press/release events."""
        self.events = []
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self, settle_seconds: float = 0.3) -> List[KeyEvent]:
        """
        Stop listening and return the recorded events.

        A brief settle delay lets the release event for the final
        keystroke (typically Enter) be captured before the listener
        is torn down.
        """
        time.sleep(settle_seconds)
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        with self._lock:
            return list(self.events)