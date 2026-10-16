"""
Command-line entry point for the Keystroke Dynamics Research tool.

Run with:
    python -m keystroke_research.main
"""

from . import config, participant
from .session_runner import run_typing_session


def _prompt_participant() -> str:
    print("1) New participant")
    print("2) Existing participant")
    choice = input("Choose an option (1/2): ").strip()

    if choice == "1":
        custom_id = input(
            "Enter a participant ID (leave blank to auto-generate): "
        ).strip()
        pid = participant.create_participant(custom_id or None)
        print(f"Created participant: {pid}")
        return pid

    pid = input("Enter existing participant ID: ").strip()
    if not participant.participant_exists(pid):
        print(f"'{pid}' not found; registering it as a new participant.")
        participant.create_participant(pid)
    return pid


def _prompt_session_type() -> str:
    print("\nChoose Session Type")
    print("1. Relaxed")
    print("2. Timed Typing")
    print("3. Stress Task")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice not in config.SESSION_TYPES:
        print("Invalid choice, defaulting to Relaxed.")
        choice = "1"
    return choice


def main() -> None:
    print("Keystroke Dynamics Research Data Collector")
    print("-" * 45)

    participant_id = _prompt_participant()

    print(
        "\nFor a full dataset, consider running all three session types "
        "(Relaxed, Timed Typing, Stress Task) for this participant."
    )

    while True:
        session_type_key = _prompt_session_type()
        run_typing_session(participant_id, session_type_key)
        again = input("\nRun another session for this participant? (y/n): ").strip().lower()
        if again != "y":
            break

    print("Session data collection finished. Thank you.")


if __name__ == "__main__":
    main()