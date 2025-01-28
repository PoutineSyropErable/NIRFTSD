import argparse
import os
import signal
import psutil
from enum import Enum

# Constants
SCRIPT_NAME = "14_train_nn_family.py"

from __TRAINING_FILE import SignalType


def __signal_catch_NEVER_CALL():
    """
    Placeholder function for signal handlers in the controlled script.
    These handlers are defined in the actual script being controlled.
    """

    # Placeholder handlers that do nothing meaningful
    handle_termination_time = lambda signum, frame: None
    handle_termination_epoch = lambda signum, frame: None
    handle_stop_time_signal = lambda signum, frame: None
    handle_stop_epoch_signal = lambda signum, frame: None
    handle_save_last_time_signal = lambda signum, frame: None
    handle_save_last_epoch_signal = lambda signum, frame: None
    handle_save_next_time_signal = lambda signum, frame: None
    handle_save_next_epoch_signal = lambda signum, frame: None

    # Register the signal handlers
    signal.signal(signal.SIGTERM, handle_termination_time)  # Terminate the process (soft kill)
    signal.signal(signal.SIGINT, handle_termination_epoch)  # Keyboard interrupt
    signal.signal(signal.SIGUSR1, handle_stop_time_signal)  # Stop after the current time iteration
    signal.signal(signal.SIGUSR2, handle_stop_epoch_signal)  # Stop after the current epoch iteration
    signal.signal(signal.SIGRTMIN, handle_save_last_time_signal)  # Save at the current time (last time)
    signal.signal(signal.SIGRTMIN + 1, handle_save_last_epoch_signal)  # Save at the current epoch (last epoch)
    signal.signal(signal.SIGRTMIN + 2, handle_save_next_time_signal)  # Save after the next time iteration
    signal.signal(signal.SIGRTMIN + 3, handle_save_next_epoch_signal)  # Save after the next epoch iteration


def send_signal(pid, signal_type):
    """
    Send a signal to the process with the given PID.

    Args:
        pid (int): Process ID of the target process.
        signal_type (SignalType): Signal type to send.
    """
    try:
        os.kill(pid, signal_type.value)
        print(f"Signal {signal_type.name} ({signal_type.value}) sent to process {pid}.")
    except ProcessLookupError:
        print(f"No process with PID {pid} found.")
    except PermissionError:
        print(f"Permission denied to send signal {signal_type.name} ({signal_type.value}) to process {pid}.")


def find_processes_by_script(script_name):
    """
    Find all processes running a specific Python script.

    Args:
        script_name (str): Name of the target Python script.
    Returns:
        list: List of PIDs of processes running the script.
    """
    pids = []
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if cmdline:
                if os.path.basename(script_name) in [os.path.basename(arg) for arg in cmdline]:
                    pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def main(signal_type):
    """
    Main function to find processes by script and send the specified signal.

    Args:
        signal_type (SignalType): Signal type to send.
    """

    pids = find_processes_by_script(SCRIPT_NAME)

    if not pids:
        print(f"No processes found for script '{SCRIPT_NAME}'.")
        return

    for pid in pids:
        send_signal(pid, signal_type)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Send signals to {SCRIPT_NAME}.")

    # Add mutually exclusive group for signals
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--terminate_time",
        action="store_const",
        const=SignalType.TERMINATE_TIME,
        dest="signal_type",
        help="Send SIGTERM to terminate the process (time iteration).",
    )
    group.add_argument(
        "--terminate_epoch",
        action="store_const",
        const=SignalType.TERMINATE_EPOCH,
        dest="signal_type",
        help="Send SIGINT to terminate the process (epoch iteration).",
    )
    group.add_argument(
        "--stop_next_epoch",
        action="store_const",
        const=SignalType.STOP_NEXT_EPOCH,
        dest="signal_type",
        help="Send SIGUSR2 to stop after the next epoch.",
    )
    group.add_argument(
        "--stop_next_time",
        action="store_const",
        const=SignalType.STOP_NEXT_TIME,
        dest="signal_type",
        help="Send SIGUSR1 to stop after the next time iteration.",
    )
    group.add_argument(
        "--save_last_epoch",
        action="store_const",
        const=SignalType.SAVE_LAST_EPOCH,
        dest="signal_type",
        help="Send SIGRTMIN to save weights for the last epoch.",
    )
    group.add_argument(
        "--save_last_time",
        action="store_const",
        const=SignalType.SAVE_LAST_TIME,
        dest="signal_type",
        help="Send SIGRTMIN+1 to save weights for the last time iteration.",
    )
    group.add_argument(
        "--save_next_time",
        action="store_const",
        const=SignalType.SAVE_NEXT_TIME,
        dest="signal_type",
        help="Send SIGRTMIN+1 to save weights for the next time iteration.",
    )
    group.add_argument(
        "--save_next_epoch",
        action="store_const",
        const=SignalType.SAVE_NEXT_EPOCH,
        dest="signal_type",
        help="Send SIGRTMIN+1 to save weights for the next time iteration.",
    )

    args = parser.parse_args()

    # Call main with the resolved SignalType
    ret = main(args.signal_type)
    exit(ret)
