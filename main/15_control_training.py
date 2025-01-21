import argparse
import os
import signal
import psutil
from enum import Enum

# Constants
SCRIPT_NAME = "14_train_nn_family.py"


class SignalType(Enum):
    TERMINATE_TIME = signal.SIGTERM
    TERMINATE_EPOCH = signal.SIGINT
    STOP_NEXT_EPOCH = signal.SIGUSR2
    STOP_NEXT_TIME = signal.SIGUSR1
    SAVE_LAST_EPOCH = signal.SIGRTMIN
    SAVE_LAST_TIME = signal.SIGRTMIN + 1


def signal_catch():
    """
    Placeholder function for signal handlers in the controlled script.
    These are defined in the actual script being controlled.
    """
    handle_termination_time = lambda a, b: True
    handle_termination_epoch = lambda a, b: True
    handle_stop_time_signal = lambda a, b: True
    handle_stop_epoch_signal = lambda a, b: True
    handle_save_time_signal = lambda a, b: True
    handle_save_epoch_signal = lambda a, b: True

    signal.signal(signal.SIGTERM, handle_termination_time)
    signal.signal(signal.SIGINT, handle_termination_epoch)
    signal.signal(signal.SIGTSTP, handle_termination_time)
    signal.signal(signal.SIGUSR1, handle_stop_time_signal)
    signal.signal(signal.SIGUSR2, handle_stop_epoch_signal)
    signal.signal(signal.SIGRTMIN, handle_save_epoch_signal)
    signal.signal(signal.SIGRTMIN + 1, handle_save_time_signal)


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

    args = parser.parse_args()

    # Call main with the resolved SignalType
    ret = main(args.signal_type)
    exit(ret)
