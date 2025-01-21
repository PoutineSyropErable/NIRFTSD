import argparse
import os
import sys
import signal
import subprocess

# Change to the real path of the script's directory
REAL_PATH = os.path.realpath(__file__)
BASE_DIR = os.path.dirname(REAL_PATH)
os.chdir(BASE_DIR)

# Constant: The filename of the process to control
FILENAME = "loop_and_stop.py"


def get_process_details(filename):
    """
    Retrieve details of processes running the specified file.

    Args:
        filename (str): The filename to search for.

    Returns:
        list: A list of tuples containing PID and process details.
    """
    try:
        result = subprocess.run(
            ["ps", "axo", "pid,cmd", "--no-headers"],
            check=True,
            text=True,
            capture_output=True,
        )
        lines = result.stdout.strip().split("\n")
        processes = [(int(line.split(maxsplit=1)[0]), line) for line in lines if filename in line]
        return processes
    except subprocess.CalledProcessError:
        print(f"No processes found running {filename}.")
        return []


def send_signal(pid, signal_type):
    """Send the specified signal to the process with the given PID."""
    try:
        os.kill(pid, signal_type)
        print(f"Signal {signal_type} sent to process {pid}.")
    except ProcessLookupError:
        print(f"No process found with PID {pid}.")
    except PermissionError:
        print(f"Permission denied to send signal {signal_type} to process {pid}.")
    except Exception as e:
        print(f"An error occurred while sending the signal to PID {pid}: {e}")


def select_process_with_fzf(processes):
    """
    Use fzf to allow the user to select a process from multiple options.

    Args:
        processes (list): A list of tuples containing PID and process details.

    Returns:
        int: The selected PID, or None if no selection was made.
    """
    try:
        process_list = "\n".join([details for _, details in processes])
        result = subprocess.run(["fzf"], input=process_list, text=True, capture_output=True, check=True)
        selected_line = result.stdout.strip()
        selected_pid = int(selected_line.split(maxsplit=1)[0])
        return selected_pid
    except subprocess.CalledProcessError:
        print("No process selected or fzf was canceled.")
        return None


def start_loop_and_stop(args):
    """
    Start loop_and_stop.py as a subprocess with the specified arguments.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    command = ["python", FILENAME]
    if args.continue_run:
        command.append("--continue_run")
    if args.terminate:
        command.append("--terminate")
    if args.stop:
        command.append("--stop")
    if args.epoch is not None:
        command.extend(["--epoch", str(args.epoch)])

    print(f"Starting {FILENAME} with command: {' '.join(command)}")
    try:
        subprocess.Popen(command)  # Run as a background process
        print(f"{FILENAME} started successfully.")
    except Exception as e:
        print(f"Error while starting {FILENAME}: {e}")


def main(stop, terminate, kill, all, start, args):
    # If --start is specified, start the process
    if start:
        start_loop_and_stop(args)
        return 0

    # Get the processes matching the filename
    processes = get_process_details(FILENAME)
    if not processes:
        print(f"No processes found running {FILENAME}.")
        return 1

    # Extract PIDs from the process details
    pids = [pid for pid, _ in processes]

    # Handle multiple processes
    if len(processes) > 1:
        if all:
            print(f"Multiple processes found:\n" + "\n".join(details for _, details in processes))
            print("Sending signal to all.")
        else:
            print(f"Multiple processes found:\n" + "\n".join(details for _, details in processes))
            print("Use --all to send the signal to all, or select one using fzf.")
            selected_pid = select_process_with_fzf(processes)
            if not selected_pid:
                return 2  # Exit if no PID is selected
            pids = [selected_pid]  # Only send the signal to the selected PID

    # Determine which signal to send
    signal_type = None
    if stop:
        signal_type = signal.SIGUSR2  # Custom stop signal
    elif terminate:
        signal_type = signal.SIGTERM  # Termination signal
    elif kill:
        signal_type = signal.SIGUSR1  # Kill signal

    # Send signal to the PIDs
    for pid in pids:
        send_signal(pid, signal_type)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control the loop_and_stop.py process.")
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start loop_and_stop.py as a background process.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Send the custom stop signal (SIGUSR2) to gracefully stop the process.",
    )
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Send the termination signal (SIGTERM) to save state and terminate the process.",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Send the kill signal (SIGUSR1) to immediately terminate the process without saving.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If multiple processes are found, send the signal to all of them. If not provided, fzf will be used to select one.",
    )
    parser.add_argument(
        "--continue_run",
        action="store_true",
        help="Continue training from a saved state (used with --start).",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Specify the epoch to start from if continuing (used with --start).",
    )
    args = parser.parse_args()

    # Validation for `--start`
    if args.start:
        # Check for invalid combinations with `--start`
        if args.kill:
            print("Error: --kill cannot be used with --start.")
            sys.exit(1)
        if args.stop and args.terminate:
            print("Error: --stop and --terminate cannot both be used with --start.")
            sys.exit(1)
    else:
        # Ensure a signal flag is provided if `--start` is not used
        if sum([args.stop, args.terminate, args.kill]) != 1:
            print("Error: You must specify exactly one of --stop, --terminate, or --kill.")
            sys.exit(1)

    # Call the main function
    ret = main(args.stop, args.terminate, args.kill, args.all, args.start, args)
    sys.exit(ret)
