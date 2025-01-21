#!/bin/bash

SCRIPT_NAME="loop_and_stop.py"

# Find the PID of the running script
PID=$(pgrep -f "$SCRIPT_NAME")

if [ -z "$PID" ]; then
  echo "Error: Script '$SCRIPT_NAME' is not running."
  exit 1
fi

# Send the SIGTERM signal
echo "Sending SIGTERM to process with PID $PID..."
kill -SIGTERM "$PID"

if [ $? -eq 0 ]; then
  echo "Signal sent successfully."
else
  echo "Failed to send signal."
fi

