import time
from datetime import datetime, timedelta
from __send_notification_sl import main as send_notification_to_my_phone


class CustomNotifier:
    def __init__(self):
        self.last_execution_time = datetime.min  # Initialize to a very old time

    def run(self):
        while True:
            # Get the current time
            now = datetime.now()

            # Check if at least 1 minute has passed
            if now - self.last_execution_time >= timedelta(minutes=1):
                # Update last execution time
                self.last_execution_time = now
                send_notification_to_my_phone("Machine Learning", "Training is ongoing, keep it up!")

            # Avoid CPU overuse by adding a small sleep
            time.sleep(1)  # Sleep for 1 second to prevent busy looping


# Instantiate and run the notifier
if __name__ == "__main__":
    notifier = CustomNotifier()
    notifier.run()
