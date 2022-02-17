from time import time


class Timer:
    def __init__(self, auto_start=True):
        self.started_at = time() if auto_start else None

    def start(self):
        if self.started_at:
            raise RuntimeError("Timer has been already started!")
        self.started_at = time()

    def stop(self):
        if not self.started_at:
            return "Not started!"
        hours, rest = divmod(time() - self.started_at, 3600)
        minutes, seconds = divmod(rest, 60)
        self.started_at = None
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
