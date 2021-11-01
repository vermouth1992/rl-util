import time
from rlutils.interface.logging import LogUser


class StopWatch(LogUser):
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def lap(self):
        self.current_time = time.time()

    def log_tabular(self):
        self.lap()
        self.logger.log_tabular('Time', self.current_time - self.start_time)
