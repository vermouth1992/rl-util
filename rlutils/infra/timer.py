import time


class StopWatch(object):
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def lap(self):
        self.current_time = time.time()

    def set_logger(self, logger):
        self.logger = logger

    def log_tabular(self):
        self.lap()
        self.logger.log_tabular('Time', self.current_time - self.start_time)
