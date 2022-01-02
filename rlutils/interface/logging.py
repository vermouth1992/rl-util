from abc import ABC


class LogUser(ABC):
    def __init__(self):
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger
        self.logger.register(self.log_tabular)

    def log_tabular(self):
        assert self.logger is not None
