from abc import ABC, abstractmethod


class LogUser(ABC):
    def set_logger(self, logger):
        self.logger = logger
        self.logger.register(self.log_tabular)

    @abstractmethod
    def log_tabular(self):
        pass
