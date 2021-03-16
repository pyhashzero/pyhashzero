from enum import Enum

from hz.core import CoreObject


class LogLevel(Enum):
    NotSet = 1 << 0
    Debug = 1 << 1
    Info = 1 << 2
    Warning = 1 << 3
    Error = 1 << 4
    Critical = 1 << 5


class Logger(CoreObject):
    def __init__(self, level: str):
        super(Logger, self).__init__()

        _level_key = list(filter(lambda x: level.lower() == x.lower(), LogLevel.__dict__.keys()))
        if len(_level_key) == 1:
            _level_key = _level_key[0]
        else:
            raise ValueError(f'could not found the log level {level}')

        self._level = LogLevel[_level_key]

    @staticmethod
    def _create_record(log):
        raise NotImplementedError

    def _write(self, log):
        raise NotImplementedError

    def log(self, log, level: LogLevel = LogLevel.Debug):
        try:
            if self._level.value <= level.value:
                log.level = level.name
                self._write(log)
        except Exception as ex:
            print(str(ex))

    def debug(self, log):
        self.log(log=log, level=LogLevel.Debug)

    def info(self, log):
        self.log(log=log, level=LogLevel.Info)

    def warning(self, log):
        self.log(log=log, level=LogLevel.Warning)

    def error(self, log):
        self.log(log=log, level=LogLevel.Error)

    def critical(self, log):
        self.log(log=log, level=LogLevel.Critical)
