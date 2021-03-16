import os
from datetime import datetime

from hz.log.core import Logger


class FileLogger(Logger):
    def __init__(self, level: str, connection: str):
        super(FileLogger, self).__init__(level)

        self.path = connection

    @staticmethod
    def _create_record(log) -> str:
        return f'Level: {log.level} - Timestamp: {datetime.utcnow().isoformat()} - Message: {log}'

    def _write(self, log):
        date, time = datetime.utcnow().isoformat().split('T')
        year, month, day = date.split('-')
        folder_path = os.path.join(year, month)
        if not os.path.exists(os.path.join(self.path, folder_path)):
            os.makedirs(os.path.join(self.path, folder_path))

        file_path = os.path.join(self.path, folder_path, f'{day}.log')
        with open(file_path, 'a') as file:
            message = FileLogger._create_record(log)
            file.write(message + '\n')
