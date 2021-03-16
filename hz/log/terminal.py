import json
from datetime import datetime

from hz.log.core import Logger
from hz.utility import JSONEncoder


class TerminalLogger(Logger):
    def __init__(self, level: str):
        super(TerminalLogger, self).__init__(level)

    @staticmethod
    def _create_record(log) -> str:
        return f'Level: {log.level} - Timestamp: {datetime.utcnow().isoformat()} - Message: \n{json.dumps(log.read_dict, indent=4, cls=JSONEncoder)}'

    def _write(self, log):
        message = TerminalLogger._create_record(log)
        print(message)
