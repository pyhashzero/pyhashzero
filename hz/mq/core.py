from hz.core import CoreObject


class MQ(CoreObject):
    def __init__(self):
        super(MQ, self).__init__()

    def send(self, message):
        raise NotImplementedError
