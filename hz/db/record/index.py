from hz.serializable import Serializable


class Index(Serializable):
    def __init__(self, field_name):
        super(Index, self).__init__()

        self.field_name = field_name
