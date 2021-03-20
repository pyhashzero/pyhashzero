class Type:
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'({self._name})'
