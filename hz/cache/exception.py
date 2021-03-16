from hz.exception import CoreException


class KeyNotFoundError(CoreException):
    pass


class KeyAlreadyExists(CoreException):
    pass
