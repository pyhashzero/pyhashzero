from hz.exception import CoreException


class DuplicateKeyError(CoreException):
    pass


class InvalidDocumentError(CoreException):
    pass
