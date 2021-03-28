from hz.exception import CoreException

__all__ = ['DuplicateKeyError', 'InvalidDocumentError']


class DuplicateKeyError(CoreException):
    pass


class InvalidDocumentError(CoreException):
    pass
