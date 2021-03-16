from hz.exception import CoreException


class ModuleAttributeException(CoreException, AttributeError):
    pass
