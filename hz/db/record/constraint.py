from hz.serializable import Serializable


class Constraint(Serializable):
    def __init__(self, field_name, name, validator=None):
        super(Constraint, self).__init__()

        self.field_name = field_name
        self.name = name
        self.validator = validator

    def check(self, obj):
        if self.validator is not None:
            if callable(self.validator):
                if self.validator(obj):
                    return True
                else:
                    raise ValueError(f'field {self.field_name} does not meet {self.field_name}-{self.name} constraint conditions')
            else:
                return True
        else:
            return True


class LengthConstraint(Constraint):
    def __init__(self, field_name, min_length=None, max_length=None):
        if min_length is None and max_length is None:
            raise ValueError('at least one of the min_length, max_length has to be provided')

        if min_length is None:
            validator = lambda x: len(x) <= max_length
        elif max_length is None:
            validator = lambda x: min_length <= len(x)
        else:
            validator = lambda x: min_length <= len(x) <= max_length

        super(LengthConstraint, self).__init__(field_name, 'length', validator)

        self.min_length = min_length
        self.max_length = max_length


class IntegerValueConstraint(Constraint):
    def __init__(self, field_name, min_value=None, max_value=None):
        if min_value is None and max_value is None:
            raise ValueError('at least one of the min_value, max_value has to be provided')

        if min_value is None:
            validator = lambda x: x <= max_value
        elif max_value is None:
            validator = lambda x: min_value <= x
        else:
            validator = lambda x: min_value <= x <= max_value

        super(IntegerValueConstraint, self).__init__(field_name, 'value', validator)

        self.min_value = min_value
        self.max_value = max_value


# unique constraint should have more than one field
class UniqueConstraint(Constraint):
    def __init__(self, field_name):
        super(UniqueConstraint, self).__init__(field_name, 'unique')


class CustomConstraint(Constraint):
    def __init__(self, field_name, name, validator=lambda x: True):
        super(CustomConstraint, self).__init__(field_name, name, validator)
