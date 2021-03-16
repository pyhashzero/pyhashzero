import time


def auto_pause(duration):
    def _wrapper(func):
        def __wrapper(*args, **kwargs):
            ret = func(args, kwargs)
            time.sleep(duration)
            return ret

        return __wrapper

    return _wrapper


def as_admin():
    pass
