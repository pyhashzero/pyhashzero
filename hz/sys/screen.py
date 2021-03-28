import ctypes
import sys

__all__ = ['resolution', 'cursor']

if sys.platform != 'win32':
    raise Exception('The screen module should only be used on a Windows system.')


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long)
    ]


def resolution():
    return ctypes.windll.user32.GetSystemMetrics(1), ctypes.windll.user32.GetSystemMetrics(0)


def cursor():
    _cursor = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(_cursor))
    return _cursor.y, _cursor.x
