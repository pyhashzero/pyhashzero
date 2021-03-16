import ctypes

import os
import sys
import traceback


def is_user_admin():
    if os.name == 'nt':
        import ctypes
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except Exception as ex:
            print(str(ex))
            traceback.print_exc()
            print('Admin check failed, assuming not an admin.')
            return False
    elif os.name == 'posix':
        return os.getuid() == 0
    else:
        raise RuntimeError(f'Unsupported operation system for this module: {os.name}')


def as_admin(cmd_line=None, wait=True):
    if os.name != 'nt':
        raise RuntimeError('This function is only implemented in Windows.')

    import win32con
    import win32event
    import win32process
    from win32com.shell.shell import ShellExecuteEx
    from win32com.shell import shellcon

    python_exe = sys.executable
    if cmd_line is None:
        cmd_line = [python_exe] + sys.argv
    elif not isinstance(cmd_line, (tuple, list)):
        raise ValueError('cmd_line is not a sequence')

    cmd = '"%s"' % (cmd_line[0],)
    params = ' '.join(['"%s"' % (x,) for x in cmd_line[1:]])
    show_cmd = win32con.SW_SHOWNORMAL
    lp_verb = 'runas'

    proc_info = ShellExecuteEx(
        nShow=show_cmd,
        fMask=shellcon.SEE_MASK_NOCLOSEPROCESS,
        lpVerb=lp_verb,
        lpFile=cmd,
        lpParameters=params
    )

    if wait:
        proc_handle = proc_info['hProcess']
        _ = win32event.WaitForSingleObject(proc_handle, win32event.INFINITE)
        rc = win32process.GetExitCodeProcess(proc_handle)
    else:
        rc = None

    return rc


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
