import ctypes
import random
import sys

if sys.platform != 'win32':
    raise Exception('The speaker module should only be used on a Windows system.')


class MCI:
    def __init__(self):
        self.w32mci = ctypes.windll.winmm.mciSendStringA
        self.w32mcierror = ctypes.windll.winmm.mciGetErrorStringA

    def send(self, command):
        buffer = ctypes.c_buffer(255)
        errorcode = self.w32mci(str(command).encode(), buffer, 254, 0)
        if errorcode:
            return errorcode, self.get_error(errorcode)
        else:
            return errorcode, buffer.value

    def get_error(self, error):
        error = int(error)
        buffer = ctypes.c_buffer(255)
        self.w32mcierror(error, buffer, 254)
        return buffer.value

    def directsend(self, txt):
        err, buf = self.send(txt)
        print(f'{txt}, {err}, {buf}')
        if err != 0:
            print('Error %s for "%s": %s' % (str(err), txt, buf))
        return err, buf


class AudioClip:
    def __init__(self, filename):
        self._mci = MCI()
        self._length_ms = 0

        filename = filename.replace('/', '\\')
        self.filename = filename
        self._alias = 'mp3_%s' % str(random.random())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def open(self):
        self._mci.directsend(r'open "%s" alias %s' % (self.filename, self._alias))
        self._mci.directsend('set %s time format milliseconds' % self._alias)

        err, buff = self._mci.directsend('status %s length' % self._alias)
        self._length_ms = int(buff)

    def close(self):
        self._mci.directsend('close %s' % self._alias)

    def volume(self, level):
        self._mci.directsend('setaudio %s volume to %d' % (self._alias, level * 10))

    def play(self, start_ms=None, end_ms=None):
        start_ms = 0 if not start_ms else start_ms
        end_ms = self.milliseconds() if not end_ms else end_ms
        self._mci.directsend('play %s from %d to %d' % (self._alias, start_ms, end_ms))

    def isplaying(self):
        return self._mode() == 'playing'

    def _mode(self):
        err, buf = self._mci.directsend('status %s mode' % self._alias)
        return buf

    def pause(self):
        self._mci.directsend('pause %s' % self._alias)

    def unpause(self):
        self._mci.directsend('resume %s' % self._alias)

    def ispaused(self):
        return self._mode() == 'paused'

    def stop(self):
        self._mci.directsend('stop %s' % self._alias)
        self._mci.directsend('seek %s to start' % self._alias)

    def milliseconds(self):
        return self._length_ms
