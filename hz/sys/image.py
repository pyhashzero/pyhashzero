import cv2
import numpy
import win32api
import win32con
import win32gui
import win32ui
from PIL import Image


def to_numpy(image) -> numpy.ndarray:
    return numpy.asarray(image, dtype='uint8')


def to_pil(image) -> Image.Image:
    if isinstance(image, numpy.ndarray):
        return Image.fromarray(image)
    else:
        return image


def grab(region=None) -> numpy.ndarray:
    window_handle = win32gui.GetDesktopWindow()

    if region:
        left, top, right, bot = region
        width = right - left + 1
        height = bot - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    window_handle_dc = win32gui.GetWindowDC(window_handle)
    source_dc = win32ui.CreateDCFromHandle(window_handle_dc)
    memory_dc = source_dc.CreateCompatibleDC()
    bit_map = win32ui.CreateBitmap()
    bit_map.CreateCompatibleBitmap(source_dc, width, height)
    memory_dc.SelectObject(bit_map)
    memory_dc.BitBlt((0, 0), (width, height), source_dc, (left, top), win32con.SRCCOPY)

    signed_ints_array = bit_map.GetBitmapBits(True)
    image = numpy.fromstring(signed_ints_array, dtype=numpy.uint8)
    image.shape = (height, width, 4)

    source_dc.DeleteDC()
    memory_dc.DeleteDC()
    win32gui.ReleaseDC(window_handle, window_handle_dc)
    win32gui.DeleteObject(bit_map.GetHandle())

    ret = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    ret = to_numpy(ret)
    return ret


def watermark(image=None, mark=None, region=None, alpha=0.2) -> numpy.ndarray:
    if mark is None:
        raise Exception

    if image is None:
        image = grab()

    if region:
        left, top, right, bottom = region
    else:
        raise Exception

    image = to_numpy(image)
    mark = to_numpy(mark)

    image_copy = image.copy()
    image_copy = numpy.dstack([image_copy, numpy.ones(image_copy.shape[:2], dtype='uint8') * 255])

    overlay = numpy.zeros(image_copy.shape[:2] + (4,), dtype='uint8')
    overlay[top:bottom, left:right] = mark

    output = image_copy.copy()
    cv2.addWeighted(overlay, alpha, output, 1.0, 0.0, output)
    output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)

    ret = to_numpy(output)
    return ret


def crop(image=None, region=None) -> numpy.ndarray:
    if image is None:
        image = grab()
    image = to_pil(image)

    if region:
        left, top, right, bottom = region
    else:
        left, top, (right, bottom) = 0, 0, *image.size

    return to_numpy(image.crop((left, top, right, bottom)))


def show(image, title) -> None:
    image = to_numpy(image)
    cv2.imshow(title, image)
    cv2.waitKey(1)


def save(image, path) -> None:
    image = to_numpy(image)
    cv2.imwrite(path, image)


def match_template(image=None, template=None, threshold=0.8) -> list:
    if template is None:
        raise Exception

    if image is None:
        image = grab()

    image = to_numpy(image)
    template = to_numpy(template)

    res = cv2.matchTemplate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
    loc = numpy.where(res >= threshold)
    return list(zip(*loc[::-1]))
