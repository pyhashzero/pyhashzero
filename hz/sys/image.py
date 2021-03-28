import sys

import cv2
import numpy

from hz.sys.screen import resolution

if sys.platform != 'win32':
    raise Exception('The image module should only be used on a Windows system.')


def grab(region=None) -> numpy.ndarray:
    if region:
        left, top, right, bot = region
        width = right - left + 1
        height = bot - top + 1
    else:
        width, height = resolution()
        left = 0
        top = 0

    image = numpy.zeros((height, width, 4))

    ret = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
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

    image_copy = image.copy()
    image_copy = numpy.dstack([image_copy, numpy.ones(image_copy.shape[:2], dtype='uint8') * 255])

    overlay = numpy.zeros(image_copy.shape[:2] + (4,), dtype='uint8')
    overlay[top:bottom, left:right] = mark

    output = image_copy.copy()
    cv2.addWeighted(overlay, alpha, output, 1.0, 0.0, output)
    return cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)


def crop(image=None, region=None) -> numpy.ndarray:
    if image is None:
        image = grab()

    if region:
        left, top, right, bottom = region
    else:
        left, top, (right, bottom) = 0, 0, *image.size

    return image[top: bottom, left: right]


def show(image, title) -> None:
    cv2.imshow(title, image)
    cv2.waitKey(1)


def save(image, path) -> None:
    cv2.imwrite(path, image)


def match_template(image=None, template=None, threshold=0.8) -> list:
    if template is None:
        raise Exception

    if image is None:
        image = grab()

    res = cv2.matchTemplate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
    loc = numpy.where(res >= threshold)
    return list(zip(*loc[::-1]))
