import cv2
import numpy
from numba import jit


def set_brightness(img, val, flag):
    r"""
    This function sets the brightness to an image brighter or darker depending on the flag.

    :param img: The img you want to alter
    :param val: The value by which you want to alter the image
    :param flag: The flag that indicates if you want to brighten or darken the image
    :return: The new edited image
    """

    matrix = numpy.ones(img.shape, dtype=numpy.uint8) * val

    if flag == 0:
        img_brighter = cv2.add(img, matrix)
        return img_brighter
    elif flag == 1:
        img_darker = cv2.subtract(img, matrix)
        return img_darker


def rescale(scale, frame):
    r"""
    This function can rescale an img/frame.

    :param scale: The scale of the new frame
    :param frame: The frame/img that you want to resize
    """

    w, h, _ = frame.shape
    w, h = int(w * scale), int(h * scale)
    dimensions = (w, h)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def translate(img, x, y):
    r"""
    This function translates (put img/frame in different position) img/frame.

    :param img: The img/frame you want to translate
    :param x: The new x coordinate
    :param y: The new y coordinate
    :return: The translated image
    """

    trans_mat = numpy.float32([[1, 0, x], [0, 1, y]])
    dimension = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, trans_mat, dimension)


def rotate(img, angle, rotation_point=None, rot_scale=1.0):
    r"""
    This function rotates the img/frame.

    :param img: The img you want to rotate
    :param angle: The angle for the image rotation
    :param rotation_point: The point of rotation
    :param rot_scale: The scale of the rotation
    :return: The rotated image
    """

    width, height = img.shape[:2]
    dimensions = (width, height)
    if rotation_point is None:
        rot_point = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(rot_point, angle, rot_scale)
    return cv2.warpAffine(img, rot_mat, dimensions)


@jit(nopython=True)
def process(frame, box_height=6, box_width=16):
    r"""

    """

    height, width, _ = frame.shape
    for i in range(0, height, box_height):
        for j in range(0, width, box_width):
            roi = frame[i:i + box_height, j:j + box_width]
            b_mean = numpy.mean(roi[:, :, 0])
            g_mean = numpy.mean(roi[:, :, 1])
            r_mean = numpy.mean(roi[:, :, 2])
            roi[:, :, 0] = b_mean
            roi[:, :, 1] = g_mean
            roi[:, :, 2] = r_mean
    return frame

