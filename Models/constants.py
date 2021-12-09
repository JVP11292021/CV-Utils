try:
    import numpy
    import cv2 as cv
except Exception:
    print("Something went wrong with the package dependencies. ")


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (91, 91, 91)
DARK_GRAY = (61, 61, 61)
CREME_BLUE = (96, 84, 178)
CYAN = (25, 212, 187)
ORANGE = (255, 196, 0)
YELLOW = (255, 255, 0)
PINK = (255, 0, 255)
PURPLE = (210, 34, 229)
DARK_PURPLE = (128, 62, 135)
MARINE_BLUE = (57, 99, 131)
LIGHT_BLUE = (146, 182, 209)
DARK_BLUE = (26, 71, 105)
LIGHT_RED = (255, 129, 129)
DARK_RED = (138, 0, 0)
LIGHT_GREEN = (141, 252, 152)
DARK_GREEN = (4, 112, 15)
LIGHT_YELLOW = (255, 255, 144)
DARK_YELLOW = (141, 141, 0)
CYAN_GRAY = (79, 111, 114)
ROSE_RED = (94, 48, 29)
BROWN = (150, 75, 0)
LIGHT_BROWN = (200, 100, 0)
DARK_BROWN = (111, 55, 0)
HEAVENLY_CYAN = (215, 255, 242)
SKY_BLUE = (215, 227, 255)
SLIME_GREEN = (183, 255, 232)
MOLD_GREEN = (0, 104, 69)
VIOLET = (93, 0, 104)
LAVENDER = (189, 125, 196)
SEA_BLUE = (179, 173, 238)
BRIGHT_PINK = (255, 0, 128)
DEEP_SEA_BLUE = (0, 162, 255)
BLOOD_RED = (149, 52, 52)
GRASS_GREEN = (38, 206, 49)
DIRTY_GRAY = (81, 103, 83)
PURPLE_GRAY = (102, 81, 100)
DIRTY_HARRY = (132, 195, 50)
DEEP_PURPLE = (85, 45, 125)
CACTUS_GREEN = (85, 125, 45)
DIRTY_YELLOW = (175, 192, 93)

STANDARD_WIN_SIZE = (750, 520)

BLANK_FRAME = numpy.zeros((0), numpy.uint8)

ALL_CONTOURS = -1

THUMB = 4
INDEX_FINGER = 8
MIDDLE_FINGER = 12
RING_FINGER = 16
PINKY_FINGER = 20


