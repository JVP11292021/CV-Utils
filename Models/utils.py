import os
import time
import face_recognition as fr
import cv2
import numpy
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image


class RecButton:
    r"""
    This class is used to draw a button on the video loop of opencv. This class was created to make the drawing
    of buttons easier and more accessible to newer developers. This button creates a rectangular button.
    """
    def __init__(self, text, pos, size=(85, 85), fg=(255, 255, 255), bg=(0, 0, 0), text_thickness=1):
        r"""
        In the constructor the values that are passed in are initialized.

        :param text: The text that will be put on the button
        :param pos: The top left position of the rectangle
        :param size: The top right position of the rectangle
        :param text_pos: The position of the text
        :param fg: The color of the text
        :param bg: The background color of the button
        :param text_thickness: The thickness of the text
        """

        self.text = text
        self.pos = pos
        self.size = size
        self.text_thickness = text_thickness
        self.fg = fg
        self.bg = bg

    def draw(self, frame, bg=(0, 0, 0), c_x=0, c_y=0):
        r"""
        This function is used to draw the button on the frame. The button is rectangular if the values are
        passed in correctly.

        The positions for the rectangle
        pos
        x1,y1 ------
        |          |
        |          |
        |          |
        --------x2,y2 size

        :param frame: Takes in the frame on which you want to draw the button
        :param c_x: Shift the text on the x-axis
        :param c_y: Shift the text on the y-axis
        :return: The frame with the button drawn on it
        """

        x, y = self.pos
        w, h = self.size
        self.bg = bg

        cv2.rectangle(frame, (x, y), (x + w, y + h), self.bg, cv2.FILLED)
        cv2.putText(frame, self.text, (x + (w // 2) + c_x, y + (h // 2) + c_y), cv2.FONT_HERSHEY_PLAIN, 2, self.fg, self.text_thickness)

        return frame

    def highlight(self, frame, bg_highlight, fg_highlight=(255, 255, 255), c_x=0, c_y=0):
        r"""
        Highlights the rectangle by drawing a new one on top of it with other colors. The other rectangle
        will be deleted

        :param frame: Takes in the frame on which you want to draw the button
        :param bg_highlight: The new background color
        :param fg_highlight: The new foreground color
        :param c_x: Shift the text on the x-axis
        :param c_y: Shift the text on the y-axis
        :return: frame on which the new rectangle will be drawn on
        """

        x, y = self.pos
        w, h = self.size

        cv2.rectangle(frame, (x, y), (x + w, y + h), bg_highlight, cv2.FILLED)
        cv2.putText(frame, self.text, (x + (w // 2) + c_x, y + (h // 2) + c_y), cv2.FONT_HERSHEY_PLAIN, 2, fg_highlight, self.text_thickness)

        return frame

    @staticmethod
    def command(command, *args, new_condition=True):
        r"""
        This function performs a command (function). You can pass the parameters into *args tuple.

        :param command: The function you want to execute
        :param *args: The parameters of the function if the function has any
        :param new_condition: an extra condition if you want.
        :return: The command()
        """

        def unpack_params(args):
            for i in args:
                params = i
            return params

        if new_condition:
            if args == ():
                return command()
            else:
                return command(*unpack_params(args))
        else:
            return None


class CirButton:
    r"""
    This class is used to draw a button on the video loop of opencv. This class was created to make the drawing
    of buttons easier and more accessible to newer developers. This button creates a circular button.
    """
    def __init__(self, text, text_pos, center_pos, r, thickness, bg=(0, 0, 0), fg=(255, 255, 255)):
        r"""
        In the constructor the values are being initialized.

        :param text: The text that will be put on the button
        :param text_pos: The position of the text
        :param center_pos: The position of the center of the circle
        :param r: The radius of the circle
        :param thickness: the thickness of the text
        :param bg: The background color
        :param fg: The foreground color of the text
        """

        self.text = text
        self.text_pos = text_pos
        self.center_pos = center_pos
        self.r = r
        self.thickness = thickness
        self.bg = bg
        self.fg = fg

    def draw(self, frame, bg=(0, 0, 0)):
        r"""
        This function is used to draw the button on the frame.

        :param frame: Takes in the frame on which you want to draw the button
        :return: The frame with the button drawn on it
        """
        text_x, text_y = self.text_pos
        cx, cy = self.center_pos
        self.bg = bg

        cv2.circle(frame, (cx, cy), self.r, self.bg, cv2.FILLED)
        cv2.putText(frame, self.text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 2, self.fg, self.thickness)

        return frame

    def highlight(self, frame, bg_highlight, fg_highlight=(255, 255, 255)):
        r"""
        Highlights the rectangle by drawing a new one on top of it with other colors. The other rectangle
        will be deleted

        :param frame: Takes in the frame on which you want to draw the button
        :param bg_highlight: The new background color
        :param fg_highlight: The new foreground color
        :return: frame on which the new rectangle will be drawn on
        """

        cx, cy = self.center_pos
        text_x, text_y = self.text_pos

        cv2.circle(frame, (cx, cy), self.r, bg_highlight, cv2.FILLED)
        cv2.putText(frame, self.text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 2, fg_highlight, self.thickness)

        return frame

    @staticmethod
    def command(command, *args, new_condition=True):
        r"""
        This function performs a command (function). You can pass the parameters into *args tuple.

        :param command: The function you want to execute
        :param *args: The parameters of the function if the function has any
        :param new_condition: an extra condition if you want.
        :return: The command()
        """

        def unpack_params(args):
            for i in args:
                params = i
            return params

        if new_condition:
            if args == ():
                return command()
            else:
                return command(*unpack_params(args))
        else:
            return None


class LiveColorDetector:
    def __init__(self, name_tb, win_w, win_h, hue_min=(81, 179), hue_max=(179, 179), sat_min=(0, 255), sat_max=(255, 255), val_min=(0, 255), val_max=(105, 255)):
        r"""
        In the constructor the values are initialized. The trackbar window to control the image mask is also
        created.

        :param name_tb: The slider name
        :param win_w: The width of the window
        :param win_h: The height of the window
        :param hue_min: The initial starting value of the slider
        :param hue_max: The initial starting value of the slider
        :param sat_min: The initial starting value of the slider
        :param sat_max: The initial starting value of the slider
        :param val_min: The initial starting value of the slider
        :param val_max: The initial starting value of the slider
        """

        self.name_tb = name_tb
        self.win_w = win_w
        self.win_h = win_h
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max

        cv2.namedWindow(self.name_tb)
        cv2.resizeWindow(self.name_tb, 640, 240)
        cv2.createTrackbar("Hue min", self.name_tb, self.hue_min[0], self.hue_min[1], self.__empty)
        cv2.createTrackbar("Hue max", self.name_tb, self.hue_max[0], self.hue_max[1], self.__empty)
        cv2.createTrackbar("Sat min", self.name_tb, self.sat_min[0], self.sat_min[1], self.__empty)
        cv2.createTrackbar("Sat max", self.name_tb, self.sat_max[0], self.sat_max[1], self.__empty)
        cv2.createTrackbar("Val min", self.name_tb, self.val_min[0], self.val_min[1], self.__empty)
        cv2.createTrackbar("Val max", self.name_tb, self.val_max[0], self.val_max[1], self.__empty)

    def __del__(self):
        r"""
        The destructor is used for garbage collection. To destroy the memory of the object, and reset the
        values of the variables.

        :return:
        """

        self.name_tb = None
        self.win_w = None
        self.win_h = None
        self.hue_min = None
        self.hue_max = None
        self.sat_min = None
        self.sat_max = None
        self.val_min = None
        self.val_max = None

    def detect(self, img_path, title="Live color detector", show_img=True, img_scale=1):
        r"""
        This function is used for controlling the HSV values live via an mask.

        :param img_path: The path to the image you want to show
        :param title: The title of the window
        :param show_img: If you want to show the standard window
        :param img_scale: The scale of the displayed image

        :return: returns the full stacked img

        """

        img = cv2.imread(img_path)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        self.h_min = cv2.getTrackbarPos("Hue min", self.name_tb)
        self.h_max = cv2.getTrackbarPos("Hue max", self.name_tb)
        self.s_min = cv2.getTrackbarPos("Sat min", self.name_tb)
        self.s_max = cv2.getTrackbarPos("Sat max", self.name_tb)
        self.v_min = cv2.getTrackbarPos("Val min", self.name_tb)
        self.v_max = cv2.getTrackbarPos("Val max", self.name_tb)

        lower = numpy.array([self.h_min, self.s_min, self.v_min])
        upper = numpy.array([self.h_max, self.s_max, self.v_max])

        mask = cv2.inRange(img_hsv, lower, upper)

        self.img_result = cv2.bitwise_and(img, img, mask=mask)

        full_img = img_stacker(img_scale, ([img, img_hsv], [mask, self.img_result]))

        if show_img:
            cv2.imshow(title, full_img)

        return full_img

    def save_result(self, save_path):
        r"""
        This function save the result of the img.

        :param save_path: The path where you want the image to be saved
        """
        cv2.imwrite(save_path, self.img_result)

    def get_hue_min(self):
        r"""
        This function gets the current trackbar value from the Hue min.

        :return: h_min
        """

        return self.h_min

    def get_hue_max(self):
        r"""
        This function gets the current trackbar value from the Hue max.

        :return: h_max
        """

        return self.h_max

    def get_sat_min(self):
        r"""
        This function gets the current trackbar value from the Sat min.

        :return: s_min
        """

        return self.sat_min

    def get_sat_max(self):
        r"""
        This function gets the current trackbar value from the Sat max.

        :return: s_max
        """

        return self.sat_max

    def get_val_min(self):
        r"""
        This function gets the current trackbar value from the Val min.

        :return: v_min
        """

        return self.val_min

    def get_val_max(self):
        r"""
        This function gets the current trackbar value from the Val max.

        :return: v_max
        """

        return self.v_max

    @staticmethod
    def __empty(x):
        r"""
        This function is empty.
        """
        pass


class FPS:
    def __init__(self):
        r"""
        In the constructor the values are initialized
        """

        self.p_time = 0
        self.c_time = 0
        self.fps = 0

    def __del__(self):
        r"""
        In the destructor the values are being reset and the objects will be garbage collected.
        :return:
        """
        self.p_time = 0
        self.c_time = 0
        self.fps = 0

    def draw_fps(self, frame, pos: tuple, font, draw=True, font_scale=3, color=(0, 0, 0), thickness=2):
        r"""
        This function handles the frames per second (fps) of the model. The fps indicates how many times the frame
        is being played per second.

        :arg
        :param frame: the frame on which the fps will be displayed on
        :param pos: the position of fps
        :param font: the different text styles
        :param draw: true or false if you want the fps to be drawn on the screen
        :param font_scale: the scale of the font
        :param color: input is a tuple with rgb values
        :param thickness: the thickness of the font
        :return: the fps
        """

        self.c_time = time.time()
        self.fps = 1 / (self.c_time - self.p_time)
        self.p_time = self.c_time

        if draw:
            cv2.putText(frame, f"FPS: {int(self.fps)}", pos, font, font_scale, color, thickness)

    def get_fps(self):
        r"""
        This function gets the current fps.

        :return: int(fps)
        """
        return int(self.fps)


def draw_background(frame_shape, color=(255, 255, 255)):
    r"""
    This function can draw a background on a opencv window frame. The background can only be black or white.

    :param frame_shape: Takes in the shape of the frame as parameter, so the fw, fh, c
    :param color: The color of the background
    :return: The new and updated frame
    """

    w, h, c = frame_shape

    frame = numpy.zeros((w, h, c), numpy.uint8)
    frame[:] = color

    return frame


def key_pressed(key):
    r"""
    This function returns some hexadecimal logic to break out of a unlimited loop.

    :param key: takes in a char to break out of the loop
    :return: key
    """
    return cv2.waitKey(1) & 0xFF == ord(key)


def face_locations(frame):
    r"""
    This function returns the locations of the face.

    :param frame: the frame on which the fps will be displayed on
    :return: tuple of face locations
    """
    return fr.face_locations(frame)


def face_encodings(frame, face_locations):
    r"""
    This function finds the encodings in the face.

    :param frame: the frame on which the fps will be displayed on
    :param face_locations: gets a tuple of the current face locations
    :return: returns a tuple of face locations
    """
    return fr.face_encodings(frame, face_locations)


def change_res(w, h, capture):
    r"""
    This function easily sets the width and height of the capture

    :param capture: Takes in the opencv-python capture objects
    :param w: The width of the capture
    :param h: The height of the capture
    """

    capture.set(3, w)
    capture.set(4, h)


def draw_custom_text(frame, font_ttf, text, pos, font_size=32, color=(255, 255, 255, 0)):
    r"""
    This function can be used to draw text on the opencv frame with a custom font.

    :param frame: The opencv capture frame
    :param font_ttf: The file path to the font
    :param text: The text that you want to display
    :param pos: The position of the text
    :param font_size: The size of the font, the standard value is 32
    :param color: The color of the text
    :return: Returns a new frame on which the text is drawn
    """

    font = ImageFont.truetype(font_ttf, font_size)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    new_frame = numpy.array(img_pil)

    return new_frame


def for_each_img(dir_path: str):
    r"""
    This function can be used to loop through a directory to get all th images within.

    :param dir_path: Takes in the path to the directory with the images in it
    :yield: This function returns a iterable object
    """

    if len(os.listdir(dir_path)) == 0:
        yield "The directory you've given is empty"
    else:
        for file in os.listdir(dir_path):
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".bmp"):
                yield file


def file_splitter(dir_path: str):
    r"""
    This function can be used to split file names from the extensions. This function returns
    a iterable object

    :param dir_path: The path to the directory in which you want to split the files
    :yield: A list containing the new image names
    """

    new_image_name = []

    if len(os.listdir(dir_path)) == 0:
        yield "The directory you've given is empty"
    else:
        for files in os.listdir(dir_path):
            new_image_name.append(os.path.splitext(files)[0])

        yield new_image_name


def img_stacker(scale, img_array):
    r"""
    This function is able to stack images vertically or horizontally.

    :param scale: The scale of the joint together images
    :param img_array: Pass in a img array like ([img, img, img], [img, img, img]) or in a different form
    :return: The stacked images
    """

    rows = len(img_array)
    cols = len(img_array[0])
    rowsAvailable = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = numpy.zeros((height, width, 3), numpy.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = numpy.hstack(img_array[x])
        ver = numpy.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = numpy.hstack(img_array)
        ver = hor
    return ver


def get_contours(img_canny, img, draw_contours, draw=True, color=(0, 0, 0), thickness=2, min_area=500, ct_mode=cv2.RETR_EXTERNAL):
    r"""
    This function displays the contours of a image.

    :param img_canny: cv.Canny img
    :param img: The original img
    :param draw_contours: The contours you want to draw
    :param draw: If you want to draw on the img, true = draw
    :param color: The color of the contours
    :param thickness: The thickness of the drawn contours
    :param min_area: The minimum area the contours cover
    :param ct_mode: The getting contours mode
    :return: The amount of corners the object has & the img where the contours have been drawn on
    """

    contours, hierarchy = cv2.findContours(img_canny, ct_mode, cv2.CHAIN_APPROX_NONE)
    object_corners = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(img, cnt, draw_contours, color, thickness)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            obj_cor = len(approx)
            object_corners.append(obj_cor)
            if draw:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(img, (x, y), (x + w, y + h), (67, 255, 87), 1)

    return object_corners, img


def img_hist_gray(img, img_title="Gray img", mask=None, range=[0, 256], num_bins=256, fig_name="Gray Histogram", plt_title="Histogram", plt_x_label="Bins", plt_y_label="# of pixels"):
    r"""
    This function displays the img + a histogram for the grayscale values.

    :param img: The img in which you want to display the grayscale histogram
    :param img_title: The title of the img
    :param mask: The mask of the img
    :param range: The range of the bins on the histogram
    :param num_bins: The number of bins on the x-axis
    :param fig_name: The name of the matplotlib figure
    :param plt_title: The title of the figure
    :param plt_x_label: The label of the x-axis
    :param plt_y_label: The label of the y-axis
    """

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(img_title, gray_img)

    gray_hist = cv2.calcHist([gray_img], [0], mask, [num_bins], range)

    plt.figure(fig_name)
    plt.title(plt_title)
    plt.xlabel(plt_x_label)
    plt.ylabel(plt_y_label)
    plt.plot(gray_hist)
    plt.xlim(range)
    plt.show()


def img_hist_bgr(img, img_title="BGR img", mask=None, range=[0, 256], num_bins=256, fig_name="BGR Histogram", plt_title="Histogram", plt_x_label="Bins", plt_y_label="# of pixels"):
    r"""
    This function displays the img + a histogram for the BGR values.


    :param img: The img in which you want to display the bgr histogram
    :param img_title: The title of the img
    :param mask: The mask of the img
    :param range: The range of the bins on the histogram
    :param num_bins: The number of bins on the x-axis
    :param fig_name: The name of the matplotlib figure
    :param plt_title: The title of the figure
    :param plt_x_label: The label of the x-axis
    :param plt_y_label: The label of the y-axis
    """

    cv2.imshow(img_title, img)

    plt.figure(fig_name)
    plt.title(plt_title)
    plt.xlabel(plt_x_label)
    plt.ylabel(plt_y_label)
    colors = ('b', 'g', 'r')

    for idx, col in enumerate(colors):
        hist = cv2.calcHist([img], [idx], mask, [num_bins], range)
        plt.plot(hist, color=col)
        plt.xlim(range)

    plt.show()


def corner_rectangle(frame, bbox, color, cons=30, thickness=1, rectangle_thickness=1, line_type=None):
    r"""
    This function draws a corner rectangle on the screen.

    :param frame: The frame on which you want to display the corner rectangle
    :param bbox: The bbox of the rectangle (x, y, w, h)
    :param color: The color of the rectangle
    :param cons: The length off the different lines
    :param thickness: The thickness of the lines
    :param rectangle_thickness: The thickness of the rectangle, if rt = 0 then the rectangle isn't drawn
    :param line_type: The type of the lines
    """

    x, y, w, h = bbox

    if rectangle_thickness != 0:
        cv2.rectangle(frame, bbox, color, rectangle_thickness)

    cv2.line(frame, (x, y), (x + cons, y), color, thickness, lineType=line_type)
    cv2.line(frame, (x, y), (x, y + cons), color, thickness, lineType=line_type)
    cv2.line(frame, ((x + w), y), ((x + w) - cons, y), color, thickness, lineType=line_type)
    cv2.line(frame, ((x + w), y), ((x + w), y + cons), color, thickness, lineType=line_type)
    cv2.line(frame, (x, (y + h)), (x + cons, (y + h)), color, thickness, lineType=line_type)
    cv2.line(frame, (x, (y + h)), (x, (y + h) - cons), color, thickness, lineType=line_type)
    cv2.line(frame, ((x + w), (y + h)), ((x + w) - cons, (y + h)), color, thickness, lineType=line_type)
    cv2.line(frame, ((x + w), (y + h)), ((x + w), (y + h) - cons), color, thickness, lineType=line_type)

    return frame