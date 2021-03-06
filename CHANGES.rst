Version 0.0.1
-------------

Released

-   Support for Python 3.6.
-   Support for Python 3.7.
-   Support for Python 3.8.
-   Support for Python 3.9.

- Additions utility functions/classes
    -   Added ``FaceRecognition`` class
    -   Added ``FaceDetection`` class
    -   Added ``HandTracker`` class
    -   Added ``FPS`` class
    -   Added ``RecButton`` class
    -   Added ``CirButton`` class
    -   Added ``draw_background`` function
    -   Added ``close_win`` function
    -   Added ``face_locations`` function
    -   Added ``face_encodings`` function
    -   Added ``set_capture_size`` function
    -   Added ``draw_custom_text`` function
    -   Added ``for_each_img`` function
    -   Added ``file_splitter`` function

Version 0.0.2
-------------

Unreleased

- Hotfixes
    - Removed ``WEBCAM_CAP`` from constants.py.
    - Removed unnecessary import from models.py.
    - Added color selection system to the ``draw_background`` function.
    - Changed parameters from ``draw_background``.
    - Added ``ALL_CONTOURS`` constant.
    - Added finger constants for `HandTracker`.
    - Made some parameter changes to `RecButton` class.
    - Changed the color constants from RGB to BGR.
    - Changed function name ``set_capture_size`` to ``change_res``.
    - Changed calculations for width and height calculations for the ``RecButton().draw()`` class function.

- Additions utility functions/classes.
    - Added ``LiveColorDetector`` class.
    - Added ``img_stacker`` function.
    - Added ``get_contour`` function.
    - Added ``rescale_frame`` function.
    - Added ``img_hist_gray`` function.
    - Added ``img_hist_bgr`` function.
    - Added mobilenet ``ObjectDetection`` class.
    - Added ``set_brightness`` function.
    - Added command features to the ``Recbutton & CirButton`` classes. Now the can call functions with the ``.command(condition, command)`` function.
    - Created new ``img_processing.py`` file.
    - Added ``process`` function.


