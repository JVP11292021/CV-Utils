import os

try:
    import numpy
    import cv2
    import face_recognition as fr
    import mediapipe as mp
    import matplotlib.pyplot as plt
except Exception:
    print("Something went wrong with the package dependencies. ")


class FaceRecognition:
    r"""This class is used for facial recognition. It works by way of a dataset that the user has to create him/herself
    The dataset consists of one folder with in it one picture of the different people you would like the model to
    recognize. The files in the dataset need to be named accordingly to the corresponding persons name, because
    the program uses that name to identify that person.

    attr:
    saved_images: data for all the images in the dataset, data is seen as array data
    image_names: displays all the names of the images in the dataset folder
    """

    saved_images = []
    image_names = []

    def __init__(self, dataset, split_files=True):
        r"""
        In the constructor the image names are converted from *.jpg to *. setting some values in the constructor.
        
        :arg
        :param dataset: string path to dataset
        :param split_files:
        """
        self.dataset = dataset
        self.face_location = ()
        self.is_recognized = None

        if split_files:
            img_file_names = os.listdir(self.dataset)

            for files in img_file_names:
                # reading in each image in the path
                cur_img = cv2.imread(f"{self.dataset}/{files}")
                # saving that image in the saved_images list
                self.saved_images.append(cur_img)
                # splitting the .* from the image name
                self.image_names.append(os.path.splitext(files)[0])

    def __del__(self):
        r"""
        The destructor is used for garbage collection and to reset the values of the variables.

        :return:
        """
        self.image_names.clear()
        self.saved_images.clear()
        self.face_location = ()

    @staticmethod
    def find_encodings(images: list):
        r"""
        This function returns face encodings, face encodings are features in the face like special keypoints nose
        and so on.

        :arg
        :param images: takes in list of images
        :return: list of image encodings
        """

        encode_list = []

        # this returns the amount of people the recognizer knows in total
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = fr.face_encodings(img)[0]
            encode_list.append(encode)

        return encode_list

    def find_faces(self, frame, face_encodings, face_locations, encode_known_faces, color=(255, 255, 255), draw=True):
        r"""
        This function finds the faces in the frames that are being displayed.

        :arg
        :param frame: the frame on which the fps will be displayed on
        :param face_encodings: wants a list with the facial features
        :param face_locations: wants a tuple with the live face locations of a person
        :param encode_known_faces: wants a list from the find_encodings function
        :param color: input is a tuple with rgb values
        :param draw: true or false if you want to have the standard drawing active
        :return: tuple data of facial keypoints
        """

        for encode_face, face_loc in zip(face_encodings, face_locations):
            # compare the faces
            matches = fr.compare_faces(encode_known_faces, encode_face)
            # the lower face distance means that the model is sure about it's prediction
            face_dis = fr.face_distance(encode_known_faces, encode_face)
            # match index takes the minimum number of the facedis array that way it knows which is the correct face
            match_index = numpy.argmin(face_dis)

            if matches[match_index]:
                self.is_recognized = matches[match_index]
                # the name of the person the model has predicted
                name = self.image_names[match_index].upper()

                # the locations of the face the correct order[3-0, 1-2]
                y1, x2, y2, x1 = face_loc
                # multiplying the values by four because i sized down the image earlier
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                self.face_location = y1, x2, y2, x1

                if draw:
                    # drawing the rectangles and the text on the screen
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)

                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 2)
        return self.face_location

    def is_face_detected(self):
        r"""
        This function checks if there is a face being detected. NOT RECOGNIZED.

        :arg
        :return: false or true depending if there is a face being detected
        """

        if self.face_location is None:
            return False
        else:
            return True

    def is_face_recognized(self):
        r"""
        This function checks if there is a face being recognized. NOT DETECTED.

        :arg
        :return: false or true depending if there is a face being recognized
        """

        if self.is_recognized:
            return True
        else:
            return False


class FaceDetection:
    r"""
    This class is used to make face detection. With this class and opencv-python you can create working
    face detection code.
    """

    def __init__(self):
        r"""
        In the constructor the standard values are being set.
        """

        self.face_location = ()

    def __del__(self):
        r"""
        The destructor is used for garbage collection and to reset the values of the variables.

        :return:
        """

        self.face_location = ()

    def detect_faces(self, frame, face_locations, face_encodings, color=(255, 255, 255), draw=True):
        r"""
        This function detects the faces that are being displayed in the frames.

        :param frame: the frame on which the fps will be displayed on
        :param face_locations: wants a list with the facial features
        :param face_encodings: wants a list with the facial features
        :param color: input is a tuple with rgb values
        :param draw: true or false if you want to have the standard drawing active
        :return: tuple data of facial keypoints
        """

        for face_encodings, face_loc in zip(face_encodings, face_locations):
            y1, x2, y2, x1 = face_loc
            self.face_location = face_loc

            if draw:
                cv2.rectangle(frame, (x2, y1), (x1, y2), color, 2)
        return self.face_location

    def is_face_detected(self):
        r"""
        This function checks if there is a face being detected. NOT RECOGNIZED.

        :arg
        :return: false or true depending if there is a face being detected
        """

        if self.face_location is None:
            return False
        else:
            return True


class HandTracker:
    r"""
    This class is used to make Hand tracker. With this class and opencv-python you can create working
    hand tracker code.
    """

    def __init__(self, mode=False, max_hands=2, complexity=1, detection_conf=0.5, tracking_conf=0.5):
        r"""
        initializes the class attributes and creates an HandTracker object.

        Args:
          :param mode: Whether to treat the input images as a batch of static
            and possibly unrelated images, or a video stream.
          :param max_hands: Maximum number of hands to detect.
          :param complexity: Complexity of the hand landmark model: 0 or 1.
            Landmark accuracy as well as inference latency generally go up with the
            model complexity.
          :param detection_conf: Minimum confidence value ([0.0, 1.0]) for hand
            detection to be considered successful.
          :param tracking_conf: Minimum confidence value ([0.0, 1.0]) for the
            hand landmarks to be considered tracked successfully.
        """

        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode,
                                         max_num_hands=self.max_hands,
                                         model_complexity=self.complexity,
                                         min_detection_confidence=self.detection_conf,
                                         min_tracking_confidence=self.tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def __del__(self):
        r"""
        The destructor is used for garbage collection. To destroy the memory of the object, and reset the
        values of the variables.

        :return:
        """
        self.mode = False
        self.max_hands = 2
        self.complexity = 1
        self.detection_conf = 0.5
        self.tracking_conf = 0.5

    def find_hands(self, frame, draw=True):
        r"""
        This function finds the hands present within the frame. This function converts the color channels of
        the frame from BGR to RGB.

        :param frame: the frame on which the fps will be displayed on
        :param draw: true or false if you want the standard drawing to be drawn on the screen
        :return: frame color channels RGB
        """

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return frame

    def find_position(self, frame, hand_no=0, draw=True):
        r"""
        This function returns a list empty or full of the hand landmarks that are within  the frame.

        :param frame: the frame on which the fps will be displayed on
        :param hand_no:
        :param draw: true or false if you want the all landmark numbers to be drawn on the hand
        :return: returns a list with 3 points (index of lm, x coord, y coord)
        """

        lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for index, lm in enumerate(my_hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lm_list.append([index, cx, cy])

                if draw:
                    cv2.putText(frame, str(index), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        return lm_list

    def is_hand_detected(self):
        r"""
        This function checks if there is a hand present.

        :arg
        :return: false or true depending if there is a hand being detected
        """

        if self.results.multi_hand_landmarks is None:
            return False
        else:
            return True


class MobNetObjectDetection:
    def __init__(self, class_file, config_file, model_file, seed=543210):
        r"""

        """

        with open(class_file) as fp:
            self.labels = fp.read().split("\n")

        numpy.random.seed(seed)
        self.colors = numpy.random.uniform(0, 255, size=(len(self.labels), 3))

        # Read the Tensorflow network
        self.net = cv2.dnn.readNetFromTensorflow(model=model_file, config=config_file)

    def __del__(self):
        self.labels.clear()
        self.net = None

    def pre_process(self, img, scale_factor=0.007, mean=(0, 0, 0)):
        r"""

        """

        dim = 300

        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                     scalefactor=scale_factor,
                                     size=(dim, dim), mean=mean,
                                     swapRB=True, crop=False)

        # Pass blob to the network
        self.net.setInput(blob)

        # Peform Prediction
        detected_objects = self.net.forward()
        return detected_objects

    def display_cv_objects(self, img, detected_objects, min_confidence=0.25, draw=True, title="Object-detector"):
        r"""

        """

        def draw_features(img, text, font=cv2.FONT_HERSHEY_PLAIN, font_scale=0.7, thickness=1):
            cv2.rectangle(img,
                          (upper_left_x, upper_left_y),
                          (lower_right_x, lower_right_y),
                          self.colors[class_index], thickness + 2)

            cv2.putText(img, text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        font, font_scale, self.colors[class_index], thickness)


        width, height = img.shape[1], img.shape[0]

        for i in range(detected_objects.shape[2]):
            confidence = float(detected_objects[0][0][i][2])

            if confidence > min_confidence:
                class_index = int(detected_objects[0, 0, i, 1])

                upper_left_x = int(detected_objects[0, 0, i, 3] * width)
                upper_left_y = int(detected_objects[0, 0, i, 4] * height)
                lower_right_x = int(detected_objects[0, 0, i, 5] * width)
                lower_right_y = int(detected_objects[0, 0, i, 6] * height)

                if draw:
                    draw_features(img=img, text=f"{self.labels[class_index]}: {confidence:.2f}%")
        if draw:
            cv2.imshow(title, img)

        return img

    def display_plt_objects(self, img, objects, min_confidence=0.25, draw=True):
        r"""

        """

        def draw_features(img, text, font=cv2.FONT_HERSHEY_PLAIN, font_scale=0.7, thickness=1):
            cv2.rectangle(img, (upper_left_x, upper_left_y),
                          (upper_left_x + lower_right_x, upper_left_y + lower_right_y),
                          self.colors[class_index], thickness + 2)

            cv2.putText(img, text, (upper_left_x, upper_left_y - 5), font,
                        font_scale, self.colors[class_index], thickness, cv2.LINE_AA)

        height = img.shape[0]
        width = img.shape[1]

        # For every Detected Object
        for i in range(objects.shape[2]):
            # Find the class and confidence
            confidence = float(objects[0, 0, i, 2])

            if confidence > min_confidence:
                class_index = int(objects[0, 0, i, 1])

                # Recover original cordinates from normalized coordinates
                upper_left_x = int(objects[0, 0, i, 3] * width)
                upper_left_y = int(objects[0, 0, i, 4] * height)
                lower_right_x = int(objects[0, 0, i, 5] * width - upper_left_x)
                lower_right_y = int(objects[0, 0, i, 6] * height - upper_left_y)

                # Check if the detection is of good quality
                if draw:
                    draw_features(img, f"{self.labels[class_index]}: {confidence:.2f}%")

        if draw:
            # Convert Image to RGB since we are using Matplotlib for displaying image
            mp_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(30, 10))
            plt.imshow(mp_img)
            plt.show()

        return img

    def get_class_labels(self):
        r"""

        """

        return self.labels




