import unittest
import Models
from Models.constants import BLANK_FRAME
import cv2 as cv


class TestFunc(unittest.TestCase):
    def test_close_win(self):
        self.assertEqual(Models.close_win('q'), False)

class TestFPS(unittest.TestCase):
    fps = Models.FPS()

    def test_draw_fps(self):
        self.assertEqual(self.fps.draw_fps(BLANK_FRAME, (60, 70), cv.FONT_HERSHEY_TRIPLEX), None)

    def test_get_fps(self):
        self.assertGreaterEqual(self.fps.get_fps(), 0)


class TestFR(unittest.TestCase):
    rec = Models.models.FaceRecognition("", split_files=False)

    def test_attr(self):
        self.assertEqual(self.rec.saved_images, [])
        self.assertEqual(self.rec.image_names, [])

    def test_find_encodings(self):
        self.assertEqual(self.rec.find_encodings([]), [])

    def test_is_face_recognized(self):
        self.assertEqual(self.rec.is_face_recognized(), False)
        self.assertEqual(self.rec.is_recognized, None)

    def test_is_face_detected(self):
        self.assertEqual(self.rec.is_face_detected(), True)
        self.assertEqual(self.rec.face_location, ())


class TestFD(unittest.TestCase):
    det = Models.models.FaceDetection()

    def test_attr(self):

        self.assertEqual(self.det.face_location, ())

    def test_is_face_detected(self):
        self.assertEqual(self.det.is_face_detected(), True)
        self.assertEqual(self.det.face_location, ())


class TestHT(unittest.TestCase):
    ht = Models.models.HandTracker()

    def test_attr(self):
        self.assertEqual(self.ht.mode, False)
        self.assertEqual(self.ht.detection_conf, 0.5)
        self.assertEqual(self.ht.tracking_conf, 0.5)
        self.assertEqual(self.ht.complexity, 1)
        self.assertEqual(self.ht.max_hands, 2)


if __name__ == '__main__':
    unittest.main()
