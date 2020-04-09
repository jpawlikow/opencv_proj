import cv2
import numpy
from collections import namedtuple


DetectedObject = namedtuple('DetectedObject', ('frame', 'origin_start', 'origin_end'))


class FaceDetector:
    DEFAULT_CLASSIFIER = 'cascades/data/haarcascades/haarcascade_frontalface_alt2.xml'

    def __init__(self, classifier: str = None):
        """

        :param classifier:
        """
        self.classifier = classifier or self.DEFAULT_CLASSIFIER
        self.cascade = cv2.CascadeClassifier(self.classifier)

    def _check_frame(self, frame: numpy.ndarray):
        """

        :param frame:
        :return:
        """
        return self.cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)

    def inspect_frame(self, frame: numpy.ndarray, gray: bool = False):
        """

        :param frame:
        :param gray:
        :return:
        """
        detected_objects = []
        prepared_frame = frame
        faces = self._check_frame(prepared_frame)
        for x, y, width, height in faces:
            origin_start, origin_end = (x, y), (x + width, y + height)
            frm = prepared_frame[y:y + height, x:x + width] if gray else frame[y:y + height, x:x + width]
            detected_objects.append(DetectedObject(frame=frm, origin_start=origin_start, origin_end=origin_end))

        return detected_objects
