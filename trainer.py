import cv2
import numpy

from face_detector import FaceDetector
from service.io_service import IOService
from service.trainer_service import SampleService
from service.frame_service import FrameService

SS = SampleService()
IOS = IOService()
FS = FrameService()
FD = FaceDetector()


def train():
    training_samples = []
    training_samples_labels = []
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    samples = SS.fetch_samples()
    SS.create_id_label_map(samples)
    for sample in samples:
        frame = IOS.fetch_frame(sample.path)
        gray_frame = FS.gray_frame(frame)
        for face in FD.inspect_frame(gray_frame, gray=True):
            training_samples.append(face.frame)
            training_samples_labels.append(sample.label_id)
    recognizer.train(training_samples, numpy.array(training_samples_labels))
    recognizer.save('samplesDB.yml')
