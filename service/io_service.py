from collections import namedtuple
import numpy
import os

import cv2


VideoSettings = namedtuple('VideoSettings', ['filename', 'fps', 'resolution'])
my_settings = VideoSettings(filename='video.avi', fps=24.0, resolution='720p')


class IOService:

    @staticmethod
    def fetch_frame(source: str) -> numpy.ndarray:
        return cv2.imread(source)

    @staticmethod
    def display_frame(framename: str, frame: numpy.ndarray):
        cv2.imshow(winname=framename, mat=frame)

    @staticmethod
    def save_frame(filename: str, frame: numpy.ndarray):
        cv2.imwrite(filename=filename, img=frame)

    # @staticmethod
    # def wait():
    #     key = cv2.waitKey()
    #     if key == 27:
    #         cv2.destroyAllWindows()


class VideoService(IOService):
    # Standard Vie Dimensions Sizes
    STD_DIMENSIONS_MAP = {
        '480p': (640, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '4k': (3840, 2160),
    }

    VIDEO_TYPE_MAP = {
        '.avi': cv2.VideoWriter_fourcc(*'XVID'),
        '.mp4': cv2.VideoWriter_fourcc(*'XVID'),
    }

    @staticmethod
    def fetch_frame(source: cv2.VideoCapture) -> numpy.ndarray:
        ret, current_frame = source.read()
        return current_frame

    @staticmethod
    def change_resolution(capture: cv2.VideoCapture, width: int = 640, height: int = 480) -> cv2.VideoCapture:
        """
        Set a resolution for the video capture.
        :param capture:
        :param width:
        :param height:
        :return:
        """
        capture.set(3, width)
        capture.set(4, height)
        return capture

    def get_dimensions(self, resolution: str = '480p') -> tuple:
        """
        Returns dimensions for requested resolution.
        :param resolution:
        :return:
        """
        return self.STD_DIMENSIONS_MAP.get(resolution) or self.STD_DIMENSIONS_MAP.get('480p')

    def get_video_codec(self, filename: str) -> int:
        """
        Fetch video codec using file extension.
        :param filename:
        :return:
        """
        _, ext = os.path.splitext(filename)
        return self.VIDEO_TYPE_MAP.get(ext) or self.VIDEO_TYPE_MAP.get('.avi')

    def set_video_capture(self, settings: VideoSettings):
        capture = cv2.VideoCapture(0)
        dimensions = self.get_dimensions(settings.resolution)
        return self.change_resolution(capture, *dimensions)

    def set_video_writer(self, settings: VideoSettings):
        video_codec = self.get_video_codec(settings.filename)
        dimensions = self.get_dimensions(my_settings.resolution)
        return cv2.VideoWriter(settings.filename, video_codec, settings.fps, dimensions)  # `isColor=False` if gray needed