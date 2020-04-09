import cv2
import numpy
from pytesseract import pytesseract


class TrackBar:
    def __init__(self, name: str, track_type: str):
        self.window_name = name
        self.track_type = track_type

    def track(self, *args, **kwargs):
        def placeholder(value):
            pass
        cv2.namedWindow(self.window_name)
        if self.track_type == 'HSV':
            cv2.createTrackbar('Lower H', self.window_name, 0, 179, placeholder)
            cv2.createTrackbar('Lower S', self.window_name, 0, 255, placeholder)
            cv2.createTrackbar('Lower V', self.window_name, 0, 255, placeholder)
            cv2.createTrackbar('Upper H', self.window_name, 179, 179, placeholder)
            cv2.createTrackbar('Upper S', self.window_name, 255, 255, placeholder)
            cv2.createTrackbar('Upper V', self.window_name, 255, 255, placeholder)


class FrameService:
    @staticmethod
    def rescale_frame(frame: numpy.ndarray, percent: int = 75) -> numpy.ndarray:
        """
        Rescales passed frame by provided percent.
        :param frame: Frame to be modified
        :param percent: Percent to which frame will be rescaled
        :return: Rescaled frame
        """
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def gray_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Set passed frame to gray scale.
        Probably something like -> gray = 0.299 * R + 0.587 * G + 0.114 * B
        :param frame: Frame to be modified
        :return: Frame in gray scale
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def rectangle_mark_frame(frame: numpy.ndarray, point1: tuple, point2: tuple):
        color = (255, 0, 0)
        stroke = 2
        cv2.rectangle(img=frame, pt1=point1, pt2=point2, color=color, thickness=stroke)

    @staticmethod
    def text_sign_frame(frame: numpy.ndarray,
                        value: str,
                        start_point: tuple,
                        font: int = cv2.FONT_HERSHEY_SIMPLEX,
                        color: tuple = (255, 255, 255),
                        stroke: int = 2):
        """
        Applies text to frame - due to specified parameters
        :param frame: Frame to be modified
        :param value: Value - text - to be applied
        :param start_point: Start point, where text will be applied
        :param font: Font for text
        :param color: Color for text
        :param stroke: Text stroke
        :return: Frame with text applied
        """
        cv2.putText(frame, value, start_point, font, 1, color, stroke, cv2.LINE_AA)

    @staticmethod
    def invert_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Inverts frame -> R = 255 - R, G = 255 - G, B = 255 -B
        :param frame: Frame to be modified
        :return: Inverted frame
        """
        return cv2.bitwise_not(frame)

    @staticmethod
    def normalize_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Modifies frame's pixels so that the number of pixels at each gray level will be approximately the same.
        That can increase frame contrast.
        :param frame: Frame to be modified
        :return: Frame with normalized pixels
        """
        normalized_image = cv2.equalizeHist(frame)
        # normalized_image = numpy.zeros((len(image), len(image[1])))
        # normalized_image = cv2.normalize(image, normalized_image, 0, 255, cv2.NORM_MINMAX)
        return normalized_image

    @staticmethod
    def resize_frame(frame: numpy.ndarray, size: tuple) -> numpy.ndarray:
        resized_frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
        return resized_frame

    @staticmethod
    def threshold_frame(frame: numpy.ndarray,
                        threshold: int = 127,
                        max_val: int = 255,
                        method: str = 'binary') -> numpy.ndarray:
        """
        Threshold checks/modified pixels by specified range.
        eg. if pixel < 127 -> pixel = 0; else pixel = 255 (binary method)
        There is bunch of algorithms which decides how to modify pixels which met requirements.
        :param frame: Frame to be modified
        :param threshold: Value below which pixel will be modified to lowest
        :param max_val: Max value to be injected
        :param method: Method of threshold
        :return: Threshold frame
        """
        method_map = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
        }
        ret, thresh1 = cv2.threshold(frame, threshold, max_val, method_map[method])
        return thresh1

    @staticmethod
    def low_pass_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Blur, smooth - kernel [[1,1,1], [1,1,1], [1,1,1]].
        That makes pixel "average" of its neighbours.
        Rising up value of central pixel, blur is less visible
        :param frame: Frame to be modified
        :return: Low passed frame
        """
        return cv2.blur(frame, (36, 36))

    @staticmethod
    def median_pass_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Takes median value of neighbours and put inplace high frequented pixel.
        Remove salt & pepper
        :param frame: Frame to be modified
        :return: Smoothed frame
        """
        return cv2.medianBlur(frame, 11)

    @staticmethod
    def high_pass_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Allows to detect elements with high frequency. Rising up noise
        :param frame: Frame to be modified
        :return: High passed frame
        """
        kernel = numpy.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), numpy.float32)
        return cv2.filter2D(frame, -1, kernel)

    @staticmethod
    def red_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Makes frame red oriented by removing rest of colors.
        :param frame: Frame to be modified
        :return: Frame in red scale
        """
        frame[:, :, 0] = 0
        frame[:, :, 1] = 0
        return frame

    @staticmethod
    def rotate_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Rotates frame 90 degrees (left)
        :param frame: Frame to be modified
        :return: Rotated frame
        """
        frame = FrameService.resize_frame(frame, (480, 480))
        h, w, c = frame.shape
        empty_img = numpy.zeros([h, w, c], dtype=numpy.uint8)
        for i in range(h):
            for j in range(w):
                empty_img[i, j] = frame[j, i]
        return empty_img

    @staticmethod
    def inc_bright_frame(frame: numpy.ndarray, increase: int = 100) -> numpy.ndarray:
        """
        Using Hue Saturation Value (in this case Value - which stands for brightness).
        Increases Value(scale from black - 0 to white - 255) of image by passed value.
        Can be also be done by modifying RGB values by increasing its value.
        :param frame: Frame to be modified
        :param increase: Value to be added to each pixel's Value
        :return: Frame with raised brightness
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value = frame[:, :, 2]
        value = numpy.where(value <= 255 - increase, value + increase, 255)
        frame[:, :, 2] = value
        return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    @staticmethod
    def detect_edges_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Detects edges. Probably using Sobol filtering (eg. [[1,1,1], [0,0,0], [-1,-1,-1]])
        :param frame: Frame to be modified
        :return: Frame with detected edges
        """
        gray_filtered = cv2.bilateralFilter(frame, 7, 50, 50)
        return cv2.Canny(gray_filtered, 40, 80)

    @staticmethod
    def segment_frame(frame: numpy.ndarray, tracker: TrackBar) -> numpy.ndarray:
        """
        Using tracker to mask a frame by HSV range
        :param frame: Frame to be modified
        :param tracker: Used to gather range data
        :return: Segmented frame
        """
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        low_h = cv2.getTrackbarPos('Lower H', tracker.window_name)
        low_s = cv2.getTrackbarPos('Lower S', tracker.window_name)
        low_v = cv2.getTrackbarPos('Lower V', tracker.window_name)
        upp_h = cv2.getTrackbarPos('Upper H', tracker.window_name)
        upp_s = cv2.getTrackbarPos('Upper S', tracker.window_name)
        upp_v = cv2.getTrackbarPos('Upper V', tracker.window_name)

        start = numpy.array([low_h, low_s, low_v])
        end = numpy.array([upp_h, upp_s, upp_v])

        mask = cv2.inRange(hsv_image, start, end)
        return cv2.bitwise_and(frame, frame, mask=mask)

    @staticmethod
    def skeletonize_frame(frame: numpy.ndarray) -> numpy.ndarray:
        """
        Skeletonize frame by continuously dilating and eroding threshed frame.
        :param frame: Frame to be modified
        :return: Skeleton of frame
        """
        current_frame = FrameService.threshold_frame(frame)
        skeleton = numpy.zeros(current_frame.shape, numpy.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(current_frame, kernel)
            dilated = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(current_frame, dilated)
            skeleton = cv2.bitwise_or(skeleton, temp)
            current_frame = eroded.copy()
            if cv2.countNonZero(current_frame) == 0:
                break
        return skeleton

    @staticmethod
    def dilated_frame(frame: numpy.ndarray, iterations: int = 1) -> numpy.ndarray:
        """
        Dilates frame. Takes kernel and takes max value of it (and applies to filtered pixel)
        :param frame: Frame to be modified
        :param iterations: Number of iterations
        :return: Dilated frame
        """
        kernel = numpy.ones((5, 5), numpy.uint8)
        eroded_frame = cv2.dilate(frame, kernel, iterations=iterations)
        return eroded_frame

    @staticmethod
    def detect_text_frame(frame: numpy.ndarray) -> numpy.ndarray:
        kernel = numpy.ones((1, 1), numpy.uint8)
        frame = cv2.dilate(frame, kernel, iterations=1)
        frame = cv2.erode(frame, kernel, iterations=1)
        # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite('tmp.png', frame)
        result = pytesseract.image_to_string('tmp.png')
        print(result)
        return frame
