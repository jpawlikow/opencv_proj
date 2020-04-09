import cv2
import json
import sys

from sampler import sampler
from trainer import train
from service.io_service import VideoService, VideoSettings, IOService
from service.frame_service import FrameService
from face_detector import FaceDetector
from filters import FUNCTIONS_MAP, Function


def filters_cam():
    my_settings = VideoSettings(filename='video.avi', fps=24.0, resolution='480p')
    VS = VideoService()
    cap = VS.set_video_capture(my_settings)
    func = None
    tracker = None
    while True:
        current_frame = VS.fetch_frame(cap)
        key = cv2.waitKey(20)
        if key in FUNCTIONS_MAP:
            func = FUNCTIONS_MAP[key]
        elif key == ord('w'):
            func = None
        elif key == ord('q'):
            break
        if isinstance(func, Function) and func.track and not tracker:
            tracker = func.track(**func.track_params)
            tracker.track()
            func.params.update({'tracker': tracker})

        if isinstance(func, Function) and func.gray:
            current_frame = func.func(FrameService.gray_frame(current_frame), **func.params)
        elif isinstance(func, Function):
            current_frame = func.func(current_frame, **func.params)

        VS.display_frame('frame', current_frame)

    cap.release()
    cv2.destroyAllWindows()


def filters_img(path: str):
    IO = IOService()
    frame = IO.fetch_frame(path)
    func = None
    tracker = None
    while True:
        current_frame = frame.copy()
        key = cv2.waitKey()
        if key in FUNCTIONS_MAP:
            func = FUNCTIONS_MAP[key]
        elif key == ord('w'):
            func = None
            current_frame = frame
        elif key == ord('q'):
            break
        if isinstance(func, Function) and func.track and not tracker:
            tracker = func.track(**func.track_params)
            tracker.track()
            func.params.update({'tracker': tracker})

        if isinstance(func, Function) and func.gray:
            current_frame = func.func(FrameService.gray_frame(current_frame), **func.params)
        elif isinstance(func, Function):
            current_frame = func.func(current_frame, **func.params)

        IO.display_frame('frame', current_frame)

    cv2.destroyAllWindows()


def find_face():
    with open('labels.json', 'r') as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    my_settings = VideoSettings(filename='video.avi', fps=24.0, resolution='720p')
    VS = VideoService()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('samplesDB.yml')

    cap = VS.set_video_capture(my_settings)

    while True:
        current_frame = VS.fetch_frame(cap)
        gray_frame = FrameService.gray_frame(current_frame)
        for face in FaceDetector().inspect_frame(gray_frame, gray=True):
            id_, conf = recognizer.predict(face.frame)
            print(id_, conf)
            FrameService.text_sign_frame(frame=current_frame, value=labels[id_], start_point=face.origin_start)
            FrameService().rectangle_mark_frame(frame=current_frame, point1=face.origin_start, point2=face.origin_end)
        VS.display_frame('frame', current_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if sys.argv[1:]:
        if sys.argv[1] == 'detect_face':
            find_face()
        elif sys.argv[1] == 'sample_face' and sys.argv[2:]:
            sampler(sys.argv[2])
        elif sys.argv[1] == 'train_face':
            train()
        elif sys.argv[1] == 'filters_cam':
            filters_cam()
        elif sys.argv[1] == 'filters_img':
            try:
                filters_img(sys.argv[2])
            except IndexError:
                filters_img('sample1.png')
