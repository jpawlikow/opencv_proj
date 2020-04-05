from collections import namedtuple
import cv2
import json
import sys

from sampler import sampler
from trainer import train
from service.io_service import VideoService, VideoSettings
from service.frame_service import FrameService, TrackBar
from face_detector import FaceDetector

Function = namedtuple('Function', ('func', 'gray', 'params', 'track', 'track_params'))

FUNCTIONS_MAP = {
    ord('a'): Function(func=FrameService.invert_frame, gray=False, params={}, track=None, track_params={}),
    ord('b'): Function(func=FrameService.gray_frame, gray=False, params={}, track=None, track_params={}),
    ord('c'): Function(func=FrameService.normalize_frame, gray=True, params={}, track=None, track_params={}),
    ord('d'): Function(func=FrameService.rescale_frame, gray=False, params={}, track=None, track_params={}),
    ord('e'): Function(func=FrameService.threshold_frame,
                       gray=True, params={'threshold': 127,
                                          'max_val': 255,
                                          'method': 'binary'},
                       track=None,
                       track_params={}),
    ord('f'): Function(func=FrameService.low_pass_frame, gray=False, params={}, track=None, track_params={}),
    ord('g'): Function(func=FrameService.median_pass_frame, gray=False, params={}, track=None, track_params={}),
    ord('h'): Function(func=FrameService.high_pass_frame, gray=False, params={}, track=None, track_params={}),
    ord('i'): Function(func=FrameService.red_frame, gray=False, params={}, track=None, track_params={}),
    ord('j'): Function(func=FrameService.rotate_frame, gray=False, params={}, track=None, track_params={}),
    ord('k'): Function(func=FrameService.inc_bright_frame,
                       gray=False,
                       params={'increase': 100},
                       track=None,
                       track_params={}),
    ord('l'): Function(func=FrameService.detect_edges_frame, gray=True, params={}, track=None, track_params={}),
    ord('m'): Function(func=FrameService.segment_frame,
                       gray=False,
                       params={},
                       track=TrackBar,
                       track_params={'name': 'Segment Track Bar', 'track_type': 'HSV'}),
    ord('n'): Function(func=FrameService.skeletonize_frame, gray=True, params={}, track=None, track_params={}),
    ord('o'): Function(func=FrameService.dilated_frame,
                       gray=False,
                       params={'iterations': 2},
                       track=None,
                       track_params={}),
}


def main():
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
        elif sys.argv[1] == 'filters':
            main()
