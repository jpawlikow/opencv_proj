from collections import namedtuple

from service.frame_service import FrameService, TrackBar


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
    ord('p'): Function(func=FrameService.detect_text_frame, gray=True, params={}, track=None, track_params={})
}
