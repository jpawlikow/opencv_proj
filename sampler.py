import cv2

from service.io_service import VideoService, VideoSettings, IOService
from service.frame_service import FrameService
from face_detector import FaceDetector


def sampler(name: str):
    my_settings = VideoSettings(filename='video.avi', fps=24.0, resolution='720p')
    VS = VideoService()

    cap = VS.set_video_capture(my_settings)
    i = 1
    while i < 100:
        current_frame = VS.fetch_frame(cap)
        gray_frame = FrameService.gray_frame(current_frame)
        for face in FaceDetector().inspect_frame(gray_frame, gray=True):
            print(i)
            IOService.save_frame('images/{}/{}k.png'.format(name, i), face.frame)
            i += 1
        VS.display_frame('frame', current_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
