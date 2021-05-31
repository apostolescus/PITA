from threading import Lock
import cv2


class CameraManagerSingleton:
    """Class that manages camera frame capture and resize it to configured dimenssions.
    It supports two mode video capture: from already recorded video or live stream from camera.
    For live camera stream it will select the default camera for the device. If used in video mode
    specify the path to the video file when calling the constructor.
    Use mode='video' or 'camera'.
    Call the get_frame() method to obtain the most recent frame."""

    _lock = Lock()
    __instance = None

    @staticmethod
    def get_instance(config_file):
        if CameraManagerSingleton.__instance is None:
            CameraManagerSingleton(config_file)
        return CameraManagerSingleton.__instance

    def __init__(self, config_file):

        mode = config_file["VIDEO"]["mode"]
        path = config_file["VIDEO"]["path"]
        self.width = int(config_file["VIDEO"]["width"])
        self.height = int(config_file["VIDEO"]["height"])

        if mode == "camera":
            self.camera = cv2.VideoCapture(0)
        else:
            self.camera = cv2.VideoCapture(path)

        CameraManagerSingleton.__instance = self

    def get_frame(self):
        """Reads frame by frame from camera and resize it to specific sizes"""
        ret, frame = self.camera.read()

        frame_shape = frame.shape

        if ret is True:
            if frame_shape[0] > self.height or frame_shape[1] > self.width:
                resized = cv2.resize(
                    frame, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
                self.frame = resized
            else:
                self.frame = frame
            return self.frame

    def close_camera(self):
        self.camera.release()

    def show(self, image=None):
        if image is None:
            cv2.imshow("image", self.frame)
        else:
            cv2.imshow("img", image)


def test_cam_manager_video_rec():

    # initialize cameraManager
    camManager = CameraManagerSingleton.get_instance()

    # initialize videoManager
    vidManager = VideoManagerSingleton.get_instance()

    # specify a value for maximum frames
    counter = 0

    while counter != 200:

        # capture frame from camera
        frame = camManager.get_frame()

        # send it to videoRecorder
        vidManager.record(frame, True)
        counter += 1

    # test if singleton works
    anotherVideo = VideoManagerSingleton.get_instance()

    # save the video
    anotherVideo.save(True)
    camManager.close_camera()


def Test_cam_manager_wrapper():

    camManager = CameraManagerSingleton.get_instance()
    wrapper = VideoManagerWrapper()
    counter = 0

    wrapper.start()

    while counter != 200:
        frame = camManager.get_frame()
        wrapper.record(frame)
        counter += 1

    wrapper.stop()
    camManager.close_camera()
