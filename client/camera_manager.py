import cv2
from threading import Lock

# TODO: LOAD image resize from config_file

# config_file: video_path, resize shape

class CameraManagerSingleton:
    """Class that manages camera frame capture and resize it to configured dimenssions.
    It supports two mode video capture: from already recorded video or live stream from camera.
    For live camera stream it will select the default camera for the device. If used in video mode 
    specify the path to the video file when calling the constructor. 
    Use mode='video' or 'camera'. 
    Call the getFrame() method to obtain the most recent frame. """

    _lock = Lock()
    __instance = None

    @staticmethod
    def getInstance(mode, path="../video/good.mp4"):
        if CameraManagerSingleton.__instance == None:
            CameraManagerSingleton(mode, path)
        return CameraManagerSingleton.__instance

    def __init__(self, mode, path):
        if mode == "camera":
            self.camera = cv2.VideoCapture(0)
        else:
            self.camera = cv2.VideoCapture(path)

        CameraManagerSingleton.__instance = self

    def getFrame(self):
        """Reads frame by frame from camera and resize it to specific sizes"""
        ret, frame = self.camera.read()

        frame_shape = frame.shape

        if ret is True:
            if frame_shape[0] > 480 or frame_shape[1] > 640:
                resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                self.frame = resized
            else:
                self.frame = frame
            return self.frame

    def closeCamera(self):
        self.camera.release()

    def show(self, image=None):
        if image is None:
            cv2.imshow("image", self.frame)
        else:
            cv2.imshow("img", image)


def Test_CamManagerVideoRec():

    # initialize cameraManager
    camManager = CameraManagerSingleton.getInstance()
    
    # initialize videoManager
    vidManager = VideoManagerSingleton.getInstance()

    # specify a value for maximum frames
    counter = 0

    while counter != 200:

        # capture frame from camera
        frame = camManager.getFrame()

        # send it to videoRecorder
        vidManager.record(frame, True)
        counter += 1

    # test if singleton works
    anotherVideo = VideoManagerSingleton.getInstance()

    # save the video
    anotherVideo.save(True)
    camManager.closeCamera()

def Test_CamManagerWrapper():

    camManager = CameraManagerSingleton.getInstance()
    wrapper = VideoManagerWrapper()
    counter = 0

    wrapper.start()

    while counter != 200:
        frame = camManager.getFrame()
        wrapper.record(frame)
        counter += 1

    wrapper.stop()
    camManager.closeCamera()

# Test_CamManagerWrapper()
# Test_CamManagerVideoRec()
