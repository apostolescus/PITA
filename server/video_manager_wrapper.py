import threading
import cv2

from video_manager import VideoManagerSingleton
from storage import RecordStorage, logger


class VideoManagerWrapper:
    """
        Singleton thread safe VideoManager wrapper.
    Mediates acces between GUI, other threads and low level VideoManager.
    Use getInstance() method.
    """

    __instance = None

    @staticmethod
    def getInstance():
        if VideoManagerWrapper.__instance is None:
            VideoManagerWrapper.__instance = VideoManagerWrapper()
        return VideoManagerWrapper.__instance

    def __init__(self):
        self.vm = VideoManagerSingleton.getInstance()
        self.lock = threading.Lock()

    def safe_enter(self):
        self.lock.acquire()

    def safe_exit(self):
        self.lock.release()

    def record(self, frame):
        self.safe_enter()

        # if it is suposed to rec
        if RecordStorage.recording:
            # if smart mode is enable
            if RecordStorage.mode == 0:
                # and objcct is detected close enough
                if RecordStorage.start_smart:
                    logger.level("VIDEO", "Smart mode Started")
                    self.vm.record(frame, True)
            # if permanent mode enable
            elif RecordStorage.mode == 1:
                self.vm.record(frame, True)
            # if fix size mode enable
            elif RecordStorage.mode == 2:
                self.vm.record(frame, False)

        self.safe_exit()

    def start(self):
        self.safe_enter()
        logger.level("VIDEO", "Video Manager Started recording")
        RecordStorage.recording = True

        self.vm.set_name()

        self.safe_exit()

    def stop(self):
        logger.level("VIDEO", "Video Manager Stopped recording")
        self.safe_enter()

        check = True

        RecordStorage.recording = False

        if RecordStorage.mode == 2:
            check = False
        else:
            check = True

        # start a new thread to save the video
        save_thread = threading.Thread(target=self.vm.save, args=(check,))
        save_thread.start()

        logger.level("VIDEO", "Video Saved")
        self.safe_exit()

    def start_smart(self):

        self.safe_enter()
        RecordStorage.start_smart = True

        self.safe_exit()

    def stop_smart(self):

        self.safe_enter()
        RecordStorage.start_smart = False

        self.safe_exit()


def test():
    camera = cv2.VideoCapture(0)

    videoWrapper = VideoManagerWrapper.getInstance()

    videoWrapper.start()
    RecordStorage.mode = 1
    counter = 0

    while counter != 100:
        ret, frame = camera.read()
        videoWrapper.record(frame)
        counter += 1

    videoWrapper.stop()
    camera.release()


# test()
