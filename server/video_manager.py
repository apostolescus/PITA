from threading import Lock
from datetime import datetime
import cv2
from storage import config_file

class VideoManagerSingleton:

    """
    Singleton class manages low level image operations.
    Used with a wrapper above.

    time : by default 180 secs
    mu : False if time is in seconds, else True
    frame_size : modify depending on the camera
    FPS : modify depending on the frame size you obtain after camera processing
    """

    __instance = None

    # modify default frame size (depends on camera)
    @staticmethod
    def getInstance(
        time=10,
        mu=False,
        # VERY IMPORTANT, VIDEO SAVING WON'T WORK WITHOUT
        # PROPER RESOLUTION
        frame_size=(config_file["VIDEO"].getint("width"), (config_file["VIDEO"].getint("height")),
        FPS=8,
    ):

        if VideoManagerSingleton.__instance == None:
            VideoManagerSingleton(time, mu, frame_size, FPS)
        return VideoManagerSingleton.__instance

    def __init__(self, time, mu, frame_size, FPS):

        self.directory_name = "recordings/"
        self.time = time
        self.lock = Lock()
        self.FPS = FPS
        self.writer = None

        # mu is false time is in seconds
        if mu is False:
            self.buffer_len = self.time * FPS
        # else is in minutes
        else:
            self.buffer_len = self.time * FPS * 60
        self.buffer = None
        self.counter = 0
        self.check = False
        self.frame_size = frame_size

        VideoManagerSingleton.__instance = self

    def set_name(self):

        now = datetime.now()
        out_file = now.strftime("%Y%m%d_%H:%M") + ".avi"
        out_file = self.directory_name + out_file
        self.writer = cv2.VideoWriter(
            out_file, cv2.VideoWriter_fourcc(*"MJPG"), self.FPS, self.frame_size
        )

    def record(self, frame, long=False):

        if self.buffer is None:
            self.buffer = [frame] * self.buffer_len

        if long is False:
            self.buffer[self.counter] = frame
            self.counter += 1

            if self.counter == self.buffer_len:
                self.counter = 0
                self.check = True
        else:
            self.writer.write(frame)

    def save(self, long=False):

        if long is False:
            if self.check is False:
                for i in range(0, self.counter):
                    self.writer.write(self.buffer[i])
            else:
                for i in range(self.counter, self.buffer_len):
                    self.writer.write(self.buffer[i])

                for i in range(0, self.counter):
                    self.writer.write(self.buffer[i])

        self.writer.release()


def test():

    camera = cv2.VideoCapture(0)
    video_manager = VideoManagerSingleton.getInstance()

    counter = 0
    video_manager.set_name()

    while counter != 100:
        ret, frame = camera.read()

        video_manager.record(frame, long=True)
        counter += 1

    video_manager.save(True)
    camera.release()


# test()
