from VideoManager import VideoManagerSingleton
from Storage import RecordStorage
import threading
import cv2


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
        try:
            # print("Record Storage recording: ", RecordStorage.recording)
            # print("Record Mode: ", RecordStorage.mode)
            # if it is suposed to rec
            if RecordStorage.recording:
                # if smart mode is enable
                if RecordStorage.mode == 0:
                    # and objcect is detected close enough
                    if RecordStorage.start_smart:
                        self.vm.record(frame, True)
                # if permanent mode enable
                elif RecordStorage.mode == 1:
                    print("Recording permanent mode")
                    self.vm.record(frame, True)
                # if fix size mode enable
                elif RecordStorage.mode == 2:
                    self.vm.record(frame, False)
        except Exception as e:
            print("EXCEPTIOOON: ", e)
        self.safe_exit()

    def start(self):
        self.safe_enter()
        print("Video manager start")
        RecordStorage.recording = True

        self.vm.set_name()

        self.safe_exit()

    def stop(self):
        print("Video manager stopping")
        self.safe_enter()

        check = True

        RecordStorage.recording = False

        if RecordStorage.mode == 2:
            check = False
        else:
            check = True
        print("Check is: ", check)
        # start a new thread to save the video
        # save_thread = threading.Thread(target=self.vm.save, args=(check,))
        # save_thread.start()
        self.vm.save(check)

        print("Saved finished")
        self.safe_exit()

    def start_smart(self):

        self.safe_enter()
        RecordStorage.start_smart = True
        #elf.start_rec = True
        self.safe_exit()

    def stop_smart(self):

        self.safe_enter()
        RecordStorage.start_smart = False
        #self.start_rec = False
        self.safe_exit()

    def update(self):

        self.safe_enter()

        self.global_record = RecordStorage.record
        self.smart = RecordStorage.smart
        self.start_rec = RecordStorage.start_rec_var
        self.permanent = RecordStorage.permanent
        self.fixed = RecordStorage.fix

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