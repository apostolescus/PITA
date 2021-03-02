from VideoManager import VideoManagerSingleton
from globals import RecordStorage, record_mode
import threading


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
        self.global_record = RecordStorage.record
        self.smart = RecordStorage.smart
        self.start_rec = False
        self.permanent = RecordStorage.permanent
        self.fixed = RecordStorage.fix
        self.lock = threading.Lock()

    def safe_enter(self):
        self.lock.acquire()

    def safe_exit(self):
        self.lock.release()

    def record(self, frame):
        self.safe_enter()

        # if it is suposed to rec
        if self.global_record is True:
            # if smart mode is enable
            if self.smart is True:
                # and objcect is detected close enough
                if self.start_rec is True:
                    self.vm.record(frame, True)
            elif self.permanent is True:
                self.vm.record(frame, True)
            elif self.fixed is True:
                self.vm.record(frame, False)
        self.safe_exit()

    def start(self):
        self.safe_enter()

        self.global_record = True
        self.vm.set_name()

        RecordStorage.record = True

        self.safe_exit()

    def stop(self):

        self.safe_enter()

        check = False

        if self.global_record is False:
            self.safe_exit()
            return

        self.global_record = False
        RecordStorage.record = False

        if self.fixed is True:
            check = False
        else:
            check = True

        # start a new thread to save the video
        save_thread = threading.Thread(target=self.vm.save, args=(check,))
        save_thread.start()

        self.safe_exit()

    def start_smart(self):

        self.safe_enter()
        self.start_rec = True
        self.safe_exit()

    def stop_smart(self):

        self.safe_enter()
        self.start_rec = False
        self.safe_exit()

    def update(self):

        self.safe_enter()

        self.global_record = RecordStorage.record
        self.smart = RecordStorage.smart
        self.start_rec = RecordStorage.start_rec_var
        self.permanent = RecordStorage.permanent
        self.fixed = RecordStorage.fix

        # print("permanent: ", self.permanent)
        # print("smart: ", self.smart)
        # print("fixed: ", self.fixed)
        # print("record: ", self.global_record)
        # print("start_var rec " , self.start_rec)
        self.safe_exit()
