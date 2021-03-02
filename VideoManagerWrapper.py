from VideoManager import VideoManagerSingleton
from globals import RecordStorage, record_mode
import threading

class VideoManagerWrapper():

    __instance = None

    @staticmethod
    def getInstance():
        if VideoManagerWrapper.__instance is None:
            VideoManagerWrapper.__instance =  VideoManagerWrapper()
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

        #if it is suposed to rec
        if self.global_record is True:
            #if smart mode is enable
            if self.smart is True:
                #and objcect is detected close enough
                if self.start_rec is True:
                    print("smart recording ...")
                    self.vm.record(frame, True)
            elif self.permanent is True:
                print("permanent recording frame...")
                self.vm.recod(frame, True)
            elif self.fixed is True:
                self.vm.record(frame, False)    
 
        self.safe_exit()

    def start(self):
        self.safe_enter()

        self.global_record = True
        self.vm.set_name()

        RecordStorage.record = True
        print("Started video recording ...")

        self.safe_exit()

    def stop(self):
        
        self.safe_enter()                                
        print("Video manager stopped")
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

        #start new thread to do the savings    
        save_thread = threading.Thread(target=self.vm.save, args=(check,))
        save_thread.start()
        
        self.safe_exit()

    def start_smart(self):
        self.safe_enter()
        print("Smart mode started")
        self.start_rec = True
        self.safe_exit()

    def stop_smart(self):
        self.safe_enter()
        print("Smart mode stopped")
        self.start_rec = False
        self.safe_exit()

    def update(self):
        
        self.safe_enter()

        self.global_record = RecordStorage.record
        self.smart = RecordStorage.smart
        self.start_rec = RecordStorage.start_rec_var
        self.permanent = RecordStorage.permanent
        self.fixed = RecordStorage.fix

        self.safe_exit()
        