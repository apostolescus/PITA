from threading import Thread, Event, Lock
import copy
import logging

lock = Lock()
close = False
switch_sound = False
update_message = False

class DetectedPipeline:

    def __init__(self, image):

        self.image = image
        self.detected = False
        self.detected_objects = None
        self.danger = 0
        self.frontal_objects = None
        self.frontal_distances = None
        self.line_array = None
    
    def get_serializable(self):
        return None

class FrictionCoefficient:

    class standard_stock:
        dry_asphalt = 1.3
        wet_asphalt = 0.8
        snow = 0.2
        ice = 0.1

    class truck:
        dry_asphalt = 0.8
        wet_asphalt = 0.55
        snow = 0.2
        ice = 0.1    

    class high_performance:
        dry_asphalt = 1
        wet_asphalt = 0.7
        snow = 0.15
        ice =  0.08      

    class tourism:
        dry_asphalt = 0.9
        wet_asphalt = 0.6
        snow = 0.2
        ice = 0.1
    
    class formula:
        multiplier = 0.003914

def get_car_by_index(index):

    if index == 0:
        return "standard_stock"
    elif index == 1:
        return "truck"
    elif index == 2:
        return "tourism"
    else:
        return "high_performance"

def get_weather_by_index(index):

    if index == 0:
        return "dry_asphalt"
    elif index == 1:
        return "wet_asphalt"
    elif index == 2:
        return "snow"
    else:
        return "ice"

def get_driver_level_by_index(index):

    time = 0

    if index == 0:
        time = 1
    elif index == 1:
        time = 0.6
    else:
        time = 0.2
    
    return time

class Constants:
    km_to_h = 0.277
    sound_duration = 1000
    sound_freq = 440

class UISelected:

    car_type = 0 # 0 stock, 1 truck 2 bus 3 sport
    weather = 0 # 0 dry 1 wet 2 snow 3 ice
    experience = 0 #0 biginner 1 intermediate 2 advanced
    rec_mode = 0 # 0 smart mode 1 permanent 2 fix-size
    reaction_time = 0.5
    lane_detection = False

def get_update_message():
    return update_message

def toggle_update_message():
    global update_message
    
    if update_message:
        print("Update message is False")
        update_message = False
    else:
        print("Update message is True")
        update_message = True

class StoppableThread(Thread):
    def __init__(self, name):
        self.stopevent = Event()
        Thread.__init__(self, name=name)

    def join(self):
        self.stopevent.set()
        Thread.join(self)

class RecordStorage:
    recording = False
    mode = 0
    start_smart = False
    
def initialize(): 
    global num 
    global speed
    global last_val

    speed = 30
    num = 1 
    last_val = 0

def record_mode(mode):
    
    print("rec mode for: ", mode)
    #0 smart mode + auto start
    if mode == 0:
        RecordStorage.record = False
        RecordStorage.smart = True
        RecordStorage.fix = False
        RecordStorage.save = True
        RecordStorage.permanent = False
    #permanent mode
    elif mode == 1:
        RecordStorage.record = False
        RecordStorage.permanent = True
        RecordStorage.fix = False
        RecordStorage.smart = False
        RecordStorage.save = True
    # fix sized
    elif mode == 2:
        RecordStorage.record = False
        RecordStorage.permanent = False
        RecordStorage.fix = True
        RecordStorage.save = True
        RecordStorage.smart = False
    # no record
    else:
        RecordStorage.record = False
        RecordStorage.permanent = False
        RecordStorage.fix = False
        RecordStorage.smart = False
        RecordStorage.save = False
    #print("Record value record_mode: ", RecordStorage.record)
    return True

def start_rec(check = False, start = False, stop = False):

    if stop is True:
        RecordStorage.start_rec_var = False
        return

    if start is True:
        RecordStorage.start_rec_var = True
        return
    
    if check is True:
        return RecordStorage.start_rec_var

def modify_speed(val):
    global speed
    lock.acquire()
    speed = val
    lock.release()

def get_speed():
    global speed
    # lock.acquire()
    # a = copy.deepcopy(speed)
    # lock.release()

    return speed
    
def stop_program():
    global close
    close = True

def should_stop():
    lock.acquire()
    if close is False:
        lock.release()
        return False
    lock.release()
    return True
