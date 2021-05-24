from threading import Thread, Event, Lock
from queue import Queue
import copy
import logging

lock = Lock()
gps_lock = Lock()
gps_queue = Queue()

close = False
switch_sound = False
update_message = False

# store last gps data
old_gps_value:list = [0,0,0]
old_gps_lat: float = 0
old_gps_lon: float = 0

class UISelected:

    car_type = 0  # 0 stock, 1 truck 2 bus 3 sport
    weather = 0  # 0 dry 1 wet 2 snow 3 ice
    experience = 0  # 0 biginner 1 intermediate 2 advanced
    rec_mode = 0  # 0 smart mode 1 permanent 2 fix-size
    reaction_time = 0.5
    lane_detection = False
    updated = True

def get_update_message():
    return update_message

def toggle_update_message():
    global update_message

    if update_message:
        update_message = False
    else:
        update_message = True

def get_switch_sound():
    return switch_sound

def toggle_switch_sound():
    global switch_sound

    if switch_sound:
        switch_sound = False
    else:
        switch_sound = True

def gps_update_infos(speed:int, lat:float, lon:float)->None:

    gps_lock.acquire()

    # if gps_queue.full():
    #     gps_queue.get()

    gps_queue.put((speed, lat, lon))
   
    gps_lock.release()

def get_gps_infos()->[int, float, float]:

    global old_gps_value
    infos = []
   
    gps_lock.acquire()
    
    infos = gps_queue.get()
    # if gps_queue.full():
    #     infos = gps_queue.get()
    #     old_gps_value = infos
    # else:
    #     infos = old_gps_value

    gps_lock.release()

    return infos

class StoppableThread(Thread):
    def __init__(self, name):
        self.stopevent = Event()
        Thread.__init__(self, name=name)

    def join(self):
        self.stopevent.set()
        Thread.join(self)
