""" File use for storing and sharing common variable between client classes"""
from threading import Thread, Event, Lock
import configparser
from queue import Queue
import loguru

lock = Lock()
gps_lock = Lock()
gps_queue = Queue()
last_alert_queue = Queue(1)

switch_sound = False
update_message = False

config_file = configparser.ConfigParser()
config_file.read("config.file")

# alerter dictionary
alerter_dictionary = {
    "frontal_collision": "Frontal Colision",
    "priority": "Priority Sign",
    "keep-right": "Keep-Right Sign",
    "stop": "Stop",
    "curve-right": "Curve-Right",
    "parking": "Parking Sign Detected",
    "curve-left": "Curve-Left",
    "no-entry": "No entry sign detected",
    "pedestrians": "Pedestrians sign detected",
    "give-way": "Give way sign detected",
    "bike": "Bike",
    "bus": "Bus",
    "car": "Car",
    "person": "Person",
    "motorbike": "Motorbike",
    "green": "Green semaphore",
    "red": "Red semaphore",
    "red-left": "Red semaphore left",
    "truck": "Truck",
}

alerter_color = {
    "frontal_collision": 2,
    "priority": 0,
    "keep-right": 1,
    "stop": 2,
    "curve-right": 1,
    "parking": 0,
    "curve-left": 1,
    "no-entry": 2,
    "pedestrians": 1,
    "give-way": 2,
    "bike": 0,
    "bus": 0,
    "car":0,
    "person":0,
    "motorbike": 0,
    "green": 0,
    "red": 2,
    "red-left": 2,
    "truck": 0,
}

alerter_priority = {"frontal_collision": 100, "stop": 98, "red": 99, "no-entry": 97}

# store last gps data
old_gps_value: list = [0, 0, 0]
old_gps_lat: float = 0
old_gps_lon: float = 0

# initialize loguru
logger = loguru.logger
logger.add(
    config_file["LOGGER"]["file"], level="DEBUG", format="{time}{level}{message}"
)


class UISelected:
    """Class used for storing environmental
    variables and updating server information"""

    # 0 stock, 1 truck 2 bus 3 sport
    car_type = config_file["CONFIGURATION"].getint("car_type")
    # 0 dry 1 wet 2 snow 3 ice
    weather = int(config_file["CONFIGURATION"]["weather"])
    # 0 biginner 1 intermediate 2 advanced
    experience = int(config_file["CONFIGURATION"]["experience"])
    # 0 smart mode 1 permanent 2 fix-size
    rec_mode = int(config_file["CONFIGURATION"]["rec_mode"])
    reaction_time = float(config_file["CONFIGURATION"]["reaction_time"])
    lane_detection = False
    updated = True


class StoppableThread(Thread):
    """ Implements thread join of Thread class"""

    def __init__(self, name):
        self.stopevent = Event()
        Thread.__init__(self, name=name)

    def join(self):
        self.stopevent.set()
        Thread.join(self)


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


def gps_update_infos(speed: int, lat: float, lon: float) -> None:

    gps_lock.acquire()
    gps_queue.put((speed, lat, lon))
    gps_lock.release()


def get_gps_infos() -> [int, float, float]:

    global old_gps_value
    infos = []

    gps_lock.acquire()
    infos = gps_queue.get()
    gps_lock.release()

    return infos
