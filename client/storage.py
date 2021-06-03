""" File use for storing and sharing common variable between client classes"""
from threading import Thread, Event, Lock
import configparser
from queue import Queue
import loguru

lock = Lock()
gps_lock = Lock()

# initialise queues
gps_queue = Queue()
speed_screen_queue = Queue(2)
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
    "frontal_collision": [33, 210, 202, 1],
    "priority": [252, 3, 3, 1],
    "keep-right": [3, 3, 255, 1],
    "stop": [33, 210, 202, 1],
    "curve-right": [3, 3, 255, 1],
    "parking": [252, 3, 3, 1],
    "curve-left": [3, 3, 255, 1],
    "no-entry": [33, 210, 202, 1],
    "pedestrians": [3, 3, 255, 1],
    "give-way": [33, 210, 202, 1],
    "bike": [252, 3, 3, 1],
    "bus": [252, 3, 3, 1],
    "car": [252, 3, 3, 1],
    "person": [252, 3, 3, 1],
    "motorbike": [252, 3, 3, 1],
    "green": [252, 3, 3, 1],
    "red": [33, 210, 202, 1],
    "red-left": [33, 210, 202, 1],
    "truck": [252, 3, 3, 1],
}

alerter_priority = {
    "frontal_collision": 100,
    "stop": 98,
    "red": 99,
    "no-entry": 97,
    "give-way": 96,
    "person": 95,
    "keep-right": 94,
    "curve-left": 93,
    "curve-right": 93,
}

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
