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
distance_queue = Queue(2)

switch_sound = False
update_message = False

config_file = configparser.ConfigParser()
config_file.read("config.file")

width = config_file["VIDEO"].getint("width")
height = config_file["VIDEO"].getint("height")

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

def load_polygone_lines():
    '''Loads points that build the detection triangle from
    configuration files.

    Returns two lists, one for lane detection(np) 
    and other for intersection calculation(poly).'''

    # if type == "poly":
    #     return  [
    #                 (width - 530, height - (height - 50)),
    #                 (width / 2 - 15, height - 200),
    #                 (width - 120, height - (height - 50)),
    #             ]
    # elif type == "np":
    #     return[
    #         (width - 530, height-50),
    #         (int(width/2) - 15, 200),
    #         (width - 120, height-50)
    #     ]

    l1 = config_file["FRAME"]["h1"]
    l2 = config_file["FRAME"]["h2"]
    l3 = config_file["FRAME"]["h3"]

    l11,l12 = l1.split(",")
    l21, l22 = l2.split(",")
    l31, l32 = l3.split(",")

    l1 = (width + int(l11), height + int(l12))
    l2 = (int(width/2) + int(l21), int(l22))
    l3 = (width + int(l31), height + int(l32))

    np_val = []
    np_val.append(l1)
    np_val.append(l2)
    np_val.append(l3)

    l1_poly = (width + int(l11), -int(l12))
    l3_poly = (width + int(l31), -int(l32))

    poly_val = []
    poly_val.append(l1_poly)
    poly_val.append(l2)
    poly_val.append(l3_poly)

    return np_val, poly_val

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
