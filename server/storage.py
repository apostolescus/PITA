import sys
from threading import Thread, Event, Lock
import copy
import configparser
import loguru

config_file = configparser.ConfigParser()
config_file.read("server.config")

lock = Lock()
close = False
switch_sound = False
update_message = False

# used to display times for all the processes
timer = config_file["DEBUG"].getboolean("time")
debug_mode = config_file["DEBUG"].getboolean("verbose")

width = 640
height = 480

# initialize logger
logger = loguru.logger
logger.add(
    'server_log.txt', level="DEBUG", format="{time}{level}{message}"
)

logger.level("IMAGE_DETECTOR", no=20, color="<blue>")
logger.level("ALERTER", no=20, color="<magenta>")
logger.level("VIDEO", no=15,color="<green>")
logger.level("SERVER", no=16, color="<cyan>")
logger.level("LANE_DETECTOR", no=16, color="<yellow>")

def get_polygone(type):
    if type == "poly":
        return  [
                    (width - 530, height - (height - 50)),
                    (width / 2 - 15, height - 200),
                    (width - 120, height - (height - 50)),
                ]
    elif type == "np":
        return[
            (width - 530, height-50),
            (int(width/2) - 15, 200),
            (width - 120, height-50)
        ]
            
class DetectedPipeline:
    def __init__(self, image):

        self.image = image
        self.detected = False
        self.detected_objects = None
        self.danger = 0
        self.alerts = []
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
        ice = 0.08

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

    car_type = 0  # 0 stock, 1 truck 2 bus 3 sport
    weather = 0  # 0 dry 1 wet 2 snow 3 ice
    experience = 0  # 0 biginner 1 intermediate 2 advanced
    rec_mode = 0  # 0 smart mode 1 permanent 2 fix-size
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
        print("Thread" + self.name + " is stopping")
        self.stopevent.set()
        Thread.join(self)


class RecordStorage:
    recording = False
    mode = 0
    start_smart = False
