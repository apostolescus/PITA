from threading import Thread, Event, Lock
import configparser
import loguru

# load configuration file
config_file = configparser.ConfigParser()
config_file.read("config.file")

lock = Lock()
close = False
switch_sound = False
update_message = False

# declaration of points used for
# lane and collision detection
np_lines = []
poly_lines = []

# used to display times for all the processes
timer = config_file["DEBUG"].getboolean("time")
debug_mode = config_file["DEBUG"].getboolean("verbose")

width = config_file["VIDEO"].getint("width")
height = config_file["VIDEO"].getint("height")

# initialize logger
logger = loguru.logger
logger.add("server_log.txt", level="DEBUG", format="{time}{level}{message}")

logger.level("IMAGE_DETECTOR", no=20, color="<blue>")
logger.level("ALERTER", no=20, color="<magenta>")
logger.level("VIDEO", no=15, color="<green>")
logger.level("SERVER", no=16, color="<cyan>")
logger.level("LANE_DETECTOR", no=16, color="<yellow>")

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


class StoppableThread(Thread):
    def __init__(self, name):
        # super(StoppableThread, self).__init__(*args, **kwargs)
        Thread.__init__(self, name=name)
        self.stopevent = Event()

    def stop(self):
        self.stopevent.set()

    def join(self):
        print("Thread" + self.name + " is stopping")
        self.stopevent.set()
        Thread.join(self)


class RecordStorage:
    recording = False
    mode = 0
    start_smart = False


class DetectedPipeline:
    def __init__(self, image):

        self.image = image
        self.detected: bool = False
        self.detected_objects = None
        self.danger: int = 0
        self.alert: str = ""
        self.frontal_objects = None
        self.frontal_distances = None
        self.line_array = None
        self.safe_distance = 0

    def get_serializable(self):
        return None

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
    
    return "high_performance"


def get_weather_by_index(index):

    if index == 0:
        return "dry_asphalt"
    elif index == 1:
        return "wet_asphalt"
    elif index == 2:
        return "snow"

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


def set_poly_lines(poly, nump):
    global poly_lines, np_lines
    poly_lines = poly
    np_lines = nump


def get_poly_lines(name):
    if name == "poly":
        return poly_lines
    else:
        return np_lines