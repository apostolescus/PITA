""" Part of the server module which analyse the context and determine if the driver is in danger."""

import time
from threading import Thread
from firebase import firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

from storage import get_car_by_index
from storage import get_weather_by_index, alerter_priority
from storage import FrictionCoefficient, Constants, RecordStorage, logger
from video_manager_wrapper import VideoManagerWrapper

# don't generate alert for specific objects
non_alert_list = ["car", "bus", "truck", "moto"]


class Update:
    """
    Class used for updates of the alert coefficients.
    """

    def __init__(
        self,
        record_mode=None,
        car_type=None,
        weather=None,
        experience=None,
        reaction_time=None,
    ):

        self.car_type = car_type
        self.weather = weather
        self.experience = experience
        self.reaction_time = reaction_time
        self.record_mode = record_mode


class Alerter:
    """
    Class used to calculate maximum braking distance, create, upload to firebase and
    add to user response alert if necessary.
    """

    def __init__(
        self,
        client_uid,
        car_type="standard_stock",
        weather_type="dry_asphalt",
        reaction_time=1,
    ):
        friction_coef = "FrictionCoefficient." + car_type + "." + weather_type
        friction_coef = eval(friction_coef)

        self.video_manager = VideoManagerWrapper.getInstance()
        self.multiply = FrictionCoefficient.formula.multiplier / friction_coef
        self.reaction_time = reaction_time * Constants.km_to_h
        self.recording_mode = 0
        self.alert = {}
        self.record_delay = 0
        self.recording = False
        self.started = False
        self.update_obj = None
        self.last_detected = None
       
        self._client_uid = client_uid
        self.alert["alert"] = "none"
        self.alert["priority"] = 0
        self.alert["time"] = time.time()

        # load firebase admin credentials and initialize application
        cred = credentials.Certificate("./pita-13817-firebase-adminsdk-z75j0-29674c9af0.json")
        firebase_admin.initialize_app(cred,{
            'databaseURL':'https://pita-13817-default-rtdb.europe-west1.firebasedatabase.app/'
            })

        self._client_db = db.reference("users/" + self._client_uid)
        self._backup_db = db.reference("backup/users/" + self._client_uid)

        # if "13817" not in firebase_admin._apps:
        #     firebase_admin.initialize_app(cred,{
        #     'databaseURL':'https://pita-13817-default-rtdb.europe-west1.firebasedatabase.app/'
        #     })

    def update(self, update):
        """
        Method that updates values of the parameters used to calculate
        the maximum braking distance.
        """

        if update.experience == 0:
            add_reaction_time = 1.2
        elif update.experience == 1:
            add_reaction_time = 0.7
        else:
            add_reaction_time = 0.3

        car_type = get_car_by_index(update.car_type)
        weather_type = get_weather_by_index(update.weather)
        reaction = update.reaction_time + add_reaction_time

        # get friction coefficient
        friction_coef = "FrictionCoefficient." + car_type + "." + weather_type
        friction_coef = eval(friction_coef)

        # update constants based on friction coefficient
        self.multiply = FrictionCoefficient.formula.multiplier / friction_coef
        self.reaction_time = reaction * Constants.km_to_h

        record_mode = update.record_mode

        if RecordStorage.mode != record_mode or RecordStorage.recording is False:
            if RecordStorage.recording:
                # save current video
                self.video_manager.stop()

            if record_mode != 3:
                RecordStorage.mode = record_mode
                self.video_manager.start()

        logger.log("ALERTER", "Data succesfully updated")

    def check_safety(self, detection_results, gps_infos):
        """
        Method that checks if the driver is at a safe distance.
        If not returns a danger signal and client alerts the driver.
        """

        detected_results = detection_results

        # get speed from GPS
        gps_infos = gps_infos

        speed = gps_infos[0]
        lat = gps_infos[1]
        lon = gps_infos[2]

        if detected_results:
            detected = detected_results.detected

            if detected:
                distances = detected_results.frontal_distances

                detected_results.alert = ""

                # if there is any detected object that was in the ROI
                if len(distances) >= 1:

                    dictionary = distances.items()
                    sorted_distances = sorted(dictionary)

                    max_distance = self._calculate_max_distance(speed)
                    detected_results.safe_distance = max_distance

                    for object_distance in sorted_distances:
                        if max_distance > object_distance[0]:

                            logger.log(
                                "ALERTER",
                                "Danger detected, more than 80% procent overlaping",
                            )

                            detected_results.danger = 1

                            #uploading to firebase
                            x = Thread(
                                target=self._new_upload_to_firebase,
                                args=(
                                    "frontal_collision",
                                    speed,
                                    time.time(),
                                    1,
                                    lat,
                                    lon,
                                ),
                            )
                            x.start()

                            if "frontal_collision" != self.alert["alert"]:
                                self.alert["alert"] = "frontal_collision"
                                self.alert["time"] = time.time()
                                self.alert["priority"] = 99
                                detected_results.alert = "frontal_collision"

                            elif time.time() - self.alert["time"] > 1:
                                self.alert["time"] = time.time()

                            if not RecordStorage.start_smart:
                                logger.log("ALERTER", "Smart Record Started")
                                RecordStorage.start_smart = True
                                self.record_delay = time.time()
                            else:
                                self.record_delay = time.time()

                            return

                # add alerts to response alert list
                # an object is added to an alert list only if it passed more than 4 sec
                # since the last notification

                for detected_object in detected_results.detected_objects:
                    if detected_object.label not in non_alert_list:

                        # get alert priority
                        try:
                            priority = alerter_priority[detected_object.label]
                        except KeyError:
                            priority = 1

                        # if it's of bigger priority than the current alert
                        if priority > self.alert["priority"]:
                            self.alert["alert"] = detected_object.label
                            self.alert["time"] = time.time()
                            self.alert["priority"] = priority

                            #upload to firebase
                            x = Thread(
                                target=self._new_upload_to_firebase,
                                args=(
                                    detected_object.label,
                                    speed,
                                    time.time(),
                                    0,
                                    lat,
                                    lon,
                                ),
                            )
                            x.start()

                            # add alert to the client response
                            detected_results.alert = self.alert["alert"]

                        # if it's the same alert update time
                        elif priority == self.alert["priority"]:
                            self.alert["time"] = time.time()

                        # if it's lower priority check if the time has expired
                        elif time.time() - self.alert["time"] > 1:
                            self.alert["alert"] = detected_object.label
                            self.alert["time"] = time.time()
                            self.alert["priority"] = priority

                            #upload to firebase
                            x = Thread(
                                target=self._new_upload_to_firebase,
                                args=(
                                    detected_object.label,
                                    speed,
                                    time.time(),
                                    0,
                                    lat,
                                    lon,
                                ),
                            )
                            x.start()

                            # add alert to the client response
                            detected_results.alert = self.alert["alert"]

        # check if it is still recording for more than 2 seconds
        if RecordStorage.start_smart and (time.time() - self.record_delay > 2):
            RecordStorage.start_smart = False
            logger.log("ALERTER", "Smart Record Stopped")

    def _calculate_max_distance(self, speed):
        """
        Method that calculates maximum safe distance given a speed.
        Speed is calculated in km/h.
        """

        braking_distance = (speed * speed) * self.multiply
        reaction_distance = speed * self.reaction_time

        return braking_distance + reaction_distance
    
    def _new_upload_to_firebase(self, alert_type, speed, timestamp, danger, lat, lon):

        data = {
            "time": timestamp,
            "speed": speed,
            "type": alert_type,
            "danger": danger,
            "lat": lat,
            "lon": lon,
        }

        self._client_db.push(data)
        self._backup_db.push(data)



def test_upload_to_firebase():

    import random

    alerters = [
        "priority",
        "keep-right",
        "stop",
        "curve-right",
        "parking",
        "curve-left",
        "no-entry",
        "pedestrians",
        "give-way",
        "bike",
        "bus",
        "car",
        "person",
        "motorbike",
        "green",
        "red",
        "red-left",
        "truck",
    ]

    alerter = Alerter("q3MyVhuNnZR7c7zaqJk73UM980J2")

    for i in range(0, 10):
        alert = random.choice(alerters)
        current_time = time.time()
        speed = 190
        danger = 0
        lat = 43.45
        lon = 46.76
        alerter._new_upload_to_firebase(alert, speed, current_time, danger, lat, lon)

#test_upload_to_firebase()
