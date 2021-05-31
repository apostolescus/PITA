""" Part of the server module which analyse the context and determine if the driver is in danger."""

import time

from firebase import firebase

from storage import get_car_by_index
from storage import get_weather_by_index
from storage import FrictionCoefficient, Constants, RecordStorage, logger
from video_manager_wrapper import VideoManagerWrapper

non_alert_list = ["car", "bus", "camion", "moto"]


class Update:
    """Class used for updates of the alert coefficients"""

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
    """Class used to calculate maximum braking distance, create, upload to firebase and
    add to user response alert if necessary"""

    def __init__(
        self,
        danger_area,
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
        self.alerts = {}
        self.record_delay = 0
        self.recording = False
        self.started = False
        self.update_obj = None
        self.danger_area = danger_area
        self.last_detected = None
        self.firebase_app = firebase.FirebaseApplication(
            "https://pita-13817-default-rtdb.europe-west1.firebasedatabase.app/", None
        )

    def update(self, update):
        """Method that updates values of the parameters used to calculate
        the maximum braking distance."""

        car_type = get_car_by_index(update.car_type)
        weather_type = get_weather_by_index(update.weather)
        reaction = update.reaction_time + update.experience

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

    def check_safety(self, detected_result):
        """Method that checks if the driver is at a safe distance.
        If not returns a danger signal and client alerts the driver"""

        detected_results = detected_result[1]

        # get speed from GPS
        gps_infos = detected_result[2]

        speed = gps_infos[0]
        lat = gps_infos[1]
        lon = gps_infos[2]

        if detected_results:
            detected = detected_results.detected

            if detected:
                distances = detected_results.frontal_distances

                detected_results.alerts.clear()

                # if there is any detected object that was in the ROI
                if len(distances) >= 1:

                    dictionary = distances.items()
                    sorted_distances = sorted(dictionary)

                    max_distance = self._calculate_max_distance(speed)

                    for object_distance in sorted_distances:
                        if max_distance > object_distance[0]:

                            logger.log(
                                "ALERTER",
                                "Danger detected, more than 80% procent overlaping"
                            )

                            detected_results.danger = 1

                            # uploading to firebase
                            self._upload_to_firebase(
                                "frontal_colision", speed, time.time(), 1, lat, lon
                            )

                            if "frontal_collision" in self.alerts:
                                if time.time() - self.alerts["frontal_collision"] > 2:
                                    self.alerts["frontal_collision"] = time.time()
                                    detected_results.alerts.append("frontal_collision")
                            else:
                                self.alerts["frontal_collision"] = time.time()

                            if not RecordStorage.start_smart:
                                logger.log("ALERTER","Smart Record Started")
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

                        # check if object in alert
                        if detected_object.label in self.alerts:
                            alert_time = self.alerts[detected_object.label]

                            if time.time() - alert_time > 4:

                                # upload alert to firebase
                                self._upload_to_firebase(
                                    detected_object.label,
                                    speed,
                                    time.time(),
                                    0,
                                    lat,
                                    lon,
                                )

                                # update alert timestamp
                                self.alerts[detected_object.label] = time.time()
                                detected_results.alerts.append(detected_object.label)

                        else:
                            self.alerts[detected_object.label] = time.time()

        # check if it is still recording for more than 2 seconds
        if RecordStorage.start_smart and (time.time() - self.record_delay > 2):
            logger.log("ALERTER", "Smart Record Stopped")
            RecordStorage.start_smart = False

    def _calculate_max_distance(self, speed):
        """Method that calculates maximum safe distance given a speed.
        Speed is calculated in km/h."""

        braking_distance = (speed * speed) * self.multiply
        reaction_distance = speed * self.reaction_time

        return braking_distance + reaction_distance

    def _upload_to_firebase(self, alert_type, speed, timestamp, danger, lat, lon):

        data = {
            "time": timestamp,
            "speed": speed,
            "type": alert_type,
            "danger": danger,
            "lat": lat,
            "lon": lon,
        }

        self.firebase_app.post("drivingInfos/", data)
        self.firebase_app.post("backup_driver_infos", data)
        logger.log("ALERTER", "Data added to firebase")


# def test():
#     alert = Alerter()

#     for i in range(34,67):
#         d = alert._calculate_max_distance(i)
#         print("max distance is: ", d)

# test()
