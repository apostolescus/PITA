# from globals import get_speed, record_mode, switch_sound
from storage import DetectedPipeline
from storage import get_car_by_index, get_driver_level_by_index
from storage import get_weather_by_index
from storage import FrictionCoefficient, Constants, UISelected, RecordStorage

# from VideoManager import VideoManagerSingleton
from video_manager_wrapper import VideoManagerWrapper

# from globals import RecordStorage, start_rec
from alert_logger import AlerterLogger
import logging
import numpy as np
import time
import threading
import cv2


class Update:
    def __init__(
        self,
        mode,
        record_mode=None,
        car_type=None,
        weather=None,
        experience=None,
        reaction_time=None,
    ):

        self.mode = mode

        if mode == 1:
            self.car_type = car_type
            self.weather = weather
            self.experience = experience
            self.reaction_time = reaction_time
            self.record_mode = record_mode

class Alerter:
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
        self.recording = False
        self.started = False
        self.update_obj = None
        self.danger_area = danger_area
        self.last_detected = None
        self.alert_logger = AlerterLogger()

    def update(self, update):

        mode = update.mode

        if mode == 1:
            print("Full update mode")
            car_type = get_car_by_index(update.car_type)
            weather_type = get_weather_by_index(update.weather)
            reaction = update.reaction_time + update.experience

            friction_coef = "FrictionCoefficient." + car_type + "." + weather_type
            friction_coef = eval(friction_coef)

            self.multiply = FrictionCoefficient.formula.multiplier / friction_coef
            self.reaction_time = reaction * Constants.km_to_h

            record_mode = update.record_mode

            if RecordStorage.mode != record_mode or RecordStorage.recording is False:
                if RecordStorage.recording:
                    # save current video
                    print("Already recording...")
                    self.video_manager.stop()

                if record_mode != 3:
                    print("Video manager starting...")
                    RecordStorage.mode = record_mode
                    self.video_manager.start()

        print("DATA SUCCESSFULLY UPDATED")

    def draw_image(self, image, detected_bbx, lines=None):
        # height = image.shape[0]

        # pts = np.array([[(340, height-150), (920, 550), (1570, height-150)]],
        #        np.int32)

        if len(detected_bbx) != 0:

            detected_objects = detected_bbx[1]
            labels = detected_bbx[2]
            colors = detected_bbx[3]
            # distances = detected_bbx[4]
            # objects_ids = detected_bbx[5]

            self.last_detected = detected_objects
            # self.colors = colors
            # self.labels = labels

            for obj in detected_objects:
                x, y = obj.bbx[0], obj.bbx[1]
                w, h = obj.bbx[2], obj.bbx[3]

                color = [int(c) for c in colors[obj.id]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

                text = "{}: {:.4f}".format(labels[obj.id], obj.score)
                cv2.putText(
                    image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        else:
            if self.last_detected:
                for obj in self.last_detected:
                    x, y = obj.bbx[0], obj.bbx[1]
                    w, h = obj.bbx[2], obj.bbx[3]

                    # color = [int(c) for c in self.colors[obj.id]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

                    # text = "{}: {:.4f}".format(self.labels[obj.id], obj.score)
                    # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if lines is not None:
            image = cv2.addWeighted(image, 1, lines, 0.5, 1)
        return image

    def check_safety(self, detected_result):

        safe = True
       
        # self.video_manager.record(detected_result[0])
        detected_results = detected_result[1]

        if detected_results:
            detected = detected_results.detected

            if detected:
                distances = detected_results.frontal_distances

                if len(distances) >= 1:
                    dictionary = distances.items()
                    sorted_distances = sorted(dictionary)

                    # get speed from GPS
                    gps_infos = detected_result[2]

                    speed = gps_infos[0]
                    lat = gps_infos[1]
                    lon = gps_infos[2]

                    max_distance = self._calculate_max_distance(speed)

                    for i in sorted_distances:
                        if max_distance > i[0]:
                            safe = False

                            # self.alert_logger.add_data(speed, i, time.time(), True)
                            print("Danger, more than 80% precent overlaping")
                            detected_results.danger = 1

                            # uploading to firebase
                            self.alert_logger.fast_upload("frontal_colision", speed, time.time(), 1, lat, lon)

                            if not RecordStorage.start_smart:
                                RecordStorage.start_smart = True

                            # make warning sound

                            return True

                            # 

                    # if now in safe state and in smart mode stop recording
        if safe and RecordStorage.start_smart:
            RecordStorage.start_smart = False

    def _calculate_max_distance(self, speed):
        # speed is in km/h

        d = (speed * speed) * self.multiply
        reaction_distance = speed * self.reaction_time

        return d + reaction_distance

    def update_alert_logger(self):
        self.alert_logger.upload_data()


# def test():
#     alert = Alerter()

#     for i in range(34,67):
#         d = alert._calculate_max_distance(i)
#         print("max distance is: ", d)

# test()
