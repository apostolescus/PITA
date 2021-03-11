from globals import get_speed, record_mode
from Storage import FrictionCoefficient, Constants, UISelected
from VideoManager import VideoManagerSingleton
from VideoManagerWrapper import VideoManagerWrapper
from globals import RecordStorage, start_rec
import logging
import cv2

class Alerter:
    def __init__(
        self,
        car_type="standard_stock",
        weather_type="dry_asphalt",
        reaction_time=1,
    ):
        friction_coef = "FrictionCoefficient." + car_type + "." + weather_type
        friction_coef = eval(friction_coef)
        
        self.videManager = VideoManagerWrapper.getInstance()
        self.multiply = FrictionCoefficient.formula.multiplier / friction_coef
        self.reaction_time = reaction_time * Constants.km_to_h
        self.sound_duration = Constants.sound_duration
        self.sound_freq = Constants.sound_freq
        self.recording_mode = 0
        self.recording = False
        self.started = False
        
    def update(self):
       
        tmp_var = ""
        tmp_weather = ""

        if UISelected.car_type == 0:
            tmp_var = "standard_stock"    
        elif UISelected.car_type == 1:
            tmp_var = "truck"
        elif UISelected.car_type == 2:
            tmp_var = "tourism"
        else:
            tmp_var = "high_performance"
        
        if UISelected.weather == 0:
            tmp_weather = "dry_asphalt"
        elif UISelected.weather == 1:
            tmp_weather = "wet_asphalt"
        elif UISelected.weather == 2:
            tmp_weather = "snow"
        else:
            tmp_weather = "ice"

        friction_coef = "FrictionCoefficient." + tmp_var+ "." + tmp_weather
        friction_coef = eval(friction_coef)

        if UISelected.rec_mode == 0:
            record_mode(0)
        elif UISelected.rec_mode == 1:
            record_mode(1)
        elif UISelected.rec_mode == 2:
            record_mode(2)
        else:
            record_mode(3)

        # print("permanent: ", RecordStorage.permanent)
        # print("smart: ", RecordStorage.smart)
        # print("fixed: ", RecordStorage.fix)
        # print("record: ", RecordStorage.record)
        # print("start_var rec " , RecordStorage.start_rec_var)

        self.multiply = FrictionCoefficient.formula.multiplier / friction_coef
        self.reaction_time = UISelected.reaction_time * Constants.km_to_h
        self.recording_mode = UISelected.rec_mode

    
    def draw_image(self, image, detected_bbx, lines=None):

        if len(detected_bbx) != 0 :
            boxes = detected_bbx[0]
            confidences = detected_bbx[1]
            classIDs = detected_bbx[2]
            idxs = detected_bbx[3]
            labels = detected_bbx[4]
            colors = detected_bbx[5]
            #distances = detected_bbx[6]

            for i in idxs.flatten():

                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])

                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if lines is not None:
            image = cv2.addWeighted(image, 1, lines, 0.5,1)
        return image

    def check_safety(self, distances):
        
        safe = True
        dictionary = distances.items()
        sorted_distances = sorted(dictionary)

        speed = get_speed()
        #print("speed is: ", speed)
        max_distance = self._calculate_max_distance(speed)
        #print("max safe distance is: ", max_distance, "current distances: ", sorted_distances)
        for i in sorted_distances:
            if max_distance > i[0]:
                safe = False
                # make warning sound
                #print("Alert")
                #if smart mode start recording
                if RecordStorage.smart is True and self.started is False:
                    #print("started smart recording")
                    self.videManager.start_smart()
                    self.started = True
                # print(
                #     "max distance: ", max_distance, "current distance: ", i[0]
                # )
        #if now in safe state and in smart mode stop recording
        if safe is True and RecordStorage.smart is True and self.started is True:
            print("stopping smart recording")
            self.videManager.stop_smart()
            self.started = False

    # speed is in km/h
    def _calculate_max_distance(self, speed):
        d = (speed * speed) * self.multiply
        reaction_distance = speed * self.reaction_time

        return d + reaction_distance


# def test():
#     alert = Alerter()

#     for i in range(34,67):
#         d = alert._calculate_max_distance(i)
#         print("max distance is: ", d)

# test()
