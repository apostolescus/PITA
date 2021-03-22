from CameraManager import CameraManagerSingleton
from ImageDetector import ImageDetector
from VideoManager import VideoManagerSingleton
from VideoManagerWrapper import VideoManagerWrapper
import cv2
from LaneDetector import LaneDetector
from globals import RecordStorage, record_mode, stop_program, should_stop, StoppableThread
from threading import *
from queue import Queue, Empty
from Alerter import Alerter
from GPS import GPS
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from Storage import UISelected
import logging
import sys
import time

switch = False
switch_sound = False
check = False
update = False
record = False
stop = False
videoManager = VideoManagerWrapper.getInstance()

mode = 0

# Queues for pipeline
captured_image_queue = Queue(1)
detected_object_queue = Queue(1)
analyzed_detection_queue = Queue(1)
result_queue = Queue(1)

# Thread wrappering for CameraManager.
# Captures frames from camera, sends them in pipeline
# and displays the result.

class Display(BoxLayout):

    def __init__(self, **kwargs):
        super(Display, self).__init__(**kwargs)

class Screen_One(Screen):

    def __init__(self, **kwargs):
        
        super(Screen_One, self).__init__(**kwargs)
        self.capture = CameraManagerSingleton.getInstance(mode)
        Clock.schedule_interval(self.update, 1.0 / 30)

    def update(self, dt):
        global captured_image_queue, result_queue, stop
        
        frame = self.capture.getFrame()
        #put image in pipeline and send it to video recorder
        try:
            captured_image_queue.put_nowait(frame)
            videoManager.record(frame)
        except:
            pass

        try:  
            img = result_queue.get_nowait()
            buf1 = cv2.flip(img, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            
            self.ids.imageView.texture = image_texture
        except:
            pass

class Screen_Two(Screen):
   

    def on_spinner_select(self, text):
        
        temp_var = 0
        if text == "Standard":
            temp_var = 0
        elif text == "Camion":
            temp_var = 1
        elif text == "Autobuz":
            temp_var = 2
        else:
            temp_var = 3

        UISelected.car_type = temp_var

    def on_slider_change_value(self, text):
        UISelected.reaction_time = text

    #use this 
    def on_spinner_select_driver_experience(self, text):

        temp_var = 0
        if text == "Incepator":
            temp_var = 0
        elif text == "Mediu":
            temp_var = 1
        else:
            temp_var = 2

        UISelected.experience = temp_var

    def on_spinner_select_weather_type(self, text):

        temp_var = 0
        if text == "Uscat":
            temp_var = 0
        elif text == "Umed":
            temp_var = 1
        elif temp_var == "Zapada":
            temp_var = 2
        else:
            temp_var = 3

        UISelected.weather = temp_var

    def on_spinner_select_record_type(self, text):

        temp_var = 0
        if text == "Smart Mode":
            temp_var = 0
        elif text == "Permanent":
            temp_var = 1
        elif text == "Fix sized":
            temp_var = 2
        else:
            temp_var = 3

        UISelected.rec_mode = temp_var

    def update_settings(self):
        global update
        print("Updating settings")
        update = True    

class CameraApp(App):

    def build(self):
        self.title = "PITA"
        return Display()
    
    def start_vid(self):
       
       #start video recording
        videoManager.start()

    def stop_vid(self):

        #save the video
        videoManager.stop()

    def on_stop(self):
        videoManager.stop()

        for thread in enumerate():
            if thread.name != "MainThread" and thread.name != "guiThread":
                thread.join()
      
        
    def switch_callback(self):
        global switch, lane_detector

        # print("Switch value: ", switch)
        if switch is False:
            lane_detector = LaneDetectorThread("lane_thread")
            switch = True
            lane_detector.start()
        else:
            for thread in enumerate():
                if thread.name == "lane_thread":
                    thread.join()
            #lane_detector.join()
            switch = False

    def sound_callback(self):
        global switch_sound
        
        if switch_sound is False:
            switch_sound = True
        else:
            switch_sound = False

class GUIManagerThread(StoppableThread):
    def run(self):
        gui = CameraApp()
        gui.run()

# Thread wrapper for ImageDetector.
# Performs image objct detection and measures
# distance to detected object.

class ImageObjectDetectorThread(StoppableThread):
    

    def run(self):
        global captured_image_queue
        
        #imageDetector = ImageDetector("yolo-files/yolov4-tiny.weights", "yolo-files/yolov4-tiny.cfg")
        
        imageDetector = ImageDetector("yolo-files/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite", "yolo-files/yolov4-tiny.cfg", model = "tflite")

        while not self.stopevent.isSet():
            #close app
           
            read_image = captured_image_queue.get()
            bbx_details = imageDetector.detect(read_image)
            if switch is False:
                analyzed_detection_queue.put([read_image, bbx_details])
            else:
                detected_object_queue.put([read_image, bbx_details])

        print("Image detector stopped")


class LaneDetectorThread(StoppableThread):
    def run(self):

        lane_detector = LaneDetector()
        #lane_detector = LaneDetector("calibrate_camera.p")

        while not self.stopevent.isSet():

            try:
                proprStorage = detected_object_queue.get()
                image = proprStorage[0]
               
                lines = lane_detector.detect_lane(image)

                if lines is not None:
                    proprStorage.append(lines)
            
                analyzed_detection_queue.put(proprStorage)
            except:
                continue

# Thread wrapper for Alert.
# Analyzes detected object and generates alerts
# based on speed, current weather, and current speed

#update ReactionTime
class AlertThread(StoppableThread):
    
    def run(self):
        global third_queue, result_queue, detected_object_queue, update
        
        #starts GPS thread
        gps = GPS("GPSThread")
        gps.start()
        
        self.stop = False
        height = 1080
        alerter = Alerter([(340, height-150), (920, 550), (1570, height-150)])

        while not self.stopevent.isSet():

            if update is True:
                videoManager.stop()
                alerter.update()
                videoManager.update()
                update = False
            
            res = analyzed_detection_queue.get()
           
            if len(res[1]) != 0:
                alerter.check_safety(res[1], switch_sound)
            if len(res) == 3:
                lines = res[2]
            else:
                lines = None
            drawn_image = alerter.draw_image(res[0], res[1], lines)
            result_queue.put(drawn_image)

            
        print("Alerter stopped")

if __name__ == '__main__':
    
    imageDetector = ImageObjectDetectorThread("imageDetector")
    guiManager= GUIManagerThread("guiThread")
    alerter = AlertThread("AlerterThread")
    #lane = LaneDetectorThread("laneThread")

    #lane.start()
    imageDetector.start()
    guiManager.start()
    alerter.start()


    #LaneDetectorThread().start()
    
    
    
    
