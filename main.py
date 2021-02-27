from CameraManager import CameraManagerSingleton
from ImageDetector import ImageDetector
import cv2
from threading import Thread
from queue import Queue
from Alerter import Alerter
from GPS import GPS
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from Storage import UISelected

check = False
update = False

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
        self.capture = CameraManagerSingleton().getInstance()
        Clock.schedule_interval(self.update, 1.0 / 30)

    def update(self, dt):
        global captured_image_queue, result_queue
        
        frame = self.capture.getFrame()
        try:
            captured_image_queue.put_nowait(frame)
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
        else:
            temp_var = 2

        UISelected.rec_mode = temp_var

    def update_settings(self):
        global update
        update = True
        

class CameraApp(App):

    global check

    def build(self):
        self.title = "PITA"
        return Display()
    
    def start_vid(self):
        
        print("Video started")

    def stop_vid(self):
        print("Video stopped")

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.closeCamera()

class GUIManagerThread(Thread):
    def run(self):
        gui = CameraApp()
        gui.run()

# Thread wrapper for ImageDetector.
# Performs image objct detection and measures
# distance to detected object.

class ImageObjectDetectorThread(Thread):
    def run(self):
        global captured_image_queue, second_queue
        imageDetector = ImageDetector("yolo-files/yolov4-tiny.weights", "yolo-files/yolov4-tiny.cfg")

        while True:
            result = captured_image_queue.get()
            results, distance_dict = imageDetector.detect(result)
            detected_object_queue.put((results, distance_dict))

# class LaneDetectorThread(Thread):
#     def run(self):
#         global second_queue, third_queue

#         while True:
#             results = detected_object_queue.get()
#             analyzed_detection_queue.put(results)

# Thread wrapper for Alert.
# Analyzes detected object and generates alerts
# based on speed, current weather, and current speed

class AlertThread(Thread):
    def run(self):
        global third_queue, result_queue, detected_object_queue, update
        gps = GPS()
        gps.start()
        
        alerter = Alerter()
        while True:
            # results = analyzed_detection_queue.get()
            if update is True:
                alerter.update()
                update = False
            results = detected_object_queue.get()
            distances = results[1]
            
            alerter.check_safety(distances)

            result_queue.put(results[0])
            
if __name__ == '__main__':
    
   

    # if __name__ == '__main__':
    GUIManagerThread().start()
    ImageObjectDetectorThread().start()
    #LaneDetectorThread().start()
    
    alerter = AlertThread()
    alerter.setName("alerter")
    alerter.start()