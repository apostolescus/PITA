import os
import time
from threading import enumerate
from queue import Queue, Empty

# other libraries
import psutil
import cv2

# GUI libraries
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.config import Config

# import local dependencies
from camera_manager import CameraManagerSingleton
from storage import UISelected, StoppableThread
from storage import toggle_update_message, last_alert_queue
from storage import toggle_switch_sound, config_file, get_gps_infos
from storage import alerter_dictionary, alerter_color

# Queues for pipeline
captured_image_queue = Queue(1)
result_queue = Queue(1)

# global variables
switch = False

# update_speed
last_speed_update_time = 0
SPEED_UPDATE_INTERVAL = config_file["GPS"].getfloat("update_interval")

class Display(BoxLayout):
    def __init__(self, **kwargs):
        super(Display, self).__init__(**kwargs)


class Screen_One(Screen):
    """Main Screen, displays image and has basic configuration options"""

    def __init__(self, **kwargs):
        super(Screen_One, self).__init__(**kwargs)
        self.capture = CameraManagerSingleton.get_instance(config_file)

        # schedule update
        update_interval = int(config_file["VIDEO"]["update"])
        Clock.schedule_interval(self.update, 1.0 / update_interval)

    def update(self, dt):
        """ Captures image from cameraManager and puts it in pipeline"""
        global captured_image_queue, result_queue, last_speed_update_time

        frame = self.capture.get_frame()

        # put image in pipeline
        try:
            captured_image_queue.put_nowait(frame)
        except:
            pass
        
        # fetch and display alerts
        try:
            last_alert = last_alert_queue.get_nowait()

            # # get alert description from dictionary
            # try:
            #     alert_description = alerter_dictionary[last_alert]
            # except:
            #     alert_description = last_alert

            #self.ids.alert_label.text = last_alert

            # get coresponding alert color
            # try:
            #     alert_color = alerter_color[last_alert]
                
            #     # danger - red color
            #     if alert_color == 2:
            #         alert_color = [33, 210, 202,1]
            #     # pay attention - yellow color
            #     elif alert_color == 1:
            #         alert_color = [3, 3, 255,1]
            #     # informative detection - blue
            #     else:
            #         alert_color = [252, 3, 3, 1]
            # except:
            #     alert_color = [252, 3, 3, 1]
            
            # self.ids.alert_label.color = alert_color

        except Empty:
            pass
        
        # get speed each x seconds
        # if time.time() - last_speed_update_time > SPEED_UPDATE_INTERVAL:
        #     speed = get_gps_infos()[0]
        #     last_speed_update_time = time.time()

        #     if speed > 120:
        #         self.ids.speed.color = [33, 210, 202,1]
        #         self.ids.speed.text = str(speed)
        #     else:
        #         self.ids.speed.color = [3, 3, 255,1]
        #         self.ids.speed.text = str(speed)

        # display processed image
        try:
            img = result_queue.get_nowait()
            buf1 = cv2.flip(img, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
            )
            image_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")

            self.ids.imageView.texture = image_texture
        except:
            pass


class Screen_Two(Screen):
    """Class for managing UI for settings menu.
    ALlows for multiple fine tuning selection of parameters."""

    def on_spinner_select(self, text):

        temp_var = ""
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

        temp_var = ""
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

        temp_var = ""
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

        UISelected.updated = True
        toggle_update_message()


class CameraApp(App):
    def build(self):
        self.title = "PITA"
        return Display()

    def on_stop(self):
        # stop all the other threads
        for thread in enumerate():
            if thread.name != "guiThread" and thread.name != "MainThread":
                thread.join()

        current_system_pid = os.getpid()

        ThisSystem = psutil.Process(current_system_pid)
        ThisSystem.terminate()

    def switch_callback(self):
        global switch

        if switch is False:
            switch = True
            UISelected.lane_detection = True
            toggle_update_message()
        else:
            toggle_update_message()
            switch = False
            UISelected.lane_detection = False

    def sound_callback(self):
        toggle_switch_sound()

    def update_data(self):
        for thread in enumerate():
            if thread.name == "AlerterThread":
                thread.update_data()


class GUIManagerThread(StoppableThread):
    def run(self):
        Config.set('graphics', 'fullscreen', 'auto')
        Config.set('graphics', 'window_state', 'maximized')
        Config.write()
        gui = CameraApp()
        gui.run()
