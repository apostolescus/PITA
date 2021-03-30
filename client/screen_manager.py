#GUI imports
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture

#import local dependencies
from camera_manager import CameraManagerSingleton
from Storage import UISelected, StoppableThread
from Storage import toggle_update_message, get_update_message

# other libraries
from queue import Queue, Empty
import cv2

# Queues for pipeline
captured_image_queue = Queue(1)
result_queue = Queue(1)

#global variables
switch = False

#debug
mode = "0"

class Display(BoxLayout):

    def __init__(self, **kwargs):
        super(Display, self).__init__(**kwargs)

class Screen_One(Screen):

    def __init__(self, **kwargs):
        
        super(Screen_One, self).__init__(**kwargs)
        self.capture = CameraManagerSingleton.getInstance(mode)
        Clock.schedule_interval(self.update, 1.0 / 30)

    def update(self, dt):
        global captured_image_queue, result_queue
        
        frame = self.capture.getFrame()
        #put image in pipeline and send it to video recorder
        try:
            captured_image_queue.put_nowait(frame)
            #videoManager.record(frame)
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
        print("selected text: ", text)
        if text == "Smart Mode":
            temp_var = 0
        elif text == "Permanent":
            temp_var = 1
        elif text == "Fix sized":
            temp_var = 2
        else:
            temp_var = 3
        
        print("Temp var: ", temp_var)
        UISelected.rec_mode = temp_var
        print("UISelected: ", UISelected.rec_mode)

    def update_settings(self):
        UISelected.updated = True
        toggle_update_message()

class CameraApp(App):

    def build(self):
        self.title = "PITA"
        return Display()
    
    def start_vid(self):
       
       #start video recording
        #videoManager.start()
        print("Start video rec")

    def stop_vid(self):
        print("Stop video rec")
        #save the video
        #videoManager.stop()

    def on_stop(self):
        #videoManager.stop()

        for thread in enumerate():
            if thread.name != "MainThread" and thread.name != "guiThread":
                thread.join()
      
        
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
        global switch_sound
        
        if switch_sound is False:
            switch_sound = True
        else:
            switch_sound = False

    def update_data(self):
        for thread in enumerate():
            if thread.name == "AlerterThread":
                thread.update_data()



class GUIManagerThread(StoppableThread):
    def run(self):
        gui = CameraApp()
        gui.run()
