import time
from threading import Thread
from globals import modify_speed, should_stop, StoppableThread
import random

class GPS(StoppableThread):
    def __init__(self, name):
        self.speed = 30
        StoppableThread.__init__(self, name=name)

    def run(self):
    
        while not self.stopevent.isSet():
            self.get_speed()
            modify_speed(self.speed)
        print("GPS closed")

    def get_speed(self):
        # self.speed = self.speed + random.randint(-1,1)
        # while self.speed < 20 or self.speed > 90:
        #     self.speed = self.speed + random.randint(-5,5)
        return
        

    


