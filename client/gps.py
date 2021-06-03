from threading import Lock
import time
import random
from queue import Full

from storage import StoppableThread
from storage import speed_screen_queue, gps_queue


time_rate = 0.01


class GPS(StoppableThread):
    def __init__(self, name):
        self.speed: int = 50
        self.lock = Lock()
        StoppableThread.__init__(self, name=name)

    def increase_speed(self) -> None:
        self.speed = self.speed + 1

    def decrease_speed(self) -> None:
        self.speed = self.speed - 1

    def update_speed(self):
        number = random.randint(-5,4)
        
        self.speed += number

        while self.speed <= 0:
            number = random.randint(0,3)
            self.speed += number
            
    def run(self) -> None:

        while not self.stopevent.isSet():
            
            speed = self.get_speed()
            coordinates = self.get_coordinates()

            try:
                speed_screen_queue.put_nowait(speed)
            except Full:
                pass
            # gps_update_infos(speed, coordinates[0], coordinates[1])
            try:
                gps_queue.put_nowait((speed, coordinates[0], coordinates[1]))
            except Full:
                pass
            
            self.update_speed()
            time.sleep(0.5)
            # time.sleep(time_rate)

    def get_coordinates(self) -> [float, float]:
        lat = round(random.uniform(45.33, 46.66), 5)
        lon = round(random.uniform(24.67, 25.82), 5)
        return [lat, lon]

    def get_speed(self) -> int:
        # self.speed = self.speed + random.randint(-1,1)
        # while self.speed < 20 or self.speed > 90:
        #     self.speed = self.speed + random.randint(-2,2)
        return self.speed
