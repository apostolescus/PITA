from threading import Lock
from storage import StoppableThread, gps_update_infos
import random

time_rate = 0.01


class GPS(StoppableThread):
    def __init__(self, name):
        self.speed: int = 190
        self.lock = Lock()
        StoppableThread.__init__(self, name=name)

    def increase_speed(self) -> None:
        self.speed = self.speed + 5

    def decrease_speed(self) -> None:
        self.speed = self.speed - 5

    def run(self) -> None:

        while not self.stopevent.isSet():
            speed = self.get_speed()
            coordinates = self.get_coordinates()

            gps_update_infos(speed, coordinates[0], coordinates[1])
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
