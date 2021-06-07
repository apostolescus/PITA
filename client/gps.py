"""
Module responsible for GPS location and speed.
GPS used was L76X.
The code is taken from their officialy git repository
(https://github.com/waveshare/L76X-GPS-Module) and modified.

Allows both real functioning and mocking.
To mock location and speed set mocking=True in config.file.
Modify fix_speed, fix_lat, fix_lon to desired values.
"""
import time
import random
from queue import Full, Queue, Empty

from storage import StoppableThread, config_file
from storage import speed_screen_queue, gps_queue
import L76X


class GPS(StoppableThread):
    def __init__(self, name):
        self._speed: int = 30
        self._lat: float = 0.0
        self._lon: float = 0.0

        self._mocking = config_file["GPS"].getboolean("mocking")

        if self._mocking:
            self._speed = config_file["GPS"].getint("fix_speed")
            self._lat = config_file["GPS"].getfloat("fix_lat")
            self._lon = config_file["GPS"].getfloat("fix_lon")
        else:
            # set gps connection
            self._gps = L76X.L76X()
            self._gps.L76X_Set_Baudrate(9600)
            self._gps.L76X_Send_Command(self._gps.SET_NMEA_BAUDRATE_115200)
            time.sleep(2)
            self._gps.L76X_Set_Baudrate(115200)
            self._gps.L76X_Send_Command(self._gps.SET_POS_FIX_800MS)
            self._gps.L76X_Send_Command(self._gps.SET_NMEA_OUTPUT)
            self._gps.L76X_Exit_BackupMode()

        StoppableThread.__init__(self, name=name)

    def run(self) -> None:

        while not self.stopevent.isSet():

            if not self._mocking:
                self._gps.L76X_Gat_GNRMC()

            # get updated coordinates
            speed = self.get_speed()
            coordinates = self.get_coordinates()
            coordinates[0] = round(coordinates[0], 4)
            coordinates[1] = round(coordinates[1], 4)

            try:
                speed_screen_queue.put_nowait(speed)
            except Full:
                pass

            try:
                gps_queue.put_nowait((speed, coordinates[0], coordinates[1]))
            except Full:
                pass

    def get_coordinates(self) -> [float, float]:
        lat = self._gps.Lat
        lon = self._gps.Lon
        return [lat, lon]

    def get_speed(self) -> int:
        return self._gps.Speed


def test_gps():

    gps = GPS("gps_thread").start()

    while True:
        try:
            fetch = gps_queue.get_nowait()
            print("Fetched: ", fetch)
        except Empty:
            pass


test_gps()
