import urllib.request
from firebase import firebase

import csv
import time
import random


def test_internet_connection():
    try:
        urllib.request.urlopen("http://google.com")  # Python 3.x
        return True
    except:
        return False


class AlerterLogger:
    def __init__(self):
        self.csv_file = open("alerter_log.csv", "w+")
        self.writer = csv.writer(self.csv_file)
        self.firebase_link = (
            "https://pita-13817-default-rtdb.europe-west1.firebasedatabase.app/"
        )
        self.firebase_app = firebase.FirebaseApplication(self.firebase_link, None)

    def _test_internet(self):
        try:
            urllib.request.urlopen("http://google.com")  # Python 3.x
            return True
        except:
            return False

    def add_data(self, mode, time, danger, speed, lat, lon):
        self.writer.writerow([mode, time, danger, speed, lat, lon])

    def upload_data(self):

        self.csv_file.close()
        self.csv_file = open("alerter_log.csv", "r")
        reader = csv.reader(self.csv_file)

        # first test internet connection
        if self._test_internet() is True:

            # read saved data from file
            # upload it to firebase
            _firebase = firebase.FirebaseApplication(self.firebase_link, None)

            for row in reader:
                # print(row)
                data = {
                    "time": row[1],
                    "speed": row[3],
                    "danger": row[2],
                    "type": row[0],
                    "lat":row[4],
                    "lon":row[5]
                }
                _firebase.post("drivignInfos/", data)

        else:
            # send kivy alert
            print("No internet connection")

    def fast_upload(self, alert_type, speed, timestamp, danger, lat, lon):
        
        data = {
            "time":timestamp,
            "speed":speed,
            "type":alert_type,
            "danger":danger,
            "lat":lat,
            "lon":lon
        }
        print("Adding data to firebase")
        self.firebase_app.post("drivingInfos/", data)
        self.firebase_app.post("backup_driver_infos", data)
        print("Data added to firebase")
        
def test():

    dbManager = AlerterLogger()

    for i in range(0, 45):
        speed = random.randint(30, 40)
        id = random.randint(0, 3)
        danger = False
        if id % 2 == 0:
            danger = False
        else:
            danger = True
        dbManager.add_data(speed, id, time.time(), danger)
        time.sleep(0.1)

    dbManager.upload_data()


# test()
