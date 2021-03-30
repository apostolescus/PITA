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

    def _test_internet(self):
        try:
            urllib.request.urlopen("http://google.com")  # Python 3.x
            return True
        except:
            return False

    def add_data(self, speed, object_id, timestamp, danger):
        self.writer.writerow([timestamp, speed, object_id, danger])

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
                    "time": row[0],
                    "speed": row[1],
                    "danger": row[3],
                    "object": row[2],
                }
                _firebase.post("drivignInfos/", data)

        else:
            # send kivy alert
            print("No internet connection")


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
