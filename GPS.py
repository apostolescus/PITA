from ImageDetector import ImageDector
import time

class GPS:
    __instance = None

    @staticmethod
    def getInstance():
        if GPS.__instance == None:
            GPS()
        return GPS.__instance

    def __init__(self, mode = False, imageDetector = None):
        if GPS.__instance != None:
            print("This is a singleton")
        else:
            GPS.__instance = self

        if mode == False:
            self.detector = imageDetector
        self.current_speed = 0
        
    def get_speed(self):
        return self.current_speed

    def _get_distance(self):
        d = 0
        prev_d = 0
        start = 0
        while True:
            
            d = self.detector.last_distance()

            if d != prev_d != 0:
                
                end = time.time()
                v = abs(d - prev_d)/end-start
                prev_d = d
                self.current_speed = v
                print("speed is: ", v)
                start = time.time()

            else:
                start = time.time()
        
    


