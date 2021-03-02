
import cv2
import time
from loguru import logger
from threading import Lock
from datetime import datetime

class VideoManagerSingleton:
    __instance = None
    
    #modify default frame size (depends on camera)
    @staticmethod 
    def getInstance(file_name="recording/test_test.avi", time=30, mu = False, frame_size = (640, 480), FPS=10):
        if VideoManagerSingleton.__instance == None:
            VideoManagerSingleton(file_name, time, mu, frame_size, FPS)
        return VideoManagerSingleton.__instance

    def __init__(self, file_name, time, mu, frame_size, FPS):
        
        self.file_name = file_name
        self.directory_name = "recordings/"
        self.time = time
        self.lock = Lock()
        self.FPS = FPS
        
        if mu is False:
            self.buffer_len = self.time * FPS
        else:
            self.buffer_len = self.time * FPS * 60
        self.buffer = None
        self.counter = 0
        self.check =  False
        self.frame_size = frame_size

        VideoManagerSingleton.__instance = self

    def set_name(self):
        
        now = datetime.now()
        out_file = now.strftime("%Y%m%d_%H:%M") + ".avi"
        out_file = self.directory_name + out_file
        self.writer = cv2.VideoWriter(out_file,  cv2.VideoWriter_fourcc(*'MJPG'), self.FPS, self.frame_size)

    def record(self, frame, long = False):
        
        if self.buffer is None:
            self.buffer = [frame]*self.buffer_len
        
        if long is False:
            self.buffer[self.counter] = frame
            self.counter += 1
        
            if self.counter ==  self.buffer_len:
                self.counter = 0
                self.check = True

        else:
            self.writer.write(frame)
        

    def save(self, long = False):
        
        
        if long is False:
            if self.check is False:
                
                for i in range(0, self.counter):
                    self.writer.write(self.buffer[i])    
            else:
                
                for i in range(self.counter, self.buffer_len):
                    self.writer.write(self.buffer[i])

                for i in range(0, self.counter):
                    self.writer.write(self.buffer[i])
        self.writer.release()
        

def test():
    camera = cv2.VideoCapture(0)
    videoManager = VideoManagerSingleton.getInstance()

    counter = 0

    while counter != 200:
        ret, frame = camera.read()
        videoManager.record(frame, long = True)
        counter +=1

    videoManager.save(True)
    camera.release()

#test()