
import cv2
import time
from loguru import logger

class VideoManagerSingleton:
    __instance = None
    
    @staticmethod 
    def getInstance(file_name="record.avi", time=600, mu = False, frame_size = (540, 640), FPS=30):
        if VideoManagerSingleton.__instance == None:
            VideoManagerSingleton(file_name, time, mu, frame_size, FPS)
        return VideoManagerSingleton.__instance

    def __init__(self, file_name, time, mu, frame_size, FPS):
        
        self.file_name = file_name
        self.time = time
        
        #mu False -  secunde
        if mu is False:
            self.buffer_len = self.time * FPS
        else:
            self.buffer_len = self.time * FPS * 60
        self.buffer = None
        self.counter = 0
        self.check =  False

        self.frame_size = frame_size
        self.writer = cv2.VideoWriter(self.file_name, cv2.VideoWriter_fourcc(*'MJPG'), 
						FPS, self.frame_size)
        VideoManagerSingleton.__instance = self

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
                for i in range(0, self.buffer_len):
                    self.writer.write(self.buffer[i])
            else:
                for i in range(self.counter, self.buffer_len):
                    self.writer.write(self.buffer[i])

                for i in range(0, self.counter):
                    self.writer.write(self.buffer[i])

        self.writer.release()
        