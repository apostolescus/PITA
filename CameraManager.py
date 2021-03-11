import cv2
from threading import Lock
from VideoManager import VideoManagerSingleton
from VideoManagerWrapper import VideoManagerWrapper

class CameraManagerSingleton:

    _lock = Lock()
    __instance = None
    @staticmethod
    def getInstance(mode, path="project_video.mp4"):
        if CameraManagerSingleton.__instance == None:
            CameraManagerSingleton(mode, path)
        return CameraManagerSingleton.__instance
    
    def __init__(self, mode, path):
        if mode == "1":
            self.camera = cv2.VideoCapture(0)
        else:
            self.camera = cv2.VideoCapture(path)
        CameraManagerSingleton.__instance = self

    def getFrame(self):
        
        ret, frame = self.camera.read()

        if ret is True:
            self.frame = frame
            
            return frame
        

    def closeCamera(self):
        self.camera.release()
    
    def show(self, image = None):
        if image is None:
            cv2.imshow("image", self.frame)
        else:
            cv2.imshow("img", image)

def Test_CamManagerVideoRec():
    camManager = CameraManagerSingleton.getInstance()
    vidManager = VideoManagerSingleton.getInstance()
    counter = 0
    
    while counter != 200:
        frame = camManager.getFrame()
        vidManager.record(frame, True)
        counter += 1

    anotherVideo = VideoManagerSingleton.getInstance()
    anotherVideo.save(True)
    camManager.closeCamera()

def Test_CamManagerWrapper():

    camManager = CameraManagerSingleton.getInstance()
    wrapper = VideoManagerWrapper()
    counter = 0
    
    wrapper.start()

    while counter != 200:
        frame = camManager.getFrame()
        wrapper.record(frame)
        counter += 1

    wrapper.stop()
    camManager.closeCamera()

#Test_CamManagerWrapper()
#Test_CamManagerVideoRec()

