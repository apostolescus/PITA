import cv2

class CameraManagerSingleton:

    __instance = None
    @staticmethod
    def getInstance():
        if CameraManagerSingleton.__instance == None:
            CameraManagerSingleton()
        return CameraManagerSingleton.__instance
    
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        CameraManagerSingleton.__instance = self

    def getFrame(self):
        ret, frame = self.camera.read()
        # cv2.imshow("image", frame)
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