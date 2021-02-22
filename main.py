from CameraManager import CameraManagerSingleton
from ImageDetector import ImageDetector
import cv2
from threading import Thread
from queue import Queue

captured_image_queue = Queue(1)
detected_object_queue = Queue(1)
analyzed_detection_queue = Queue(1)
result_queue = Queue(1)

class CameraManagerThread(Thread):
    def run(self):
        global captured_image_queue, result_queue
        cameraManager = CameraManagerSingleton.getInstance()

        while True:
            frame = cameraManager.getFrame()

            try:
                captured_image_queue.put_nowait(frame)
            except:
                pass

            try:  
                img = result_queue.get_nowait()
               
                cameraManager.show(img)
            except:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                cameraManager.closeCamera()
                break
            #cameraManager.show()

class ImageObjectDetectorThread(Thread):
    def run(self):
        global captured_image_queue, second_queue
        imageDetector = ImageDetector("yolo-files/yolov4-tiny.weights", "yolo-files/yolov4-tiny.cfg")

        while True:
            result = captured_image_queue.get()
            results = imageDetector.detect(result)
            detected_object_queue.put(results)

class LaneDetectorThread(Thread):
    def run(self):
        global second_queue, third_queue

        while True:
            results = detected_object_queue.get()
            analyzed_detection_queue.put(results)

class AlertThread(Thread):
    def run(self):
        global third_queue, result_queue
        while True:
            results = analyzed_detection_queue.get()
            result_queue.put(results)
            

# if __name__ == '__main__':
CameraManagerThread().start()
ImageObjectDetectorThread().start()
LaneDetectorThread().start()
AlertThread().start()