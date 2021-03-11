import cv2
import numpy as np

class LaneDetector():
    def __init__(self, accuracy=0.5, advanced_lane=True):
        self.accuracy = accuracy
        self.advanced_lane = advanced_lane
        self.avg_left = None
        self.avg_right = None
        self.counter_left = 0
        self.counter_right = 0

    def __canny(self,image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        canny = cv2.Canny(blur, 100, 150)
        return canny

    def __region_of_interest(self, image):
        height = image.shape[0]
        polygons = np.array([
                                [(340, height-150), (920, 550), (1570, height-150)]
                            ])

        # polygons = np.array([
        #                         [(250, height-50), (635,420), (1100, height-50)]
        #                     ])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def __make_coordinates(self, image, line_parameters, line):
       
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1*(3/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)

        return_value = np.array([x1, y1, x2, y2])

        if self.counter_left < 10000 and line == 'left':
            if self.avg_left is not None:
                self.avg_left = (self.avg_left*self.counter_left + return_value)/(self.counter_left+1)
                self.counter_left +=1
            else:
                self.avg_left = return_value
        
        if self.counter_right < 10000 and line == 'right':
            if self.avg_right is not None:
                self.avg_right = (self.avg_right*self.counter_right + return_value)/(self.counter_right+1)
                self.counter_right +=1
            else:
                self.avg_right = return_value
        return return_value

    def __averaged_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []
        
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < 0:
                if slope < -0.5:
                    left_fit.append((slope, intercept))  
            else:
                if slope > 0.5:
                    right_fit.append((slope, intercept))
                
        if len(left_fit) > 0 and len(right_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            right_fit_average = np.average(right_fit, axis=0)
            
            left_line = self.__make_coordinates(image, left_fit_average, 'left')
            right_line = self.__make_coordinates(image, right_fit_average, 'right')
            return np.array([left_line, right_line])

        elif len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = self.__make_coordinates(image, left_fit_average, 'left')

            if self.counter_right > 20:
                print("using average left")
                return np.array([left_line, self.avg_right])
            else:
                return np.array([left_line])

        elif len(right_fit) >0:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = self.__make_coordinates(image, right_fit_average, 'right')

            if self.counter_left > 20:
                print("using average right")
                return np.array([self.avg_left, right_line])
            return np.array([right_line])
        else:
            return
    
    def __display_lines(self, image, lines):
        line_image = np.zeros_like(image)

        if lines is not None:
            for x1, y1, x2, y2 in lines:
                try:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 190, 255), 11)
                except:
                    return None
        return line_image

    def detect_lane(self, image):
        
        lane_image = np.copy(image)
        canny_image = self.__canny(lane_image)

        cropped_image = self.__region_of_interest(canny_image)

        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

        if lines is not None:
            averaged_lines = self.__averaged_slope_intercept(lane_image, lines)

            line_image = self.__display_lines(lane_image, averaged_lines)

            return line_image
        else:
            return None

def test():
    laneDet = LaneDetector()

    cap = cv2.VideoCapture("good.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        lines = laneDet.detect_lane(frame)
        if lines is not None:
            combo = cv2.addWeighted(frame, 1, lines, 0.5,1)
            cv2.imshow("res", combo)
        else:
            cv2.imshow("res", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


