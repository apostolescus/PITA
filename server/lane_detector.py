import cv2
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from storage import get_polygone

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

class LaneDetector():
    def __init__(self):
        self.avg_left = None
        self.avg_right = None
        self.counter_left = 0
        self.counter_right = 0
        self.car_center = 0
        self.car_center_counter = 0
        self.correcting = False

    def __canny(self,image):
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        canny = cv2.Canny(blur, 100, 150)
        return canny

    def _convert_grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def __region_of_interest(self, image):
      
        polygons = get_polygone("np")
        polygons = np.array([polygons])
       
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
                #print("using average left")
                return np.array([left_line, self.avg_right])
            else:
                return np.array([left_line])

        elif len(right_fit) >0:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = self.__make_coordinates(image, right_fit_average, 'right')

            if self.counter_left > 20:
                #print("using average right")
                return np.array([self.avg_left, right_line])
            return np.array([right_line])
        else:
            return []
    
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

        gray_image = self._convert_grayscale(lane_image)
        canny_image = self.__canny(gray_image)

        cropped_image = self.__region_of_interest(canny_image)

        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

        if lines is not None:
            averaged_lines = self.__averaged_slope_intercept(lane_image, lines)

            # if self.correcting is False:
            #     if self.car_center_counter >= 10000:
            #         self.correcting = True

            #     if len(averaged_lines) == 2:
                    
            #         center = (self.avg_left[0] + self.avg_right[0]) / 2
            #         self.car_center_counter += 1
            #         self.car_center = (self.car_center*(self.car_center_counter-1) + center) / self.car_center_counter
            
            # if len(averaged_lines) > 2:
                
            #     center = ((averaged_lines[0][0] + averaged_lines[1][0]) /2)
            #     if abs(center - self.car_center)/ self.car_center > 0.1:
            #         print("More than 10% error... possible switching lanes")
            return averaged_lines

            #line_image = self.__display_lines(lane_image, averaged_lines)
            #return line_image
        else:
            return []
        
class AdvancedLaneDetector():
    
    def __init__(self, weights_path):

        self.weights_path = weights_path
        self.CFG = parse_config_utils.lanenet_cfg
        self.LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.net = lanenet.LaneNet(phase="test", cfg=self.CFG)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=self.CFG)

        #sess config
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = self.CFG.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = self.CFG.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        
        self.sess = tf.Session(config = sess_config)

        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                self.CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()    

        #saver
        self.saver = tf.train.Saver(variables_to_restore)

    def _minmax_scale(self, input_arr):
        """

        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr
        
    def detect(self, image):
        
        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path = self.weights_path)
            start_time = time.time()
            
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 -1
            
            first_step = time.time()
            #loop_times = 500
            #for i in range(loop_times):
            binary_seg_image, instance_seg_image = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],
            feed_dict={self.input_tensor:[image]})
            # t_cost = time.time() - first_step
            # t_cost /= loop_times

            # np.savetxt('binary_seg_image.out', binary_seg_image, delimiter = ',')
            # np.savetxt('instance_seg_image', instace_seg_image, delimiter=',')

           
            a = instance_seg_image
            name = "Instance segmentation"
            print((name + " dim: "), a.ndim)
            print((name + " datatype: "), a.dtype)
            print((name + " shape: "), a.shape)
            second_step = time.time()

            postprocess_result = self.postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image_vis
            )

            third_step = time.time()

            mask_image = postprocess_result["mask_image"]

            for i in range(self.CFG.MODEL.EMBEDDING_FEATS_DIMS):
                instance_seg_image[0][:, :, i] = self._minmax_scale(instance_seg_image[0][:, :, i])
            embedding_image = np.array(instance_seg_image[0], np.uint8)

            end_time = time.time()
            
            # print("Image processing time: " + str(first_step - start_time))
            # print("First run: " + str(second_step - first_step))
            # print("Average cost: ", t_cost)
            # print("Post process: " + str(third_step - second_step))
            # print("Total time: " + str(end_time-start_time))
            mask_img = mask_image[: ,:, (2,1,0)]

            # print("Mask dimensions: ", mask_img.ndim)
            # print("Mask datatype: ", mask_img.dtype)
            # print("Mask shape: ", mask_img.shape)

            cv2.imwrite("mask.png", mask_image)

            #print("Mask type: ", type(mask_image))
           


            binary_segmentation_image = binary_seg_image[0]*255
            # print("Binary dimensions: ", binary_segmentation_image.ndim)
            # print("BInary datatype: ", binary_segmentation_image.dtype)
            # print("BInary shape: ", binary_segmentation_image.shape)
            cv2.imwrite("binary.png", binary_segmentation_image)

            embedded_image = embedding_image[:, :, (2,1,0)]
            plt.figure('embeded_image')
            plt.imshow(image_vis[:, :, (2,1,0)])
            plt.savefig('source_image.png')
            cv2.imwrite("embedded.png", embedded_image)

            src_image = image_vis[:, :, (2,1,0)]
            cv2.imwrite("src_image.png", src_image)


def test():
    laneDet = LaneDetector()

    cap = cv2.VideoCapture("good.mp4")
    #cap = cv2.VideoCapture(0)
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

def advanced_test():

    image = cv2.imread("resized_image.png")
    weights_path = "./BiseNetV2_LaneNet_Tusimple_Model_Weights/tusimple_lanenet.ckpt"
    load_start = time.time()
    advanced_lane_det = AdvancedLaneDetector(weights_path)
    load_end = time.time()

    print("Total load time: ", load_end - load_start)
    advanced_lane_det.detect(image)



# test()
