""" 
Part of the PITA module that performs lane detection.
Lane detection is performed using basic image processing teqniques.
(Class LaneDetector)

Implements also lane detection using trained model of CNN ( Advanced LaneDetector). 
Given the complexity and the computational requrirements necessary to run two neural network 
in same application real time it is used only basic lane detection.
"""
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from storage import  RecordStorage, get_poly_lines
from storage import config_file, timer, logger

# for testing only
from video_manager_wrapper import VideoManagerWrapper


class LaneDetector:
    """
    Class that performs lane detection on images.

    Preprocessin: gray scale transofrmation, gaussian
    blur.
    Applies canny edge detection and
    a Hough Transform to detect lines from canny results.

    Allows for virtual lane detection. This mechanism
    creates an averaged line that can be displayed when
    only the opposite line is detected.
    """

    def __init__(self):
        self.avg_left = None
        self.avg_right = None
        self.counter_left = 0
        self.counter_right = 0
        self.average_counter = 0

        # constant information from config file
        self.AVERAGE_LINES = config_file["LANE_DETECTION"].getboolean("lane_average")
        self.AVERAGE_AVAILABLE = config_file["LANE_DETECTION"].getint(
            "average_available"
        )
        self.AVERAGE_MAX = config_file["LANE_DETECTION"].getint("average_max")
        self.MAX_COUNTER = config_file["LANE_DETECTION"].getfloat("max_counter")

    def __canny(self, gray):
        """Applies gaussian blurs over the grayed image
        and detects edged using canny edge detector."""

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 100, 150)
        return canny

    def _convert_grayscale(self, image):
        """ Converts image to grayscale. """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def __region_of_interest(self, image):
        """Selects and crops a region of interest
        for searching lines."""

        polygons = get_poly_lines("other")
        polygons = np.array([polygons])

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)

        return masked_image

    def __make_coordinates(self, image, line_parameters, line) -> []:
        """
        Calculates plottable points coordinates relative to the image shape.
        Calculates average lines.

        Returns list (x,y) points pairs for each line."""

        slope, intercept = line_parameters

        # generate line points for display
        # relative to the screen
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return_value = np.array([x1, y1, x2, y2])

        # calculate average line coordinates for left line
        if self.counter_left < self.AVERAGE_MAX and line == "left":
            if self.avg_left is not None:
                self.avg_left = (self.avg_left * self.counter_left + return_value) / (
                    self.counter_left + 1
                )
                self.counter_left += 1
            else:
                self.avg_left = return_value

        # calculate average line coordinates for right line
        if self.counter_right < self.AVERAGE_MAX and line == "right":
            if self.avg_right is not None:
                self.avg_right = (
                    self.avg_right * self.counter_right + return_value
                ) / (self.counter_right + 1)
                self.counter_right += 1
            else:
                self.avg_right = return_value

        return return_value

    def __averaged_slope_intercept(self, image, lines):
        """
        Generate lines from points.
        Calculate slope and intercept.
        Filter lines based on slope.

        Perform average on line points with previous ones if possible.
        This increases the detection robustness.

        Returns list containing left and right lanes coordinates.
        """

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

        # make aveage of line points for more robustness

        # if both lines detected
        if len(left_fit) > 0 and len(right_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            right_fit_average = np.average(right_fit, axis=0)

            self.average_counter = 0

            left_line = self.__make_coordinates(image, left_fit_average, "left")
            right_line = self.__make_coordinates(image, right_fit_average, "right")
            
            return np.array([left_line, right_line]).astype(int)

        # only if left line detected
        # then use average right line

        elif len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = self.__make_coordinates(image, left_fit_average, "left")

            self.average_counter = 0

            # using average right line
            if self.counter_right > self.AVERAGE_AVAILABLE and self.AVERAGE_LINES:
                return np.array([left_line, self.avg_right]).astype(int)
            else:
                return np.array([left_line]).astype(int)

        # only right line detected
        # then use average left line

        elif len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = self.__make_coordinates(image, right_fit_average, "right")

            self.average_counter = 0

            # using average right line if the average was calculated on more than
            # average_avilable times and it is True in config file
            if self.counter_left > self.AVERAGE_AVAILABLE and self.AVERAGE_LINES:
                return np.array([self.avg_left, right_line]).astype(int)
            else:
                return np.array([right_line]).astype(int)
        else:
            if self.average_counter < self.MAX_COUNTER:
                self.average_counter += 1
                #print("NO line detected after slope, average counter: ", self.average_counter)
                if self.avg_right is not None and self.avg_left is not None:
                        return np.array([self.avg_left, self.avg_right]).astype(int)

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
        """
        Public method exported for lane detection.

        Returns a list containing left and right lane
        points coordinates."""

        lane_image = np.copy(image)
        if timer:
            start_time = time.time()

        gray_image = self._convert_grayscale(lane_image)
        canny_image = self.__canny(gray_image)

        cropped_image = self.__region_of_interest(canny_image)

        lines = cv2.HoughLinesP(
            cropped_image,
            2,
            np.pi / 180,
            100,
            np.array([]),
            minLineLength=40,
            maxLineGap=5,
        )

        if timer:
            end_time = time.time()
            print("Total time: ", end_time - start_time)

        if lines is not None:
            averaged_lines = self.__averaged_slope_intercept(lane_image, lines)

            if timer:
                print("After average: ", time.time() - start_time)

            return averaged_lines
        else:
            # if the average time option is aviable
            if self.average_counter < self.MAX_COUNTER:
                #print("NO line detected, average counter: ", self.average_counter)
                self.average_counter += 1
                if self.avg_right is not None and self.avg_left is not None:
                    return np.array([self.avg_left, self.avg_right]).astype(int)
            return []


def test():
    laneDet = LaneDetector()

    cap = cv2.VideoCapture("../video/good.mp4")

    video_wrapper = VideoManagerWrapper.getInstance()
    video_wrapper.start()
    RecordStorage.mode = 1

    counter = 0

    # cap = cv2.VideoCapture(0)
    while cap.isOpened():
        if counter == 500:
            video_wrapper.stop()

        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        lines = laneDet.detect_lane(frame)

        if lines is not None:
            line_image = np.zeros_like(frame)

            for x1, y1, x2, y2 in lines:
                try:
                    cv2.line(
                        line_image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 190, 255),
                        11,
                    )
                except:
                    pass

            combo = cv2.addWeighted(frame, 1, line_image, 0.5, 1)

            video_wrapper.record(combo)
            counter += 1

            cv2.imshow("res", combo)
        else:
            cv2.imshow("res", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# test()


# from lanenet_model import lanenet
# from lanenet_model import lanenet_postprocess
# from local_utils.config_utils import parse_config_utils
# from local_utils.log_util import init_logger


# class AdvancedLaneDetector:
#     def __init__(self, weights_path):

#         self.weights_path = weights_path
#         self.CFG = parse_config_utils.lanenet_cfg
#         self.LOG = init_logger.get_logger(log_file_name_prefix="lanenet_test")

#         self.input_tensor = tf.placeholder(
#             dtype=tf.float32, shape=[1, 256, 512, 3], name="input_tensor"
#         )
#         self.net = lanenet.LaneNet(phase="test", cfg=self.CFG)
#         self.binary_seg_ret, self.instance_seg_ret = self.net.inference(
#             input_tensor=self.input_tensor, name="LaneNet"
#         )
#         self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=self.CFG)

#         # sess config
#         sess_config = tf.ConfigProto()
#         sess_config.gpu_options.per_process_gpu_memory_fraction = (
#             self.CFG.GPU.GPU_MEMORY_FRACTION
#         )
#         sess_config.gpu_options.allow_growth = self.CFG.GPU.TF_ALLOW_GROWTH
#         sess_config.gpu_options.allocator_type = "BFC"

#         self.sess = tf.Session(config=sess_config)

#         # define moving average version of the learned variables for eval
#         with tf.variable_scope(name_or_scope="moving_avg"):
#             variable_averages = tf.train.ExponentialMovingAverage(
#                 self.CFG.SOLVER.MOVING_AVE_DECAY
#             )
#             variables_to_restore = variable_averages.variables_to_restore()

#         # saver
#         self.saver = tf.train.Saver(variables_to_restore)

#     def _minmax_scale(self, input_arr):
#         """

#         :param input_arr:
#         :return:
#         """
#         min_val = np.min(input_arr)
#         max_val = np.max(input_arr)

#         output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

#         return output_arr

#     def detect(self, image):

#         with self.sess.as_default():
#             self.saver.restore(sess=self.sess, save_path=self.weights_path)
#             start_time = time.time()

#             image_vis = image
#             image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
#             image = image / 127.5 - 1

#             first_step = time.time()
#             # loop_times = 500
#             # for i in range(loop_times):
#             binary_seg_image, instance_seg_image = self.sess.run(
#                 [self.binary_seg_ret, self.instance_seg_ret],
#                 feed_dict={self.input_tensor: [image]},
#             )
#             # t_cost = time.time() - first_step
#             # t_cost /= loop_times

#             # np.savetxt('binary_seg_image.out', binary_seg_image, delimiter = ',')
#             # np.savetxt('instance_seg_image', instace_seg_image, delimiter=',')

#             a = instance_seg_image
#             name = "Instance segmentation"
#             print((name + " dim: "), a.ndim)
#             print((name + " datatype: "), a.dtype)
#             print((name + " shape: "), a.shape)
#             second_step = time.time()

#             postprocess_result = self.postprocessor.postprocess(
#                 binary_seg_result=binary_seg_image[0],
#                 instance_seg_result=instance_seg_image[0],
#                 source_image=image_vis,
#             )

#             third_step = time.time()

#             mask_image = postprocess_result["mask_image"]

#             for i in range(self.CFG.MODEL.EMBEDDING_FEATS_DIMS):
#                 instance_seg_image[0][:, :, i] = self._minmax_scale(
#                     instance_seg_image[0][:, :, i]
#                 )
#             embedding_image = np.array(instance_seg_image[0], np.uint8)

#             end_time = time.time()

#             # print("Image processing time: " + str(first_step - start_time))
#             # print("First run: " + str(second_step - first_step))
#             # print("Average cost: ", t_cost)
#             # print("Post process: " + str(third_step - second_step))
#             # print("Total time: " + str(end_time-start_time))
#             mask_img = mask_image[:, :, (2, 1, 0)]

#             # print("Mask dimensions: ", mask_img.ndim)
#             # print("Mask datatype: ", mask_img.dtype)
#             # print("Mask shape: ", mask_img.shape)

#             cv2.imwrite("mask.png", mask_image)

#             # print("Mask type: ", type(mask_image))

#             binary_segmentation_image = binary_seg_image[0] * 255
#             # print("Binary dimensions: ", binary_segmentation_image.ndim)
#             # print("BInary datatype: ", binary_segmentation_image.dtype)
#             # print("BInary shape: ", binary_segmentation_image.shape)
#             cv2.imwrite("binary.png", binary_segmentation_image)

#             embedded_image = embedding_image[:, :, (2, 1, 0)]
#             plt.figure("embeded_image")
#             plt.imshow(image_vis[:, :, (2, 1, 0)])
#             plt.savefig("source_image.png")
#             cv2.imwrite("embedded.png", embedded_image)

#             src_image = image_vis[:, :, (2, 1, 0)]
#             cv2.imwrite("src_image.png", src_image)

# def advanced_test():

#     image = cv2.imread("resized_image.png")
#     weights_path = "./BiseNetV2_LaneNet_Tusimple_Model_Weights/tusimple_lanenet.ckpt"
#     load_start = time.time()
#     advanced_lane_det = AdvancedLaneDetector(weights_path)
#     load_end = time.time()

#     print("Total load time: ", load_end - load_start)
#     advanced_lane_det.detect(image)
