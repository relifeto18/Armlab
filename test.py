"""!
Class to represent the camera.
"""

import queue
import cv2
import time
from matplotlib.image import imread
from matplotlib.pyplot import hsv
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError
import queue
import sys

from block_detector import BlockDetector

# Just for test


class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Constructs a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.BlockDetectionFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for color mapping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # HSV color range
        self.red = np.array([[135, 10, 70], [180, 255, 255]])
        self.orange = np.array([[4, 130, 70], [18, 255, 255]])
        self.yellow = np.array([[20, 107, 136], [48, 255, 255]])
        self.green = np.array([[35, 70, 35], [90, 150, 255]])
        self.blue = np.array([[50, 175, 70], [100, 255, 255]])
        self.purple = np.array([[95, 100, 55], [130, 160, 120]])
        
        self.color_name = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        self.color_value = [self.red, self.orange, self.yellow, self.green, self.blue, self.purple]
        self.color = dict(zip(self.color_name, self.color_value))

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([])
        self.extrinsic_matrix = np.array([[1, 0, 0, 0],
                                          [0, -1, 0, 175],
                                          [0, 0, -1, 890],
                                          [0, 0, 0, 1]])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_detections_array = queue.Queue(maxsize=100)
        self.tag_locations = np.array([[-250, -25], [250, -25], [250, 275], [-250, 275]])
        """ block info """
        self.block_contours = np.array([])
        self.block_detections_image = np.array([])
        self.block_detections_world = np.array([])
        self.block_rects = np.array([])

        self.block_detector = BlockDetector(img_size=(720, 1280), mode=('depth', 960))

        cv2.namedWindow('Raw Image')
        print('Camera Initialized!')


    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to color mapped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtBLockDetectionFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.BlockDetectionFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        # img = self.VideoFrame.copy()
        # blur = cv2.GaussianBlur(img,(9,9),1)
        # hsv_img = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
        # contours, rects, _ = self.detectBlocksInDepthImage()
        # # print(contours)
        # hsv_value = np.zeros([len(contours), 3])
        # # block_color = np.zeros(len(contours))
        # block_color = []
        # colors = self.color

        # for contour in range(len(contours)):
        #     for point in range(len(contours[contour])):
        #         # hsv_value[contour] += hsv_img[contours[contour][point]]
        #         hsv_value[contour] += hsv_img[contours[contour][point][1]][contours[contour][point][0]]
        #         # print(hsv_img[contours[contour][point][1]][contours[contour][point][0]])            
        #     hsv_value[contour] = hsv_value[contour] / len(contours[contour])
        #     hsv_value = np.array(hsv_value, dtype="int32")
        #     # print(hsv_value)
        
        #     for key in colors.keys():
        #         if hsv_value[contour][0] in range(colors[key][0][0], colors[key][1][0]):
        #             if hsv_value[contour][1] in range(colors[key][0][1], colors[key][1][1]):
        #                 if hsv_value[contour][2] in range(colors[key][0][2], colors[key][1][2]):
        #                     block_color.append(key)
        #                     break
        # print(len(contours))
        # print(block_color)

        # cv2.imwrite('hsv_test.png', cv2.cvtColor(self.VideoFrame, cv2.COLOR_BGR2RGB))
        # sys.exit()

        # cv2.namedWindow('test_hsv', cv2.WINDOW_NORMAL)
        # cv2.createTrackbar('H low', 'test_hsv',135, 180, self.HSV)
        # cv2.createTrackbar('H high', 'test_hsv', 180, 180, self.HSV)
        # cv2.createTrackbar('S low', 'test_hsv', 105, 255, self.HSV)
        # cv2.createTrackbar('S high', 'test_hsv', 255, 255, self.HSV)
        # cv2.createTrackbar('V low', 'test_hsv', 70, 255, self.HSV)
        # cv2.createTrackbar('V high', 'test_hsv', 255, 255, self.HSV)

    # @staticmethod
    def HSV(self, values):
        h_min = cv2.getTrackbarPos('H low', 'test_hsv')
        h_max = cv2.getTrackbarPos('H high', 'test_hsv')
        s_min = cv2.getTrackbarPos('S low', 'test_hsv')
        s_max = cv2.getTrackbarPos('S high', 'test_hsv')
        v_min = cv2.getTrackbarPos('V low', 'test_hsv')
        v_max = cv2.getTrackbarPos('V high', 'test_hsv')
        
        min = np.array([h_min, s_min, v_min])
        max = np.array([h_max, s_max, v_max])
        
        # read image and convert to RGB colorspace
        image = self.VideoFrame.copy()
        # raw_imgae = self.VideoFrame.copy()
        cv2.imshow("Raw Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = cv2.GaussianBlur(image,(5,5),0)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image,min,max)
        cv2.imshow("mask", mask)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()

        

    # def detectBlocksInDepthImage(self):
    #     """!
    #     @brief      Detect blocks from depth

    #                 TODO: Implement a blob detector to find blocks in the depth image
    #     """
    #     img_rgb = self.VideoFrame
    #     img_depth = self.DepthFrameRaw

        
    #     # self.block_detector.set_img_rgb(img_rgb)
    #     # print(img_rgb)

    #     top_contours, top_rects, top_centers = self.block_detector.find_centers_hierarchy(img_depth)
    #     # draw rectangles

    #     # colors = self.block_detector(top_contours, top_rects, top_centers)
    #     colors = ['red'] * len(top_centers)

    #     for rect, center, color in zip(top_rects, top_centers, colors):
    #         box = cv2.boxPoints(rect)
    #         box = np.int0(box)
    #         cv2.drawContours(img_rgb, [box], 0, (255, 0, 0), 1)
    #         cv2.circle(img_rgb, tuple(list(center)), 2, (255, 0, 0), thickness=-1)
    #         cv2.putText(img_rgb, 'Large ' + color, (center[0] - 30, center[1] + 40), 1, 1.0, (0, 0, 0), thickness=2)

    #         theta = rect[2]
    #         cv2.putText(img_rgb, str(int(theta)), (center[0], center[1]), 1, 0.5, (255, 255, 255), thickness=2)

    #     self.BlockDetectionFrame = img_rgb

    #     self.block_contours = top_contours
    #     self.block_rects = top_rects
    #     self.block_detections_image = top_centers

        # center_world = []
        # for pos in top_centers:
        #     pos_world, depth_raw = self.img2wolrd(pos)
        #     center_world.append([pos_world[0], pos_world[1], depth_raw])

        # self.block_detections_world = np.array(center_world)
        # return top_contours, top_rects, top_centers

    # def img2wolrd(self, pixel_pos):
    #     depth_raw = self.camera.DepthFrameRaw[pixel_pos[0]][pixel_pos[1]]
    #     intrinsic_inverse = np.linalg.inv(self.camera.intrinsic_matrix)
    #     extrinsic_inverse = np.linalg.inv(self.camera.extrinsic_matrix)

    #     camera_frame = depth_raw * np.dot(intrinsic_inverse,
    #                                       np.array([[pixel_pos[0]], [pixel_pos[1]], [1]]))
    #     world_frame = np.dot(extrinsic_inverse, np.vstack((camera_frame.reshape(3, 1), np.array(1))))

    #     return world_frame, depth_raw

    # def is_pixel_in_a_contour(self, pixel):
    #     for cnt, rect, center in zip(self.block_contours, self.block_rects, self.block_detections_world):
    #         if cv2.pointPolygonTest(cnt, pixel, False) == 1:
    #             return center, rect[2]

    #     return np.NaN

    # def cal_area(self, rect):
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)

    #     world_pos = []
    #     for point in box:
    #         world_pos.append(self.img2wolrd(point)[0])

    #     return np.array(world_pos)


class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        self.camera.tag_detections_array.put(data)
        #for detection in data.detections:
        #print(detection.id[0])
        #print(detection.pose.pose.pose.position)


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            # cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("Block Detection Window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)


            cv2.namedWindow('test_hsv', cv2.WINDOW_NORMAL)
            cv2.createTrackbar('H low', 'test_hsv',151, 180, self.camera.HSV)
            cv2.createTrackbar('H high', 'test_hsv', 180, 180, self.camera.HSV)
            cv2.createTrackbar('S low', 'test_hsv', 50, 255, self.camera.HSV)
            cv2.createTrackbar('S high', 'test_hsv', 255, 255, self.camera.HSV)
            cv2.createTrackbar('V low', 'test_hsv', 30, 255, self.camera.HSV)
            cv2.createTrackbar('V high', 'test_hsv', 255, 255, self.camera.HSV)
            
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            block_detection_frame = self.camera.convertQtBLockDetectionFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, block_detection_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                # cv2.imshow(
                #     "Image window",
                #     cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                # cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                # cv2.imshow(
                #     "Tag window",
                #     cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                
                # self.camera.detectBlocksInDepthImage()
                # self.camera.blockDetector()

                # cv2.imshow(
                #     "Block Detection Window",
                #     cv2.cvtColor(self.camera.BlockDetectionFrame, cv2.COLOR_RGB2BGR)
                # )
            
                cv2.waitKey(3)
                # if cv2.waitKey(1) == 27:
                #     cv2.destroyAllWindows()
                #     sys.exit()
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()