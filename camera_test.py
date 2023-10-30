"""!
Class to represent the camera.
"""

from collections import deque
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
import time

from block_detector_test import BlockDetector


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
        self.red = np.array([[151, 50, 30], [181, 256, 256]])
        self.orange = np.array([[5, 50, 30], [15, 256, 256]])
        self.yellow = np.array([[20, 50, 30], [27, 256, 256]])
        self.green = np.array([[60, 50, 30], [80, 256, 256]])
        self.blue = np.array([[100, 50, 30], [110, 256, 256]])
        self.purple = np.array([[110, 50, 30], [150, 256, 256]])
        
        self.color_name = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        self.color_value = [self.red, self.orange, self.yellow, self.green, self.blue, self.purple]
        self.color = dict(zip(self.color_name, self.color_value))

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        # self.intrinsic_matrix = np.array([[896.86108, 0, 660.52307], 
        #                                  [0.0, 897.203186, 381.4194], 
        #                                  [0.0, 0.0, 1.0]])
        self.intrinsic_matrix = np.array([])
        self.extrinsic_matrix = np.array([[1, 0, 0, 0],
                                          [0, -1, 0, 175],
                                          [0, 0, -1, 890],
                                          [0, 0, 0, 1]])
        self.last_two_locations = deque(maxlen=2)
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_detections_array = deque(maxlen=100)
        self.tag_locations = np.array([[-250, -25], [250, -25], [250, 275], [-250, 275]])
        """ block info """
        self.block_contours = np.array([])
        self.block_detections_image = np.array([])
        self.block_detections_world = np.array([])
        self.combined_detection_results = []
        self.block_rects = np.array([])

        self.block_detector = BlockDetector(img_size=(720, 1280), mode=('depth', 960)) 
        self.depth_error_after_calib = 0

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

    def blockDetector(self, contours, rects, centers):
    # def blockDetector(self):    
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        img = self.VideoFrame.copy()
        blur = cv2.GaussianBlur(img,(9,9),1)
        hsv_img = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
        # contours, rects, centers = self.detectBlocksInDepthImage()
        # print(len(centers))
        shape = img.shape
        points = []
        hsv_value = np.zeros([len(contours), 3])
        # block_color = np.zeros(len(contours))
        block_color = []
        colors = self.color

        # i = 0
        # for center in centers:
        #     i += 1
        #     print("%d  " % (i)),
        #     print(hsv_img[center[1]][center[0]])
        # print("")    

        for center in range(len(centers)):
            # print(centers[center])
            mask = np.zeros((shape[0], shape[1]), np.uint8)
            cv2.circle(mask, centers[center], 3, (255, 255, 255), -1)
            points.append(np.where(mask != 0))

        for contour in range(len(points)):
            length = len(points[contour][0])
            circle = points[contour]
            x = circle[0]
            y = circle[1]
            for point in range(len(x)):
                hsv = hsv_img[x[point]][y[point]]
                hsv_value[contour] += hsv
                if ((hsv[0] == 0) and (hsv[1] == 0) and (hsv[2] == 0)):
                    length = length - 1
            hsv_value[contour] = hsv_value[contour] / length
            hsv_value = np.array(hsv_value, dtype="int32")
            # print(hsv_value)

        # for contour in range(len(contours)):
        #     for point in range(len(contours[contour])):
        #         # hsv_value[contour] += hsv_img[contours[contour][point]]
        #         hsv_value[contour] += hsv_img[contours[contour][point][1]][contours[contour][point][0]]
        #         # print(hsv_img[contours[contour][point][1]][contours[contour][point][0]])            
        #     hsv_value[contour] = hsv_value[contour] / len(contours[contour])
        #     hsv_value = np.array(hsv_value, dtype="int32")
        #     # print(hsv_value)
        
            for key in colors.keys():
                if hsv_value[contour][0] in range(colors[key][0][0], colors[key][1][0]):
                    if hsv_value[contour][1] in range(colors[key][0][1], colors[key][1][1]):
                        if hsv_value[contour][2] in range(colors[key][0][2], colors[key][1][2]):
                            block_color.append(key)
                            break
        
        # print(len(contours))
        # print(len(block_color))
        # print(block_color)        

        return block_color

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
    # def HSV(values):
    #     h_min = cv2.getTrackbarPos('H low', 'test_hsv')
    #     h_max = cv2.getTrackbarPos('H high', 'test_hsv')
    #     s_min = cv2.getTrackbarPos('S low', 'test_hsv')
    #     s_max = cv2.getTrackbarPos('S high', 'test_hsv')
    #     v_min = cv2.getTrackbarPos('V low', 'test_hsv')
    #     v_max = cv2.getTrackbarPos('V high', 'test_hsv')
        
    #     red_min = np.array([h_min, s_min, v_min])
    #     red_max = np.array([h_max, s_max, v_max])
        
        # read image and convert to RGB colorspace
        # image = cv2.imread("image_all_blocks.png")
        # image = cv2.GaussianBlur(image,(5,5),0)
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(rgb_image,red_min,red_max)
        # img = cv2.imread("hsv_test.png")
        # blur = cv2.GaussianBlur(img,(9,9),1)
        # hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv_img,red_min,red_max)
        # cv2.imshow("red", mask)

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        time_begin = time.time()
        img_rgb = self.VideoFrame
        
        img_depth = self.DepthFrameRaw

        self.block_detector.set_img_rgb(img_rgb.copy())
        # self.block_detector.set_img_rgb(img_rgb)
        # print(img_rgb)

        for i in range(0, 10, 2):
            for j in range(0, 10, 2):
                top_contours, top_rects, top_centers = self.block_detector.find_centers_hierarchy(img_depth, i, j, \
                                                                                                    'final_contours_'+str(i)+'_'+str(j))

        # time_find_center = time.time()
        # print('time_find_center', time_find_center - time_begin)
        # draw rectangles
        top_areas = []
        for rect in top_rects:
            top_areas.append(self.cal_area(rect))

        colors = self.blockDetector(top_contours, top_rects, top_centers)
        # colors = ['red'] * len(top_centers)

        top_sizes = []
        for area in top_areas:
            if area >= 1200:
                top_sizes.append('Large')
            else:
                top_sizes.append('Small')
                
        # i = 0
        for rect, center, color, area, size in zip(top_rects, top_centers, colors, top_areas, top_sizes):
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # i += 1
            
            cv2.drawContours(img_rgb, [box], 0, (255, 0, 0), 1)
            cv2.circle(img_rgb, tuple(list(center)), 2, (255, 0, 0), thickness=-1)
            # cv2.putText(img_rgb, size + ' ' + color + ' ' + str(i) + str(center), (center[0] - 30, center[1] + 40), 1, 1.0, (255, 255, 255), thickness=1)
            cv2.putText(img_rgb, size + ' ' + color + ' ' + str(area), (center[0] - 30, center[1] + 40), 1, 1.0, (255, 255, 255), thickness=1)

            theta = rect[2]
            cv2.putText(img_rgb, str(int(theta)), (center[0], center[1]), 3, 0.5, (255, 255, 255), thickness=2)


        workspace_idx = self.block_detector.current_working_space[0]
        # print(workspace_idx)
        img_rgb = cv2.rectangle(img_rgb, tuple(self.block_detector.boarder_pos[0]), tuple(self.block_detector.boarder_pos[1]), (255, 0, 0))
        img_rgb = cv2.rectangle(img_rgb, tuple(self.block_detector.base_pos[0]), tuple(self.block_detector.base_pos[1]), (0, 255, 0))
        # print(self.block_detector.area_masks[workspace_idx][0].shape)
        img_rgb = cv2.rectangle(img_rgb, tuple(self.block_detector.area_masks[workspace_idx][0]), tuple(self.block_detector.area_masks[workspace_idx][1]), (0, 0, 255))
            

        # cv2.rectangle(working_space, (197, 116), (1083, 720), 255, cv2.2)
        # cv2.rectangle(working_space, (539, 380), (725, 720), 0, cv2.2)
        # cv2.rectangle(working_space, (0, 562), (1280, 720), 0, cv2.FILLED)

        self.BlockDetectionFrame = img_rgb

        self.block_contours = top_contours
        self.block_rects = top_rects
        self.block_detections_image = top_centers

        # print('------------------')
        # print(top_contours)
        # print(top_rects)
        # print(top_centers)
        # print('------------------')

        self.combined_detection_results = []
        for center, size, color, rect in zip(top_centers, top_sizes, colors, top_rects):
            center_world = self.img2wolrd(center)[0][:3] / 1000
            # print('center world:', center_world[:3])
            yaw = np.deg2rad(rect[2])
            self.combined_detection_results.append([center_world, size, color, yaw])

        # center_world = []
        # for pos in top_centers:
        #     pos_world, depth_raw = self.img2wolrd(pos)
        #     center_world.append([pos_world[0], pos_world[1], depth_raw])

        # self.block_detections_world = np.array(center_world)
        # time_draw_contour = time.time()
        # print('time to draw:', time_draw_contour - time_find_center)
        return top_contours, top_rects, top_centers

    def img2wolrd(self, pixel_pos):
        if pixel_pos[1] >= 720:
            pixel_pos[1] = 719
        if pixel_pos[0] >= 1280:
            pixel_pos[0] = 1279
        depth_raw = self.DepthFrameRaw[pixel_pos[1]][pixel_pos[0]]
        intrinsic_inverse = np.linalg.inv(self.intrinsic_matrix)
        extrinsic_inverse = np.linalg.inv(self.extrinsic_matrix)

        camera_frame = depth_raw * np.dot(intrinsic_inverse,
                                          np.array([[pixel_pos[0]], [pixel_pos[1]], [1]]))
        world_frame = np.dot(extrinsic_inverse, np.vstack((camera_frame.reshape(3, 1), np.array(1))))

        return world_frame, depth_raw

    def is_pixel_in_a_contour(self, pixel):
        pixel = [pixel.x(), pixel.y()]
        # pixel = [pixel.y(), pixel.x()]
        # print('contour:', self.block_contours)
        # print('rects:', self.block_rects)
        # print('center:', self.block_detections_image)
        for cnt, rect, center in zip(self.block_contours, self.block_rects, self.block_detections_image):
            # print('cv return value:', cv2.pointPolygonTest(cnt, tuple(pixel), False))
            if cv2.pointPolygonTest(cnt, tuple(pixel), False) == 1:
                return (True, self.img2wolrd(center)[0][:3], np.deg2rad(-rect[2]))

        return (False, self.img2wolrd(pixel)[0][:3], 0)

    def cal_area(self, rect):
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # print(box)

        # world_pos = []
        # for point in box:
            # print(point)
            # world_pos.append(self.img2wolrd(point)[0][:2])
            # print(self.img2wolrd(point)[0].shape)
            
        # print(world_pos[0].shape)
        # print('-'*20)
        # print(world_pos)
        # area = cv2.contourArea(np.int0(np.array(world_pos)))
        area = cv2.contourArea(np.int0(np.array(box)))


        return area

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
        self.camera.tag_detections_array.append(data)
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
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Block Detection Window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            block_detection_frame = self.camera.convertQtBLockDetectionFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, block_detection_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                
                self.camera.detectBlocksInDepthImage()
                # self.camera.blockDetector()
                cv2.imshow(
                    "Block Detection Window",
                    cv2.cvtColor(self.camera.BlockDetectionFrame, cv2.COLOR_RGB2BGR)
                )
            
                cv2.waitKey(3)
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