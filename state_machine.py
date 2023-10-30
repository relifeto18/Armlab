"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
import cv2
from kinematics import *


class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,      -0.5, -0.3, 0.0,     0.0],
            [0.75*-np.pi/2,  0.5,  0.3, 0.0,     np.pi/2],
            [0.5*-np.pi/2,  -0.5, -0.3, np.pi/2, 0.0],
            [0.25*-np.pi/2,  0.5,  0.3, 0.0,     np.pi/2],
            [0.0,            0.0,  0.0, 0.0,     0.0],
            [0.25*np.pi/2,  -0.5, -0.3, 0.0,     np.pi/2],
            [0.5*np.pi/2,    0.5,  0.3, np.pi/2, 0.0],
            [0.75*np.pi/2,  -0.5, -0.3, 0.0,     np.pi/2],
            [np.pi/2,        0.5,  0.3, 0.0,     0.0],
            [0.0,            0.0,  0.0, 0.0,     0.0]]
        self.gripper_waypoints = [False] * 10
        
        # define valid negative halves i.e. np.array([[x_min, x_max], [y_min, y_max]])
        self.valid_left_negative_half = np.array([[0.1, 0.35], [0.0, -0.15]])
        self.valid_right_negative_half = np.array([[-0.35, -0.1], [0.0, -0.15]])

        self.small_block_size, self.big_block_size = 0.025, 0.038

        self.calibrated = False

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and functions as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "record_waypoints":
            self.record_waypoints()

        if self.next_state == "capture_waypoint":
            self.capture_waypoint()

        if self.next_state == "toggle_gripper":
            self.toggle_gripper()

        if self.next_state == "stop_record_waypoints":
            self.stop_record_waypoints()

        if self.next_state == "grab_and_place":
            self.grab_and_place()

        if self.next_state == "pick_and_sort":
            self.pick_and_sort()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = 'idle'

        if not self.calibrated:
            rospy.sleep(4)
            self.calibrated = True
            self.next_state = "calibrate"

            
            
    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"
        for waypoint_index, waypoint in enumerate(self.waypoints):
            moving_time = self.calculate_moving_time(waypoint)
            accel_time = moving_time/4

            self.rxarm.set_moving_time(moving_time)
            self.rxarm.set_accel_time(accel_time)
            self.rxarm.set_positions(waypoint)

            # rospy.sleep(moving_time + 0.2)
            rospy.sleep(moving_time)

            if self.gripper_waypoints[waypoint_index]:
                self.rxarm.close_gripper()
            else:
                self.rxarm.open_gripper()

        self.camera.detectBlocksInDepthImage()
        block_detections = self.camera.combined_detection_results
        # if block_detections:

        #     # self.pick_and_sort()
        # #     # print('here?')
        # #     # self.pick_and_sort()
        #     self.next_state = 'pick_and_sort'
        # #     # self.pick_and_sort()
        # else:
        #     self.next_state = "idle"

        self.next_state = "idle"

    def calculate_moving_time(self, waypoint):
        max_angular_displacement = np.absolute(
            np.subtract(waypoint, self.rxarm.get_positions())).max()

        angular_scale = 0.6
        angular_offset = 0.1

        return ((angular_scale * max_angular_displacement + angular_offset) / (self.rxarm.max_angular_speed * self.rxarm.safety_factor))

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"

        tag_detections_array = self.camera.tag_detections_array

        position = np.zeros((1, 3))
        orientation = np.zeros((1, 4))

        for deque_items in range(tag_detections_array.maxlen):
            tag_detections = tag_detections_array.pop()
            for detection in tag_detections.detections:
                if detection.id == (1, 2, 3, 4):
                    detection_position = detection.pose.pose.pose.position
                    detection_orientation = detection.pose.pose.pose.orientation

                    new_position = np.array(
                        [detection_position.x, detection_position.y, detection_position.z])
                    new_orientation = np.array(
                        [detection_orientation.w, detection_orientation.x, detection_orientation.y, detection_orientation.z])

                    position = np.vstack((position, new_position))
                    orientation = np.vstack((orientation, new_orientation))

        position = np.mean(position, axis=0)

        weights = [1] * len(orientation)
        orientation = np.linalg.eigh(np.einsum(
            'ij,ik,i->...jk', orientation[1:, :], orientation[1:, :], weights[1:]))[1][:, -1]

        orientation = np.array([0, 1, 0, 0])

        rotation_matrix = self.quaternion_rotation_matrix(orientation)
        translation_matrix = position * 1000.0
        self.camera.extrinsic_matrix = np.transpose(np.column_stack(
            (np.row_stack((rotation_matrix, translation_matrix)), (0, 0, 0, 1))))

        if self.camera.DepthFrameRaw.any() != 0:
            depth = []

            for world_coordinates in self.camera.tag_locations:
                image_coordinates = self.world_to_image_frame(
                    np.append(world_coordinates, 0))
                depth.append(self.camera.DepthFrameRaw[int(
                    image_coordinates[1] / 1000.0)][int(image_coordinates[0] / 1000.0)])

            self.camera.extrinsic_matrix[2, 3] = np.mean(depth)
            # print(self.camera.extrinsic_matrix)
            self.camera.extrinsic_matrix[2, 3] = self.camera.extrinsic_matrix[2, 3]
            # print(self.camera.extrinsic_matrix)
        # self.camera.depth_error_after_calib = self.camera.img2wolrd([700, 600])

        # print('Initializing Working Space')
        # self.camera.init_working_space()

        self.status_message = "Calibration - Completed Calibration"
        self.next_state = "idle"

    def quaternion_rotation_matrix(self, Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (Q[0],Q[1],Q[2],Q[3]) 

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """

        # 3x3 rotation matrix
        rot_matrix = np.zeros((3, 3))

        # First row of the rotation matrix
        rot_matrix[0, 0] = 2 * (Q[0] * Q[0] + Q[1] * Q[1]) - 1
        rot_matrix[0, 1] = 2 * (Q[1] * Q[2] - Q[0] * Q[3])
        rot_matrix[0, 2] = 2 * (Q[1] * Q[3] + Q[0] * Q[2])

        # Second row of the rotation matrix
        rot_matrix[1, 0] = 2 * (Q[1] * Q[2] + Q[0] * Q[3])
        rot_matrix[1, 1] = 2 * (Q[0] * Q[0] + Q[2] * Q[2]) - 1
        rot_matrix[1, 2] = 2 * (Q[2] * Q[3] - Q[0] * Q[1])

        # Third row of the rotation matrix
        rot_matrix[2, 0] = 2 * (Q[1] * Q[3] - Q[0] * Q[2])
        rot_matrix[2, 1] = 2 * (Q[2] * Q[3] + Q[0] * Q[1])
        rot_matrix[2, 2] = 2 * (Q[0] * Q[0] + Q[3] * Q[3]) - 1

        return rot_matrix

    def world_to_image_frame(self, world_coordinates):
        camera_frame = np.matmul(
            self.camera.extrinsic_matrix, np.append(world_coordinates, 1))
        image_frame = np.matmul(
            self.camera.intrinsic_matrix, camera_frame[:3]) / camera_frame[3]

        return image_frame[:2]

    """ TODO """

    def detect(self):
        """!
        @brief      Detect the blocks
        """
        rospy.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"

        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)

        self.next_state = "idle"

    def record_waypoints(self):
        """!
        @brief      Gets the user input to record a set of waypoints
        """

        self.current_state = "record_waypoints"
        self.status_message = "State: Record Waypoints - Start recording waypoints"

        # Clear the recorded waypoints
        self.waypoints = []
        self.gripper_waypoints = []

        # Disable torque
        self.rxarm.disable_torque()

    def capture_waypoint(self):
        if self.current_state == "record_waypoints":
            current_positions = self.rxarm.get_positions()
            self.waypoints.append(current_positions)
            # True if closed
            self.gripper_waypoints.append(self.rxarm.gripper_state)

            self.status_message = "State: Captured Waypoint"
            self.next_state = "not_capture"

    def toggle_gripper(self):
        if self.current_state == "record_waypoints":
            if self.rxarm.gripper_state:
                self.rxarm.gripper_state = False
                self.status_message = "State: Set Gripper to Open"
            else:
                self.rxarm.gripper_state = True
                self.status_message = "State: Set Gripper to Closed"
            self.next_state = "not_capture"

    def stop_record_waypoints(self):
        if self.current_state == "record_waypoints":
            self.rxarm.enable_torque()
            self.status_message = "State: Record Waypoints - Stopped recording waypoints"
            self.next_state = "idle"

    def grab_and_place(self):
        self.current_state = "grab_and_place"
        self.status_message = "State: Grabbing and Placing"

        # Clear the recorded waypoints
        self.waypoints = []
        self.gripper_waypoints = []

        # print(self.camera.last_two_locations[0])
        # print(second_location)

        # get world coordinates from camera and convert to meters
        # upgrade to nearest block coordinates later
        self.camera.detectBlocksInDepthImage()
        location, mouse_pos = self.camera.last_two_locations.popleft()
        block_present, world_coord, first_yaw = self.camera.is_pixel_in_a_contour(mouse_pos)
        print('detection results:', block_present, world_coord, first_yaw)
        first_location = location / 1000
        first_location = first_location[0], first_location[1], first_location[2]

        # upgrade to expected block coordinates later
        location, mouse_pos = self.camera.last_two_locations.popleft()
        block_present, world_coord, second_yaw = self.camera.is_pixel_in_a_contour(mouse_pos)
        second_location = location / 1000
        second_location = second_location[0], second_location[1], second_location[2]

        self.go_to_safe_pose()
        self.pick_block(first_location, orientation=-np.pi/2, block_yaw=first_yaw)

        # needs to set a coordinate to aim for and then use coordinates relative to where that actually is later
        # this is a set parameter and an addition problem for each task
        self.place_block(second_location, orientation=-np.pi/2, block_yaw=second_yaw)

        # set next state to execute
        self.next_state = "execute"

    def pick_and_sort(self):
        print('pick and sort entered')
        self.current_state = "pick_and_sort"
        self.status_message = "State: Picking and Sorting"

        # Clear the recorded waypoints
        self.waypoints = []
        self.gripper_waypoints = []

        #TODO For level 3: Check if there is a stack. If there is, unstack them first 
        # and do pick and sort (Unless detection is reliable enough to detect blocks that are stacked)
        
        # The spots in the negative half of the plane to place the blocks i.e [x, y, z]
        # TODO: experiment if this is a good spot
        # right_neg_half_loc = np.array([0.275, -0.050, self.big_block_size/2], [0.350, -0.050, self.big_block_size/2], [0.425, -0.050, self.big_block_size/2])
        # left_neg_half_loc = np.array([-0.175, -0.050, self.small_block_size/2], [-0.250, -0.050, self.small_block_size/2], [-0.275, -0.050, self.small_block_size/2])
        
        # Here I am attempting stacking, even if it is not necessary for event 1. If successful, we can tackle event 
        # 1 and 2 at the same time 

        # Assume that I have detection results (block_detections) of the blocks in the positive half of the plane
        # block_detection
        self.camera.block_detector.current_working_space = self.camera.block_detector.working_space[0]
        self.camera.detectBlocksInDepthImage()
        block_detections = tuple(self.camera.combined_detection_results)
        # print('block_detections:', block_detections)
        
        self.go_to_safe_pose()
        
        for block_detection in block_detections:
            center, size, color, yaw = block_detection
            print("input yaw: ", np.rad2deg(yaw))
            self.pick_block(center, -np.pi/2, yaw)
            # # Assume that block_detection.size == 1 denotes large block

            location = self.get_placing_location(size)
            print("get placing location: ", location)

            self.place_block(location, -np.pi/2, yaw)
            
        # set next state to execute
        self.next_state = "execute"
        
    def get_placing_location(self, size):
        right_neg_half_loc = np.array([[0.275, -0.100, self.big_block_size/2], 
                                      [0.350, -0.100, self.big_block_size/2], 
                                      [0.425, -0.100, self.big_block_size/2]])
        left_neg_half_loc = np.array([[-0.175, -0.100, self.small_block_size/2], 
                                     [-0.250, -0.100, self.small_block_size/2], 
                                     [-0.275, -0.100, self.small_block_size/2]])
        
        counter = 0
        if size == "Large":
            neg_half_loc = right_neg_half_loc
            current_working_space = 2
            block_size = self.big_block_size

        else:
            neg_half_loc = left_neg_half_loc
            current_working_space = 1
            block_size = self.small_block_size

        # calculate height to place the big block 
        self.camera.block_detector.current_working_space = self.camera.block_detector.working_space[current_working_space]
        # print(self.camera.block_detector.current_working_space == self.camera.block_detector.working_space[2][1])
        self.camera.detectBlocksInDepthImage()
        block_detections_neg = self.camera.combined_detection_results
        print(block_detections_neg)

        if not block_detections_neg:
            location = neg_half_loc[2]
            print("Placing in furthest neg_half_loc")
        else:
            # center, size, color, yaw = [map(lambda x: x[0] block_detections_neg]
            center = [ele[0] for ele in block_detections_neg]
            center_candidates = []
            location = [0.425 - counter * 0.050, -0.100, self.big_block_size/2]
            counter += 1
            # for c in center:
            #     if c[2] <= self.big_block_size * 2:
            #         center_candidates.append(c)
            # if center_candidates:
            #     center_candidates = sorted(center_candidates, key=lambda x: x[0], reverse=True)
            #     location = center_candidates[0].reshape(1, -1) + np.array([0, 0, block_size]).reshape(1, -1)
            #     location = location.reshape(3, 1)
            #     print("stacking location: ", center_candidates[0], location)
            #     # location = center_candidates[0] + block_size/2
            #     print("Stacking")
            # else:
            #     tolerance = 0.01
            #     center_diff = np.linalg.norm(np.array(center[:, 2]) - np.array(neg_half_loc[1, :2])) < tolerance
            #     center_in_predefined_loc = np.any(center_diff)
            #     if center_in_predefined_loc:
            #         location = neg_half_loc[0]
            #         print("center in predefined location")
            #     else:
            #         location = neg_half_loc[1]
            #         print("center NOT in predefined location")

        return location

    def pick_block(self, location, orientation, block_yaw):
        # y_offset = -0.01
        use_second_solution = self.append_waypoint(location, orientation, block_yaw, False, offset=(0, 0, 0.12))
        print("Use second solution: ", use_second_solution)
        # self.append_waypoint(location, orientation, False, offset=(0, 0, 0.03))
        # self.waypoints[:-1]
        # self.gripper_waypoints[:-1]
        final_orientation = orientation

        if use_second_solution:
            self.append_waypoint(location, 0, block_yaw, True, offset=(0, 0, 0.015))
            final_orientation = 0
        else:
            self.append_waypoint(location, orientation, block_yaw, True, offset=(0, 0, 0.015))
            final_orientation = orientation

        # if final_orientation == -np.pi/2:
        #     if self.waypoints[-1][-1] < -np.pi/2:
        #         # self.append_waypoint(location, final_orientation, block_yaw, True, offset=(0, 0, 0.015))
        #         self.append_waypoint(location, final_orientation, block_yaw, False, offset=(0, 0, 0.015))
        #         self.append_waypoint(location, final_orientation, block_yaw + np.pi/2, True, offset=(0, 0, 0.015))
        #         self.append_waypoint(location, final_orientation, block_yaw + np.pi/2, True, offset=(0, 0, 0.015))
        #         self.append_waypoint(location, final_orientation, block_yaw + block_yaw, True, offset=(0, 0, 0.015))
                
        #         # self.append_waypoint(location, final_orientation, block_yaw + block_yaw, False, offset=(0, 0, 0.015))
        #         # self.append_waypoint(location, final_orientation, 0, False, offset=(0, 0, 0.015))
        #         # self.append_waypoint(location, final_orientation, 0, True, offset=(0, 0, 0.015))
        #     else:
        #         self.append_waypoint(location, final_orientation, block_yaw, False, offset=(0, 0, 0.015))
        #         self.append_waypoint(location, final_orientation, block_yaw - np.pi/2, True, offset=(0, 0, 0.015))
        #         self.append_waypoint(location, final_orientation, block_yaw - np.pi/2, True, offset=(0, 0, 0.015))
        #         self.append_waypoint(location, final_orientation, block_yaw + block_yaw, True, offset=(0, 0, 0.015))
                
        #         # self.append_waypoint(location, final_orientation, block_yaw + block_yaw, False, offset=(0, 0, 0.015))
        #         # self.append_waypoint(location, final_orientation, 0, False, offset=(0, 0, 0.015))
        #         # self.append_waypoint(location, final_orientation, 0, True, offset=(0, 0, 0.015))

        self.go_to_safe_pose()

        return final_orientation, self.waypoints[-1][-1]

    def place_block(self, location, orientation, block_yaw):
        self.append_waypoint(location, orientation, block_yaw, True, offset=(-0.015, 0, 0.12))
        # self.append_waypoint(location, orientation, True, offset=(0, 0, 0.06))
        self.append_waypoint(location, orientation, 0, False, offset=(-0.015, 0, 0.03))
        self.append_waypoint(location, orientation, 0, False, offset=(-0.015, 0, 0.12))
        self.go_to_safe_pose()

    def go_to_safe_pose(self):
        joint_positions = self.rxarm.get_positions()

        # # bring shoulder and elbow to safe positions
        # safe_position = -8 * np.pi/18
        # joint_positions[1] = safe_position
        # joint_positions[2] = 2 * np.pi/18
        # self.waypoints.append(joint_positions)
        # self.gripper_waypoints.append(self.rxarm.gripper_state)
        
        # bring shoulder and elbow to safe positions
        safe_position = -np.pi/3
        joint_positions = [0, safe_position, 0 , 0 , 0]
        self.waypoints.append(joint_positions)
        self.gripper_waypoints.append(self.rxarm.gripper_state)

        # turn base to 0 position
        joint_positions[0] = 0
        self.waypoints.append(joint_positions)
        self.gripper_waypoints.append(self.rxarm.gripper_state)

    def create_pose(self, location, orientation, offset=(0, 0, 0)):
        return (location[0] + offset[0], location[1] + offset[1], location[2] + offset[2], orientation)

    def append_waypoint(self, location, orientation, block_yaw, gripper_state, offset=(0, 0, 0)):
        use_second_solution = False
        solution = IK_geometric(self.rxarm.pox_params, self.create_pose(
            location, orientation, offset), block_yaw=block_yaw)

        if not solution:
            # print("location: ", location)
            magnitude = np.linalg.norm([location[0], location[1]])
            normal_vector = np.array([location[0], location[1]]) / magnitude
            new_location = np.append(normal_vector * (magnitude - 0.08), location[2]) 
            # print("new location: ", new_location)

            solution = IK_geometric(self.rxarm.pox_params, self.create_pose(
            new_location, 0, offset), block_yaw=block_yaw)
            if solution:
                use_second_solution = True

        if solution:
            self.waypoints.append(solution)
            self.gripper_waypoints.append(gripper_state)

        return use_second_solution

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)

    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm = state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)
