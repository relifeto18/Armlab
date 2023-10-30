#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import sys
import cv2
import numpy as np
import rospy
import time
from functools import partial

from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
from PyQt4.QtGui import (QPixmap, QImage, QApplication, QWidget, QLabel,
                         QMainWindow, QCursor, QFileDialog)

from ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from test_acc import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None, dh_config_file=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        print("Creating rx arm...")
        if (dh_config_file is not None):
            self.rxarm = RXArm(dh_config_file=dh_config_file)
        else:
            self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.calibrateMousePress

        # Buttons
        # Handy lambda function that can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())

        #User Buttons
        self.ui.btnUser1.setText("Calibrate")
        self.ui.btnUser1.clicked.connect(partial(nxt_if_arm_init, 'calibrate'))
        self.ui.btnUser2.setText('Open Gripper')
        self.ui.btnUser2.clicked.connect(lambda: self.rxarm.open_gripper())
        self.ui.btnUser3.setText('Close Gripper')
        self.ui.btnUser3.clicked.connect(lambda: self.rxarm.close_gripper())
        self.ui.btnUser4.setText('Execute')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'execute'))
        self.ui.btnUser5.setText('Record Waypoints')
        self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'record_waypoints'))
        self.ui.btnUser6.setText('Toggle Gripper State')
        self.ui.btnUser6.clicked.connect(partial(nxt_if_arm_init, 'toggle_gripper'))
        self.ui.btnUser7.setText('Capture Waypoint')
        self.ui.btnUser7.clicked.connect(partial(nxt_if_arm_init, 'capture_waypoint'))
        self.ui.btnUser8.setText('Stop Recording')
        self.ui.btnUser8.clicked.connect(partial(nxt_if_arm_init, 'stop_record_waypoints'))
        self.ui.btnUser9.setText('Detect')
        self.ui.btnUser9.clicked.connect(partial(nxt_if_arm_init, 'detect'))
        self.ui.btnUser10.setText('Grab and Place')
        self.ui.btnUser10.clicked.connect(partial(nxt_if_arm_init, 'grab_and_place'))
        self.ui.btnUser11.setText('Pick and Sort')
        self.ui.btnUser11.clicked.connect(partial(nxt_if_arm_init, 'pick_and_sort'))


        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        self.detection_counter = 0

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()


    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str("%+.2f mm" % (1000 * pos[0])))
        self.ui.rdoutY.setText(str("%+.2f mm" % (1000 * pos[1])))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (1000 * pos[2])))
        self.ui.rdoutPhi.setText(str("%+.2f rad" % pos[3]))
        self.ui.rdoutTheta.setText(str("%+.2f rad" % pos[4]))
        self.ui.rdoutPsi.setText(str("%+.2f rad" % pos[5]))

    @pyqtSlot(QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, block_detection_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        if (self.ui.radioVideo.isChecked()):
            # self.detection_counter += 1
            # if self.detection_counter ==10:
            #     self.camera.detectBlocksInDepthImage()
            #     self.detection_counter = 0
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            # self.detection_counter += 1
            # if self.detection_counter ==10:
            #     self.camera.detectBlocksInDepthImage()
            #     self.detection_counter = 0
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            # self.detection_counter += 1
            # if self.detection_counter ==10:
            #     self.camera.detectBlocksInDepthImage()
            #     self.detection_counter = 0
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.detection_counter += 1
            if self.detection_counter ==20:
                if self.camera.block_detector.current_working_space is None: 
                    self.camera.block_detector.current_working_space = self.camera.block_detector.working_space[0]
                self.camera.detectBlocksInDepthImage()
                self.detection_counter = 0
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(block_detection_image))

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)


    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        mouse_position = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            world_frame, depth_raw = self.camera_to_world_frame(mouse_position)

            self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                             (mouse_position.x(), mouse_position.y(), depth_raw))
            self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                             (world_frame[0], world_frame[1], world_frame[2]))


    def camera_to_world_frame(self, screen_position):
        depth_raw = self.camera.DepthFrameRaw[screen_position.y()][screen_position.x()]
        intrinsic_inverse = np.linalg.inv(self.camera.intrinsic_matrix)
        extrinsic_inverse = np.linalg.inv(self.camera.extrinsic_matrix)

        camera_frame = depth_raw * np.dot(intrinsic_inverse, np.array([[screen_position.x()], [screen_position.y()], [1]]))
        world_frame = np.dot(extrinsic_inverse, np.vstack((camera_frame.reshape(3, 1), np.array(1))))

        return world_frame, depth_raw


    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse position """
        mouse_position = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            world_frame, depth_raw = self.camera_to_world_frame(mouse_position)
            self.camera.last_two_locations.append([np.array([world_frame[0], world_frame[1], world_frame[2]]), mouse_event.pos()])



    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')


### TODO: Add ability to parse POX config file as well
def main(args=None):
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui(dh_config_file=args['dhconfig'])
    app_window.show()
    sys.exit(app.exec_())


# Run main if this file is being run directly
### TODO: Add ability to parse POX config file as well
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c",
                    "--dhconfig",
                    required=False,
                    help="path to DH parameters csv file")
    main(args=vars(ap.parse_args()))
