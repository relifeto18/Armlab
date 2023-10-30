# Armlab

Using a 6-DOF manipulator and depth camera to pick different sizes and colors of blocks and place them in the target locations. 

**Acting**: Homogeneous coordinate transforms / Forward kinematics / Inverse kinematics

**Sensing**: 3D image calibration / Object detection with OpenCV

**Reasoning**: Path planning / Path smoothing

![](https://github.com/relifeto18/Armlab/blob/main/grasp%20block.gif)

## Instruction
Launch realsense:

roslaunch realsense2_camera rs_camera.launch align_depth:=true

Launch rx200 arm:

roslaunch interbotix_sdk arm_run.launch robot_name:=rx200 use_time_based_profile:=true gripper_operating_mode:=pwm
