#!/usr/bin/env python
# import rospy
# from std_msgs.msg import String
# from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

class ResultsGenerator():
    def __init__(self):
        self.joint_positions = []
        self.time_stamp = []

        self.M_matrix = np.array([[1.0, 0.0, 0.0, 0.0    ], 
                                  [0.0, 1.0, 0.0, 0.410  ], 
                                  [0.0, 0.0, 1.0, 0.30391], 
                                  [0.0, 0.0, 0.0, 1.0    ]])

        self.S_list = np.array([[ 0.0, 0.0, 1.0,  0.0,      0.0,      0.0  ], 
                                [-1.0, 0.0, 0.0,  0.0,     -0.10391,  0.0  ], 
                                [ 1.0, 0.0, 0.0,  0.0,      0.30391, -0.050], 
                                [ 1.0, 0.0, 0.0,  0.0,      0.30391, -0.250], 
                                [ 0.0, 1.0, 0.0, -0.30391,  0.0,      0.0  ]]).T

        #POX params
        self.base = 0.10391
        self.shoulder_1 = 0.200
        self.shoulder_2 = 0.050
        self.elbow = 0.200
        self.wrist = 0.165
        self.pox_params = [self.base, self.shoulder_1, self.shoulder_2, self.elbow, self.wrist]

        # better way would be to parse the data text file but the following is done due to time constraints
        self.angle_data = np.array([
                              (-1.6919808387756348, 0.16873788833618164, -0.45099037885665894, -0.9970875382423401, -3.195281982421875), 
                              (-1.6720391511917114, 0.10431069880723953, -0.35895150899887085, -1.182699203491211, -3.195281982421875),
                              (-1.6751071214675903, 0.03374757990241051, -0.24697090685367584, -1.3284274339675903, -3.192214012145996),
                              (-0.777728259563446, 0.8421554565429688, 0.7117670774459839, -1.5569905042648315, -2.270291566848755),
                              (-0.7900001406669617, 0.7179030179977417, 0.653475821018219, -1.5217089653015137, -2.270291566848755),
                              (-0.7685244083404541, 0.7930681109428406, 0.977145791053772, -1.8208352327346802, -2.28409743309021),
                              (0.7086991667747498, 0.7992039918899536, 0.6366020441055298, -1.5370488166809082, -0.8191457390785217),
                              (0.7056311964988708, 0.7900001406669617, 0.7669904232025146, -1.6628352403640747, -0.8222137093544006),
                              (0.7086991667747498, 0.7915341258049011, 0.9679418802261353, -1.826971173286438, -0.9142525792121887),
                              (1.6306216716766357, 0.15800002217292786, -0.4663301706314087, -1.0154953002929688, 0.07516506314277649),
                              (1.6260197162628174, 0.15033012628555298, -0.30066025257110596, -1.260932207107544, 0.07363107800483704),
                              (1.6290876865386963, 0.03528155758976936, -0.24697090685367584, -1.3284274339675903, 0.07363107800483704)
                              ]) 
        
        big_block_size = 0.038
        # z coordinate for ground truth is calculated by: 
        # big_block_size * number of stacked blocks - (length from profile bar of the eef to top  - center of eef to tip)
        self.ground_truth = np.array([
                                [0.250, -0.025, 0.0],
                                [0.250, -0.025, big_block_size * 2 - 0.025],
                                [0.250, -0.025, big_block_size * 3 - 0.025],
                                [0.250, 0.275, 0.0],
                                [0.250, 0.275, big_block_size * 2 - 0.025],
                                [0.250, 0.275, big_block_size * 3 - 0.025],
                                [-0.250, 0.275, 0.0],
                                [-0.250, 0.275, big_block_size * 2 - 0.025],
                                [-0.250, 0.275, big_block_size * 3 - 0.025],
                                [-0.250, -0.025, 0.0],
                                [-0.250, -0.025, big_block_size * 2 - 0.025],
                                [-0.250, -0.025, big_block_size * 3 - 0.025]
                                ])
    
    def callback(self, data):
        self.joint_positions.append(data.position)
        self.time_stamp.append(data.header.stamp.secs)
        
    def listener(self):
        rospy.init_node('listener', anonymous=True)

        rospy.Subscriber("/rx200/joint_states", JointState, self.callback)
        self.record_joint_angles()

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def plot_results_2D(self):
        num_of_joints_of_interest = 5
        joint_names = ["base", "shoulder", "elbow", "wrist angle", "wrist rotate"]
        joint_positions = np.array(self.joint_positions).T
        self.time_stamp = np.array(self.time_stamp) - self.time_stamp[0]
        for i in range(num_of_joints_of_interest):
            plt.figure()
            plt.title("{} angle vs Time".format(joint_names[i]))
            plt.xlabel("Time (seconds)")
            plt.ylabel("Joint angle (rad)")

            plt.grid(True)
            plt.plot(self.time_stamp, joint_positions[i])
        
        plt.show(block=False)
        close = input("press 0 to close all plots: ")
        if int(close) == 0:
            plt.close("all")

    def record_joint_angles(self):
        joint_angles = []
        stop_recording = input("Press 0/1 to stop/continue joint angle recording: ")
        while int(stop_recording):
            joint_angles.append(self.joint_positions[-1])

            stop_recording = input("Press 0/1 to stop/continue joint angle recording: ")

        for joint_angle in joint_angles:
            print(joint_angle[:5])
    
    def calculate_fk_error(self):
        fk_positions = np.zeros((len(self.angle_data), 3))
        for i in range(len(self.angle_data)):
            angles = self.angle_data[i]
            fk_positions[i, :] = FK_pox(angles, self.M_matrix, self.S_list)[:3]
        
        # print(fk_positions)

        error = fk_positions - self.ground_truth
        # print(error)

        error_norm = np.linalg.norm(error, axis=1)
        print(error_norm)

        return error_norm

    def calculate_IK_error(self):
        calculated_angles = np.zeros((len(self.angle_data), len(self.angle_data.T)))

        IK_inputs = np.zeros((len(self.ground_truth), 4))
        IK_inputs[:, :3] = self.ground_truth
        IK_inputs[:, 3] = -np.pi/2

        for i in range(len(IK_inputs)):
            calculated_angles[i, :] = IK_geometric(self.pox_params, IK_inputs[i])

        error = self.angle_data - calculated_angles
        # print(error)

        error_norm = np.linalg.norm(error[:, :-1], axis=1)
        print(error_norm)
        
        error_wrist_rotate = error[:, -1] + np.pi/2
        print(error_wrist_rotate)
        
        return error_norm, error_wrist_rotate


# copied the functions from kinematics.py below because there were issues with importing the kinematics file        
def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def get_euler_angles_from_transform(transform):
    """!
    @brief      Gets the Euler angles from a transformation matrix.

    @param      transform     transformation matrix

    @return     The Euler angles from transform.
    """

    phi, psi = 0, 0
    
    sine_theta = np.sqrt(1 - transform[2, 2]**2)
    theta = np.arctan2(sine_theta, transform[2, 2])

    if sine_theta > 1e-6:
        phi = np.arctan2(transform[1, 2], transform[0, 2])
        psi = np.arctan2(transform[2, 1], -transform[2, 0])
    elif sine_theta < 1e-6:
        phi = np.arctan2(-transform[1, 2], -transform[0, 2])
        psi = np.arctan2(-transform[2, 1], transform[2, 0])

    return [phi, theta, psi]


def get_pose_from_transform(transform):
    """!
    @brief      Gets the pose from transform.

    @param      transform     transformation matrix

    @return     The pose from transform.
    """

    x = transform[0, 3]
    y = transform[1, 3]
    z = transform[2, 3]

    euler_angles = get_euler_angles_from_transform(transform)
    pose = [x, y, z]
    pose.extend(euler_angles)

    return pose


def FK_pox(joint_angles, M_matrix, S_list):
    """!
    @brief      Get a 6-tuple (x, y, z, phi, theta, psi) representing the pose of the desired link

    @param      joint_angles  The joint angles
                M_matrix      The M matrix
                S_list        List of screw vectors

    @return     A 6-tuple (x, y, z, phi, theta, psi) representing the pose of the desired link
    """

    transform = M_matrix

    for angle_index in range(len(joint_angles)-1, -1, -1):
        S_matrix = to_S_matrix(S_list[:, angle_index])
        S_exp_matrix = expm(S_matrix * joint_angles[angle_index])
        transform = np.matmul(S_exp_matrix, transform)

    x = transform[0, 3]
    y = transform[1, 3]
    z = transform[2, 3]

    return get_pose_from_transform(transform)

def to_S_matrix(S):
    """!
    @brief      Convert to S matrix.

    @param      S   A screw vector of form [w, v]

    @return     A numpy array 4x4 representing a rigid transformation
    """

    # a 4x4 screw matrix
    S_matrix = np.zeros((4, 4))

    # first row of the matrix
    S_matrix[0, 0] = 0
    S_matrix[0, 1] = -S[2]
    S_matrix[0, 2] = S[1]
    S_matrix[0, 3] = S[3]

    # second row of the matrix
    S_matrix[1, 0] = S[2]
    S_matrix[1, 1] = 0
    S_matrix[1, 2] = -S[0]
    S_matrix[1, 3] = S[4]

    # third row of the matrix
    S_matrix[2, 0] = -S[1]
    S_matrix[2, 1] = S[0]
    S_matrix[2, 2] = 0
    S_matrix[2, 3] = S[5]

    # fourth row of the matrix
    S_matrix[3, :] = 0

    return S_matrix


def IK_geometric(pox_params, pose, block_yaw=0):
    """!
    @brief      Get the joint config that produces the pose.

    @param      pox_params  The pox parameters
    @param      pose        The desired pose as np.array x,y,z,phi

    @return     Joint configuration in a list of length 4
    """

    l = pox_params
    l_1_prime = np.sqrt(l[1]**2 + l[2]**2)
    theta_s_prime = np.arctan(l[2]/l[1])

    x = pose[0]
    y = pose[1]
    z = pose[2] - l[0]
    phi = pose[3]
    r = np.sqrt(x**2 + y**2)

    r_prime = r - l[4] * np.cos(phi)
    z_prime = z - l[4] * np.sin(phi)

    P = -2 * l_1_prime * r_prime
    Q = -2 * l_1_prime * z_prime
    R = r_prime**2 + z_prime**2 + l[1]**2 - l[3]**2

    gamma = np.arctan2(Q / np.sqrt(P**2 + Q**2), P / np.sqrt(P**2 + Q**2))
    alpha = gamma - np.arccos(-R / np.sqrt(P**2 + Q**2)) # = np.pi/2 - theta_s_prime - theta_s

    theta_b = np.arctan2(y, x) - np.pi/2
    # print("length theta b ", theta_b, y , x)
    theta_s = np.pi/2 - theta_s_prime - alpha
    theta_e = theta_s + np.arctan2((z_prime - l_1_prime * np.sin(alpha)) / l[3], (r_prime - l_1_prime * np.cos(alpha)) / l[3])
    theta_w = phi + theta_s - theta_e
    theta_wa =  -block_yaw + theta_b
    if phi == 0.0:
        theta_wa = 0.0

    # print("block yaw: ", np.rad2deg(block_yaw))
    # print('theta wa:', np.rad2deg(theta_wa))
    # print("theta_b: ", np.rad2deg(theta_b))
    # print('all: ', [theta_b, theta_s, theta_e, theta_w, theta_wa])
    IK_solutions = list(map(lambda theta: clamp(theta), [theta_b, theta_s, theta_e, theta_w, theta_wa]))
    # print("all: ", IK_solutions)
    if np.any(np.isnan(IK_solutions[:-1])):
        print("No Solution")
        return []

    return IK_solutions



if __name__ == '__main__':
    rg = ResultsGenerator()
    # rg.calculate_fk_error()
    rg.calculate_IK_error()

    # rg.listener()
    # rg.record_joint_angles()
    # plot_input = input("Press 1 to plot 2D results: ")
    # if int(plot_input) == 1:
    #     rg.plot_results_2D()

        
    