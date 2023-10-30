"""!
Implements Forward and Inverse kinematics with product of exponentials
"""

import numpy as np
from scipy.linalg import expm


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