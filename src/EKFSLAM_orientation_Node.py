#!/usr/bin/env python

from math import *
import numpy as np
import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import Pose2D
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from std_msgs.msg import Float64
from messages.msg import odom
from pyquaternion import Quaternion


class EKFSLAMNode(object):

    # ---------------------------------------------------------------------
    # ---------------------- initialization -------------------------------
    # ---------------------------------------------------------------------

    def __init__(self):
        # Initialize visualization
        self.visualize = True
        self.init_plot = 1
        self.timer = 0
        self.lm_plot = []
        self.lm_orient_plot = []
        self.lm_plot_cov = []
        self.robot_pos_cov = []
        self.ax = []
        self.pos_robot = []
        self.robot_line = []
        self.robot_direction = []

        self.first_call = True
        self.dist_off_L = 0.0
        self.dist_off_R = 0.0
        self.dist_l_last = 0.0
        self.dist_r_last = 0.0

        # initialize subscribers
        self.tags_detected = []
        self.odometry = []
        self.pitch = []
        self.init_subscribers()

        # initialize publishers
        self.publishers = {}
        self.init_publishers()

        # initialize pitch angle (Kalman Angle)
        self.pitch_angle = 0

        # set number of landmarks
        self.n_landmarks = 6

        # initialize state
        self.x = np.zeros(3 + 3 * self.n_landmarks)

        # mapping matrix from low dimensional space to high dimensional space
        self.F = np.zeros((3, 3 + 3 * self.n_landmarks))
        self.F[0:3, 0:3] = np.eye(3)

        # initialize linearized motion model (low dimensional)
        self.G = np.zeros((3, 3))
        self.dist = 0.0
        self.dist_last = 0.0

        # initialize covariance matrix (robot pose: certain, landmark positions: uncertain)
        diag = [0.0] * 3
        for i in range(3 * self.n_landmarks):
            diag.append(10000)

        self.cov = np.diag(diag)

        # landmark ID's
        self.landmark_ID = []
        self.all_lm_detected = False

        # process noise (motion_model)
        self.R_motion = np.matmul(self.F.T, np.matmul(np.diag([0.00002, 0.00002, 0.00005]), self.F))

        # measurement noise (single observation)
        self.Q_meas = np.diag([0.05, 0.05, 0.08])

    def init_subscribers(self):
        self.tags_detected = rospy.Subscriber("~/tag_detections", AprilTagDetectionArray, self.correct)
        self.odometry = rospy.Subscriber("~/odometry_data", odom, self.predict)
        self.pitch = rospy.Subscriber("~/pitch_angle", Float64, self.update_pitch)

    def init_publishers(self):
        """ initialize ROS publishers and stores them in a dictionary"""
        # position of segway in world frame
        self.publishers["robot_pose"] = rospy.Publisher("~robot_pose", Pose2D, queue_size=1)
        self.publishers["covariance"] = rospy.Publisher("~covariance", numpy_msg(Floats), queue_size=1)
        self.publishers["state"] = rospy.Publisher("~state", numpy_msg(Floats), queue_size=1)

    def update_pitch(self, msg):
        self.pitch_angle = radians(msg.data)

    # ---------------------------------------------------------------------
    # ---------------------- prediction step-------------------------------
    # ---------------------------------------------------------------------

    def predict(self, odom):
        # callback function of the stepper motors --> Use odometry data as input of the motion model
        # read message
        # restrict yaw angle from -2pi to 2 pi

        if self.first_call:
            self.dist_off_L = odom.dist_L
            self.dist_off_R = odom.dist_R
            self.first_call = False
            self.x[0:3] = np.zeros(3)

        dist_l = odom.dist_R - self.dist_off_R
        dist_r = odom.dist_L - self.dist_off_L

        # integrate system dynamics
        self.dist = (dist_l + dist_r)/2
        delta = self.dist - self.dist_last
        yaw_odom = (dist_l - self.dist_l_last - (dist_r - self.dist_r_last)) / 0.184  # Todo: check wheelbase
        x_p = self.x[0:3] + np.array([cos(self.x[2]) * delta, sin(self.x[2]) * delta, yaw_odom])
        x_p[2] = atan2(sin(x_p[2]), cos(x_p[2]))

        # linearize motion model
        self.G[0, 2] = -sin(self.x[2]) * delta
        self.G[1, 2] = cos(self.x[2]) * delta
        # new: linearization correct? guess so

        # map linearized motion model to high dimensional space
        lin_motion = np.eye(3 + 3 * self.n_landmarks) + np.matmul(self.F.T, np.matmul(self.G, self.F))

        self.cov = np.matmul(lin_motion, np.matmul(self.cov, lin_motion.T)) + self.R_motion
        self.x[0:3] = x_p
        self.dist_last = self.dist
        self.dist_l_last = dist_l
        self.dist_r_last = dist_r

        # publish robots pose
        x_pub = Pose2D()
        x_pub.x = self.x[0]
        x_pub.y = self.x[1]
        x_pub.theta = self.x[2]
        self.publishers["robot_pose"].publish(x_pub)
    # ---------------------------------------------------------------------
    # ---------------------- correction step ------------------------------
    # ---------------------------------------------------------------------

    def correct(self, msg):
        # callback function of the April Tag detection algorithm
        # april tag msg contains header and detections (<-- AprilTagDetectionArray)

        # rotation matrix from segway to world frame
        rot_mat = np.array([[cos(self.x[2]), -sin(self.x[2])], [sin(self.x[2]), cos(self.x[2])]])

        # for all detected landmarks do
        for i, tag in enumerate(msg.detections):
            if int(tag.id[0]) not in self.landmark_ID and self.all_lm_detected:
                pass
            else:
                if len(self.landmark_ID) == self.n_landmarks:
                    self.all_lm_detected = True

                # convert landmark to segway frame
                landmark_pos = self.convert_cam_pos_to_segway_pos(tag.pose.pose.pose.position)

                # add yaw information from apriltags
                orient = tag.pose.pose.pose.orientation
                orient_lm = Quaternion(orient.w, orient.x, orient.y, orient.z)

                # pitch in camera frame corresponds to yaw world/segway frame (no transformation needed)
                yaw, pitch, roll = orient_lm.yaw_pitch_roll

                # if landmark never seen before, initialize landmark position with measured position
                if int(tag.id[0]) not in self.landmark_ID:
                    self.landmark_ID.append(int(tag.id[0]))
                    j = self.landmark_ID.index(int(tag.id[0]))
                    self.x[(3 + 3 * j):(3 + 3 * j + 2)] = self.x[0:2] + rot_mat.dot(landmark_pos)
                    self.x[3 + 3 * j + 2] = self.x[2] - pitch


                # get landmark index and landmark pose
                j = self.landmark_ID.index(int(tag.id[0]))

                # restrict landmark orientation

                self.x[3 + 3 * j + 2] = atan2(sin(self.x[3 + 3 * j + 2]),
                                              cos(self.x[3 + 3 * j + 2]))

                # measurements (landmark position + orientation in segway frame)
                z_meas = np.append(landmark_pos, pitch)

                # observation model (maps from state space to observation space)
                lm_hat = rot_mat.T.dot(self.x[(3 + 3 * j):(3 + 3 * j + 2)] - self.x[0:2])
                pitch_hat = atan2(sin(-(self.x[3 + 3 * j + 2] - self.x[2])), cos(-(self.x[3 + 3 * j + 2] - self.x[2])))

                # while abs(pitch_hat) >= np.pi:
                #     pitch_hat = pitch_hat - 2 * np.pi * pitch_hat/abs(pitch_hat)

                z_hat = np.append(lm_hat, pitch_hat)

                # linearized observation model for a single landmark
                H = np.zeros((3, 3 + 3 * self.n_landmarks))
                H[0, 0] = -cos(self.x[2])
                H[0, 1] = -sin(self.x[2])
                H[0, 2] = cos(self.x[2]) * (self.x[3 + 3 * j + 1] - self.x[1]) - sin(self.x[2]) * (self.x[3 + 3 * j] - self.x[0])
                H[1, 2] = -cos(self.x[2]) * (self.x[3 + 3 * j] - self.x[0]) - sin(self.x[2]) * (self.x[3 + 3 * j + 1] - self.x[1])
                H[1, 0] = sin(self.x[2])
                H[1, 1] = -cos(self.x[2])
                H[2, 2] = 1
                H[0, 3 + 3 * j] = cos(self.x[2])
                H[0, 3 + 3 * j + 1] = sin(self.x[2])
                H[1, 3 + 3 * j] = -sin(self.x[2])
                H[1, 3 + 3 * j + 1] = cos(self.x[2])
                H[2, 3 + 3 * j + 2] = -1

                # compute Kalman gain
                kalman_gain = np.matmul(self.cov, np.matmul(H.T, np.linalg.inv(np.matmul(H, np.matmul(self.cov, H.T)) + self.Q_meas)))

                # correct state
                self.x += kalman_gain.dot(z_meas - z_hat)
                self.cov = np.matmul(np.eye(3 + 3 * self.n_landmarks) - np.matmul(kalman_gain, H), self.cov)

                # TODO: check correction of yaw angle --> compass sensor needed ?

        # publish robots pose
        x_pub = Pose2D()
        x_pub.x = self.x[0]
        x_pub.y = self.x[1]
        x_pub.theta = self.x[2]

        if self.visualize and self.timer == 5:
            self.visualize_positions()
            self.timer = 0

        self.timer += 1

        self.publishers["robot_pose"].publish(x_pub)

        # pusblish covariance matrix
        self.publishers["covariance"].publish(self.cov.reshape(np.power(3 + 3*self.n_landmarks, 2)))
        self.publishers["state"].publish(self.x)

    def convert_cam_pos_to_segway_pos(self, position):
        wheelbase = 0.172
        rot_mat = np.array([[0, sin(self.pitch_angle), cos(self.pitch_angle)],
                            [1, 0, 0],
                            [0, cos(self.pitch_angle), -sin(self.pitch_angle)]])
        cam_pos = np.array([sin(self.pitch_angle) * wheelbase, 0, cos(self.pitch_angle) * wheelbase])
        lm_pos_cam = np.array([position.x, position.y, position.z])
        lm_segway = cam_pos + rot_mat.dot(lm_pos_cam)
        print lm_segway
        return lm_segway[0:2]

    def visualize_positions(self):
        d_ = 0.4
        if self.init_plot == 1:
            plt.ion()
            self.ax = plt.figure().add_subplot(111)
            self.pos_robot, = plt.plot(self.x[0], self.x[1], 'ro', markersize=10)
            self.robot_line, = plt.plot([self.x[0] - sin(self.x[2]) * d_, self.x[0] + sin(self.x[2]) * d_],
                                        [self.x[1] + cos(self.x[2]) * d_, self.x[1] - cos(self.x[2]) * d_], 'r--')
            self.robot_direction = plt.arrow(self.x[0], self.x[1], cos(self.x[2])*0.3, sin(self.x[2])*0.3,
                                             head_width=0.08, head_length=0.1)
            self.robot_pos_cov = self.draw_covariance_ellipse(self.x[0:2], self.x[2],
                                                              self.cov[0:2, 0:2], self.ax)

            for j in range(self.n_landmarks):
                pos_lm = self.x[(3 + 3 * j):(3 + 3 * j + 3)]
                # plot position of landmarks
                lm, = plt.plot(pos_lm[0], pos_lm[1], 'bo')
                d__ = 0.2
                orient_plot, = plt.plot([pos_lm[0] - sin(pos_lm[2]) * d__, pos_lm[0] + sin(pos_lm[2]) * d__],
                                        [pos_lm[1] + cos(pos_lm[2]) * d__, pos_lm[1] - cos(pos_lm[2]) * d__], 'b-.')
                self.lm_plot.append(lm)
                self.lm_orient_plot.append(orient_plot)
                self.lm_plot_cov.append(self.draw_covariance_ellipse(pos_lm, 0, np.zeros((2, 2)), self.ax))

            self.init_plot = 0
            plt.xlim((-3, 3))
            plt.grid(color='gray', linestyle='-', linewidth=1)
            plt.ylim((-3, 3))

        # update robot position
        self.pos_robot.set_xdata(self.x[0])
        self.pos_robot.set_ydata(self.x[1])
        self.robot_line.set_data([self.x[0] - sin(self.x[2]) * d_, self.x[0] + sin(self.x[2]) * d_],
                                 [self.x[1] + cos(self.x[2]) * d_, self.x[1] - cos(self.x[2]) * d_])
        try:
            self.robot_pos_cov.remove()
        except ValueError:
            pass
        self.robot_pos_cov = self.draw_covariance_ellipse(self.x[0:2], self.x[2], self.cov[0:2, 0:2], self.ax)

        for jj in range(self.n_landmarks):
            lm_ = self.lm_plot[jj]
            lm_.set_xdata(self.x[3 + 3 * jj])
            lm_.set_ydata(self.x[3 + 3 * jj + 1])

            orient_ = self.lm_orient_plot[jj]
            orient_.set_data([self.x[3+3*jj] - sin(self.x[3+3*jj+2]) * d_, self.x[3+3*jj] + sin(self.x[3+3*jj+2]) * d_],
                             [self.x[3+3*jj+1] + cos(self.x[3+3*jj+2]) * d_, self.x[3+3*jj+1] - cos(self.x[3+3*jj+2]) * d_])
            cov_lm = self.cov[(3 + 3 * jj):(3 + 3 * jj + 2), (3 + 3 * jj):(3 + 3 * jj + 2)]
            try:
                self.lm_plot_cov[jj].remove()
            except ValueError:
                pass
            self.lm_plot_cov[jj] = self.draw_covariance_ellipse(self.x[(3 + 3 * jj):(3 + 3 * jj+2)], 0, cov_lm, self.ax)
        plt.draw()
        plt.pause(0.0000000001)

    def draw_covariance_ellipse(self, pos, yaw, cov, ax):
        lambda1 = (cov[0, 0] + cov[1, 1]) / 2 + sqrt(np.power((cov[0, 0] - cov[1, 1]) / 2, 2) + np.power(cov[0, 1], 2))
        lambda2 = (cov[0, 0] + cov[1, 1]) / 2 - sqrt(np.power((cov[0, 0] - cov[1, 1]) / 2, 2) + np.power(cov[0, 1], 2))
        try:
            elps = Ellipse(pos, 2 * sqrt(lambda1), 2 * sqrt(lambda2), yaw, alpha=0.5, facecolor='pink', edgecolor='black')
            ax.add_patch(elps)
            return elps
        except ValueError:
            pass

def main():
    """Starts the EKF SLAM Node"""
    rospy.init_node("EKFSLAM_orientation_Node")
    EKFSLAMNode()
    rospy.spin()


if __name__ == "__main__":
    main()
