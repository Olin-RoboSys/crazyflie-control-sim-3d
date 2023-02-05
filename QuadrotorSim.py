"""
Class for plotting a quadrotor

Original Author: Daniel Ingram (daniel-s-ingram)

Heavily modified by Kene Mbanisi
"""


import matplotlib.pyplot as plt
import control as ct
import numpy as np
from utils import SimData
from math import sin, cos


class Quadrotor3D():
    """
    This class handles the plotting and evaluation of the crazyflie simulation

    You do not need to edit this class!
    """
    def __init__(self, init_state, pid_gains, cfparams, wp=None, size=0.25, time_delta=0, show_animation=True):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        self.show_animation = show_animation
        self.pid_gains = pid_gains
        self.cfparams = cfparams

        self.sim_data = SimData()

        self.time_delta = time_delta
        self.itx = 0

        self.waypoints = wp
        self.wp_x = [wp[i][0] for i in range(len(wp))]
        self.wp_y = [wp[i][1] for i in range(len(wp))]
        self.wp_z = [wp[i][2] for i in range(len(wp))]

        if self.show_animation:
            plt.ion()
            self.fig = plt.figure()
            # for stopping simulation with the esc key.
            self.fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            self.sub1 = self.fig.add_subplot(1,2,1, projection='3d')
            self.sub2 = self.fig.add_subplot(6,2,2)
            self.sub3 = self.fig.add_subplot(6,2,4)
            self.sub4 = self.fig.add_subplot(6,2,6)
            self.sub5 = self.fig.add_subplot(6,2,8)
            self.sub6 = self.fig.add_subplot(6,2,10)
            self.sub7 = self.fig.add_subplot(6,2,12)

            self.fig.set_size_inches(12, 10)    

            self.fig.suptitle("3D Crazyflie Quadrotor Control", fontsize=16)

        self.update_plot(init_state, init_state)


    def update_plot(self, state, set_point):
        """
        Updates the system state and calls plot function
        """
        self.x = state.x_pos
        self.y = state.y_pos
        self.z = state.z_pos
        self.phi = state.phi
        self.theta = state.theta
        self.psi = state.psi
        self.set_point_z = set_point.z_pos
        self.sim_data.x_pos.append(state.x_pos)
        self.sim_data.y_pos.append(state.y_pos)
        self.sim_data.z_pos.append(state.z_pos)
        self.sim_data.phi.append(state.phi)
        self.sim_data.theta.append(state.theta)
        self.sim_data.psi.append(state.psi)

        self.itx += 1
        
        if self.show_animation:
            self.plot()


    def transformation_matrix(self):
        """
        Calculates the ZXY rotation matrix

        Credits: https://github.com/AtsushiSakai/PythonRobotics
        """
        x = self.x
        y = self.y
        z = self.z
        roll = self.phi
        pitch = self.theta
        yaw = self.psi

        return np.array([[cos(yaw)*cos(pitch) - sin(roll)*sin(yaw)*sin(pitch),  -cos(roll)*sin(yaw),    cos(yaw)*sin(pitch) + cos(pitch)*sin(roll)*sin(yaw),    x],
            [cos(pitch)*sin(yaw) +  cos(yaw)*sin(roll)*sin(pitch),   cos(roll)*cos(yaw),    sin(yaw)*sin(pitch) - cos(yaw)*cos(pitch)*sin(roll),    y],
            [-cos(roll)*sin(pitch),                                  sin(roll),             cos(roll)*cos(pitch),                                   z]
            ])


    def plot(self):
        """ main simulation plotter function"""
        num_index = len(self.sim_data.z_pos)
        t = np.linspace(0, num_index*self.time_delta, num_index)

        T = self.transformation_matrix()
        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        self.sub1.cla()
        self.sub2.cla()
        self.sub3.cla()
        self.sub4.cla()
        self.sub5.cla()
        self.sub6.cla()
        self.sub7.cla()

        # plot waypoints        
        self.sub1.plot(self.wp_x, self.wp_y, self.wp_z, 'go--', linewidth=0.25)

        # plot rotors
        self.sub1.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.', markersize=10)

        # plot frame
        self.sub1.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        self.sub1.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')

        # plot track
        self.sub1.plot(self.sim_data.x_pos, self.sim_data.y_pos, self.sim_data.z_pos, 'b:')

        self.sub1.set_xlim(-6, 6)
        self.sub1.set_ylim(-6, 6)
        self.sub1.set_zlim(0, 5)
        self.sub1.set_xlabel('x [m]')
        self.sub1.set_ylabel('y [m]')
        self.sub1.set_title('Iteration: {}, Time: {:0.2f}'.format(self.itx, t[self.itx-1]))


        ### x plot
        self.sub2.plot(t, self.sim_data.x_pos, 'b')
        self.sub2.set_ylabel('x [m]')

        ### y plot
        self.sub3.plot(t, self.sim_data.y_pos, 'b')
        self.sub3.set_ylabel('y [m]')

        ### z plot
        self.sub4.plot(t, self.sim_data.z_pos, 'b')
        self.sub4.set_ylabel('z [m]')

        ### phi plot
        self.sub5.plot(t, self.sim_data.phi, 'b')
        self.sub5.set_ylabel('phi [rad]')

        ### theta plot
        self.sub6.plot(t, self.sim_data.theta, 'b')
        self.sub6.set_ylabel('theta [rad]')

        ### psi plot
        self.sub7.plot(t, self.sim_data.psi, 'b')
        self.sub7.set_ylabel('psi [rad]')
        
        plt.pause(0.075)


    def check_waypoint(self, wp_idx):
        """
        Checks that the drone reaches the assigned waypoint via a distance threshold
        """
        dist_thr = 0.2
        if wp_idx == 0:
            return True
        point1 = np.asarray([self.x, self.y, self.z])
        point2 = np.asarray([self.wp_x[wp_idx+1], self.wp_y[wp_idx+1], self.wp_z[wp_idx+1]])
        return True if np.linalg.norm(point1-point2) < dist_thr else False


    def print_result(self, num_wp_reached):

        textstr = '\n\
        ------------------------------------------------------\n\
        Task Performance and System PID gains \n\
        ------------------------------------------------------\n\
        Number of waypoints reached [ %d / %d ] \n\n\
        X_axis: [kp, ki, kd] =      [ %.1f , %.1f , %.1f ]\n\
        Y_axis: [kp, ki, kd] =      [ %.1f , %.1f , %.1f ]\n\
        Z_axis: [kp, ki, kd] =      [ %.1f , %.1f , %.1f ]\n\n\
        Roll_axis: [kp, ki, kd] =   [ %.1f , %.1f , %.1f ]\n\
        Pitch_axis: [kp, ki, kd] =  [ %.1f , %.1f , %.1f ]\n\
        Yaw_axis: [kp, ki, kd] =    [ %.1f , %.1f , %.1f ]\n' \
        %(num_wp_reached, len(self.waypoints)-1, \
        self.pid_gains['kp_x'], self.pid_gains['ki_x'], self.pid_gains['kd_x'], \
        self.pid_gains['kp_y'], self.pid_gains['ki_y'], self.pid_gains['kd_y'], \
        self.pid_gains['kp_z'], self.pid_gains['ki_z'], self.pid_gains['kd_z'], \
        self.pid_gains['kp_phi'], self.pid_gains['ki_phi'], self.pid_gains['kd_p'], \
        self.pid_gains['kp_theta'], self.pid_gains['ki_theta'], self.pid_gains['kd_q'], \
        self.pid_gains['kp_psi'], self.pid_gains['ki_psi'], self.pid_gains['kd_r'])

        if self.show_animation:
            plt.gcf().text(0.15, 0.01, textstr, fontsize=12)
            plt.subplots_adjust(bottom=0.3)
            plt.show(block=True)

        print(textstr)
