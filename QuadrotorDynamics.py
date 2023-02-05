"""
The quadrotor dynamics class handles the simulated dynamics 
and numerical integration of the system
"""

import random
import numpy as np
from math import sin, cos
from scipy.integrate import odeint


class QuadrotorDynamics3D():
    """
    This class handles the simulation of crazyflie dynamics using an ODE solver

    You do not need to edit this class
    """
    def __init__(self, state, cfparams):
        """
        Inputs:
        - state (State dataclass):              the current state of the system
        - cfparams (CrazyflieParams dataclass): model parameter class for the crazyflie
        """
        self.state = state
        self.params = cfparams
        self.t = 0
        self.t0 = 0
        self.y0 = [
            self.state.x_pos,
            self.state.y_pos,
            self.state.z_pos,
            self.state.x_vel,
            self.state.y_vel,
            self.state.z_vel,
            self.state.phi,
            self.state.theta,
            self.state.psi,
            self.state.p,
            self.state.q,
            self.state.r
        ]


    def dynamic_model(self, y, t, U):
        """
        Function that represents the dynamic model of the 3D crazyflie

        Inputs:
        - y (list):         the current state of the system
        - t (float):        current simulation time
        - U (np.array):     array of control inputs {u1-u4}

        Returns:
        - dydt (list):  the time derivative of the system state
        """

        F = U[0]
        M = U[1:]

        # compute rotation matrix from body frame to inertia frame
        R = self.rot_matrix(self.state.phi,
                            self.state.theta,
                            self.state.psi)

        # translational dynamics (newton's laws of motion)
        linear_acc = (1/self.params.mass) * ( np.matmul(R, np.array([0,0,F])) -  \
                                         np.array([0,0,self.params.mass * self.params.g]))
        
        # rotational dynamics (euler's law)
        omega = np.array([self.state.p, self.state.q, self.state.r])
        angular_acc = np.matmul( self.params.invI, M - np.cross(omega, np.matmul(self.params.I, omega)))


        dydt = [ 
            y[3],
            y[4],
            y[5],
            linear_acc[0],
            linear_acc[1],
            linear_acc[2],
            y[9],
            y[10],
            y[11],
            angular_acc[0],
            angular_acc[1],
            angular_acc[2]
        ]

        return dydt


    def update(self, U, time_delta):
        """
        Function advances the system state using an ODE solver

        Inputs:
        - U (list):             array of control inputs {u1-u4}
        - time_delta (float):   discrete time interval for simulation
        """

        # enforce propeller thrust limits
        props_thrusts = np.matmul(self.params.A, U[0:3])
        props_thrusts_clamped = np.clip(props_thrusts, self.params.minT, self.params.maxT)

        F = np.matmul(self.params.B[0,:], props_thrusts_clamped)        
        M12 = np.matmul(self.params.B[1:,:], props_thrusts_clamped)     
        M = np.append(M12, U[3])

        U_clamped = np.concatenate((np.array([F]),M))


        # advance state using odeint
        ts = [self.t0, self.t0 + time_delta]
        y = odeint(self.dynamic_model, self.y0, ts, args=(U_clamped,))

        # update initial state
        self.y0 = y[1]
        self.t += time_delta

        self.state.x_pos = y[1][0]
        self.state.y_pos = y[1][1]
        self.state.z_pos = y[1][2]
        self.state.x_vel = y[1][3]
        self.state.y_vel = y[1][4]
        self.state.z_vel = y[1][5]
        self.state.phi = y[1][6]
        self.state.theta = y[1][7]
        self.state.psi = y[1][8]
        self.state.p = y[1][9]
        self.state.q = y[1][10]
        self.state.r = y[1][11]


    def rot_matrix(self, roll, pitch, yaw):
        """
        Calculates the ZXY rotation matrix.

        Inputs:
        - Roll: Angular position about the x-axis in radians.
        - Pitch: Angular position about the y-axis in radians.
        - Yaw: Angular position about the z-axis in radians.

        Returns:
        - 3x3 rotation matrix as NumPy array

        Credits: https://github.com/AtsushiSakai/PythonRobotics
        """
        return np.array(
            [[cos(yaw)*cos(pitch) - sin(roll)*sin(yaw)*sin(pitch),  -cos(roll)*sin(yaw),    cos(yaw)*sin(pitch) + cos(pitch)*sin(roll)*sin(yaw)],
            [cos(pitch)*sin(yaw) +  cos(yaw)*sin(roll)*sin(pitch),   cos(roll)*cos(yaw),    sin(yaw)*sin(pitch) - cos(yaw)*cos(pitch)*sin(roll)],
            [-cos(roll)*sin(pitch),                                  sin(roll),             cos(roll)*cos(pitch)]])
        
    
    def fake_perfect_sensor_suite(self):
        """
        Simulates a fake perfect sensor that reads the true full state (pos, vel) of the system

        Returns:
        - (State dataclass):  true state of the system
        """
        return self.state