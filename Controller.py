import numpy as np
from math import sin, cos

class Controller3D():
    """
    This class computes the commanded thrusts (N) to be applied to the quadrotor plant.

    You are to implement the "compute_commands" method.
    """
    def __init__(self, cfparams, pid_gains, dt):
        """
        Inputs:
        - cfparams (CrazyflieParams dataclass):     model parameter class for the crazyflie
        - pid_gains (dict):                         pid gains

        N.B. pid_gains is a dictionary structure where the keys are 'kp_x', 'kd_z', etc.
        """
        self.params = cfparams

        # set control gains here


    def compute_commands(self, setpoint, state):
        """
        Inputs:
        - setpoint (TrajPoint dataclass):   the desired control setpoint
        - state (State dataclass):          the current state of the system
        Returns:
        - U (np.array):     array of control inputs {u1-u4}

        N.B. TrajPoint is a new dataclass. Please check it out from the utils.py script
        """
        U = np.array([0.,0.,0.,0.])

        # your code here

        return U
