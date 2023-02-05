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
        - pid_gains (PIDGains dataclass):           pid gain class
        """
        self.params = cfparams

        # set control gains here


    def compute_commands(self, setpoint, state):
        """
        Inputs:
        - setpoint (State dataclass):   the desired control setpoint
        - state (State dataclass):      the current state of the system
        Returns:
        - U (np.array):     array of control inputs {u1-u4}
        """
        U = np.array([0.,0.,0.,0.])

        # your code here

        return U
