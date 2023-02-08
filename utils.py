""" catch-all script for utility data structures (structs) and functions """

from typing import List
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class CrazyflieParams:
    """  
    This dataclass captures the basic parameters of the Crazyflie 2.0

    Credits: 2016 Bernd Pfrommer
    """

    I = np.array([[1.43e-5,   0,          0], 
                  [0,         1.43e-5,    0],
                  [0,         0,          2.89e-5]])
    invI = np.linalg.inv(I)

    mass:   float = 0.030
    I:      np.ndarray = I
    invI:   np.ndarray = invI
    g:      float = 9.81
    L:      float = 0.046
    max_angle: float = 40*math.pi/180
    maxT:   float = 2.5*mass*g
    minT:   float = 0.05*mass*g

    # matrix to compute individual rotor thrusts from U array
    A = np.array([[0.25,    0,     -0.5/L],
                  [0.25,    0.5/L,  0],
                  [0.25,    0,      0.5/L],
                  [0.25,    -0.5/L, 0]])
    
    # matrix to re-arrange the U array
    B = np.array([[1,   1,   1,  1],
                  [0,   L,   0, -L],
                  [-L,  0,   L,  0]])


@dataclass
class State:
    """This dataclass represents the system state (pos and vel) """
    x_pos:  float = 0.0
    y_pos:  float = 0.0
    z_pos:  float = 0.0
    x_vel:  float = 0.0
    y_vel:  float = 0.0
    z_vel:  float = 0.0
    phi:    float = 0.0
    theta:  float = 0.0
    psi:    float = 0.0
    p:      float = 0.0
    q:      float = 0.0
    r:      float = 0.0


@dataclass
class TrajPoint:
    """This dataclass represents the trajectory point (pos, vel and acc) """
    x_pos:  float = 0.0
    y_pos:  float = 0.0
    z_pos:  float = 0.0
    psi:    float = 0.0
    x_vel:  float = 0.0
    y_vel:  float = 0.0
    z_vel:  float = 0.0
    r:      float = 0.0
    x_acc:  float = 0.0
    y_acc:  float = 0.0
    z_acc:  float = 0.0
    r_dot:  float = 0.0


@dataclass
class SimData:
    """This dataclass captures relevant simulation data for ease of storage"""
    x_pos:          List[float] = field(default_factory=list)
    y_pos:          List[float] = field(default_factory=list)
    z_pos:          List[float] = field(default_factory=list)
    x_vel:          List[float] = field(default_factory=list)
    y_vel:          List[float] = field(default_factory=list)
    z_vel:          List[float] = field(default_factory=list)
    phi:            List[float] = field(default_factory=list)
    theta:          List[float] = field(default_factory=list)
    psi:            List[float] = field(default_factory=list)
    p:              List[float] = field(default_factory=list) 
    q:              List[float] = field(default_factory=list)
    r:              List[float] = field(default_factory=list)
    U:              List[float] = field(default_factory=list)
    U_clamped :     List[float] = field(default_factory=list)



@dataclass
class SimulationParameters:
    """This dataclass captures relevant simulation settings"""
    show_animation_flag:    bool = True
    loop_time:              float = 5.0


def parse_args(args):
    """
    This is a NAIVE argument parser to help with passing values from the commandline.
    Please use carefully. The order matters when specifying parameters in terminal/cmd.

    Inputs:
    - args (list):   argument list from sys.argv

    Returns:
    - sim_params (SimulationParameters dataclass)   simulation parameters
    """
    sim_params = SimulationParameters()

    if len(args) < 2:
        return sim_params

    if len(args) == 2:
        sim_params.show_animation_flag = True if args[1][20:] == 'True' else False
    elif len(args) == 3:
        sim_params.show_animation_flag = True if args[1][20:] == 'True' else False
        sim_params.loop_time = float(args[2][10:])
        
    return sim_params



def print_info(loop_time):
    """
    Simple function to print some simulation info to the terminal/cmdline

    Inputs:
    - loop_time (float)   time to traverse two waypoints

    """
    intro_txt = \
    f'\n \
    Crazyflie 3D Control Exercise\n \
    =====================================================\n\n \
    Simulation settings: \n\n \
    Waypoint Loop time = {loop_time} \n '
    print(intro_txt)