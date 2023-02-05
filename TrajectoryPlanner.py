from TrajectoryGenerator import *
from utils import TrajPoint 

class TrajectoryPlanner():
    """
    This class handles the trajectory planning and generation for the crazyflie

    You do not need to edit this class
    """
    def __init__(self, waypoints, traveral_time):
        
        self.waypoints = waypoints
        self.n_waypoints = len(waypoints)

        # create trajectory generation coefficients
        self.x_coeffs = [[] for i in range(self.n_waypoints-1)]
        self.y_coeffs = [[] for i in range(self.n_waypoints-1)]
        self.z_coeffs = [[] for i in range(self.n_waypoints-1)]

        for i in range(self.n_waypoints-1):
            traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1) % (self.n_waypoints-1)], traveral_time)
            traj.solve()
            self.x_coeffs[i] = traj.x_c
            self.y_coeffs[i] = traj.y_c
            self.z_coeffs[i] = traj.z_c
        
    def compute_next_point(self, wp_idx, t):
        """
        Computes the next trajectory point based on the next waypoint and time
        """        
        traj_point = TrajPoint()
        traj_point.x_pos = calculate_position(self.x_coeffs[wp_idx], t)
        traj_point.y_pos = calculate_position(self.y_coeffs[wp_idx], t)
        traj_point.z_pos = calculate_position(self.z_coeffs[wp_idx], t)
        traj_point.x_vel = calculate_velocity(self.x_coeffs[wp_idx], t)
        traj_point.y_vel = calculate_velocity(self.y_coeffs[wp_idx], t)
        traj_point.z_vel = calculate_velocity(self.z_coeffs[wp_idx], t)
        traj_point.x_acc = calculate_acceleration(self.x_coeffs[wp_idx], t)
        traj_point.y_acc = calculate_acceleration(self.y_coeffs[wp_idx], t)
        traj_point.z_acc = calculate_acceleration(self.z_coeffs[wp_idx], t)

        return traj_point