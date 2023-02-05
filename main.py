import sys

from QuadrotorSim import Quadrotor3D
from QuadrotorDynamics import QuadrotorDynamics3D
from Controller import Controller3D
from utils import *
from TrajectoryGenerator import *
from TrajectoryPlanner import *
import yaml


def run_sim(args, waypoints):
    """Main function that runs the simulation"""

    # get/set simulation parameters
    sim_params = parse_args(args)
    show_simulation = sim_params.show_animation_flag
    loop_time = sim_params.loop_time
    time_delta = 0.075
    curr_time = 0
    
    # print information on cmd line
    print_info(loop_time)

    # get pid_gains from yaml file
    with open('pid_gains.yml', 'r') as file:
        pid_gains = yaml.safe_load(file)

    # get crazyflie params
    cfparams = CrazyflieParams()

    # set initial state
    state = State()
    state.x_pos = waypoints[0][0]
    state.y_pos = waypoints[0][1]
    state.z_pos = waypoints[0][2]

    # generate trajectory
    trajectory = TrajectoryPlanner(waypoints, loop_time)

    # instantiate dynamics object
    quad_dynamics = QuadrotorDynamics3D(state, cfparams)

    # instantiate controller object
    controller = Controller3D(cfparams, pid_gains, time_delta)

    # instantiate quadrotor simulation object
    quad_sim = Quadrotor3D(state,
                           pid_gains,
                           cfparams,
                           wp=waypoints,
                           size=1.0,
                           time_delta=time_delta,
                           show_animation=show_simulation)

    waypoint_idx = 0    # tracks what waypoint you are on
    num_wp_reached = 0  # tracks how many waypoints reached

    # control loop
    while waypoint_idx < len(waypoints)-1:
        while curr_time < loop_time:
            
            # trajectory planner
            set_point = trajectory.compute_next_point(waypoint_idx, curr_time)


            # ------------------SENSE ----------------------------------------
            # read "fake perfect" state directly from dynamics
            state = quad_dynamics.fake_perfect_sensor_suite()
            # ----------------------------------------------------------------


            # -------------- (THINK &) ACT -----------------------------------
            # compute thrust (i.e. u)
            U = controller.compute_commands(set_point, state)
            # ----------------------------------------------------------------

            # advance state using dynamics
            quad_dynamics.update(U, time_delta)

            # update plot
            quad_sim.update_plot(state, set_point)
            
            curr_time += time_delta

        
        if quad_sim.check_waypoint(waypoint_idx):
            num_wp_reached += 1

        curr_time = 0
        waypoint_idx += 1


    # show final result with pid_gains
    quad_sim.print_result(num_wp_reached)



if __name__ == "__main__":

    # set the [x,y,z] positions to track
    waypoints = [[-3, -3, 0], 
                [-3, -3, 3], 
                [-3, 3, 3], 
                [3, 3, 3], 
                [3, -3, 3], 
                [-3, -3, 0]]
    
    run_sim(sys.argv, waypoints)
