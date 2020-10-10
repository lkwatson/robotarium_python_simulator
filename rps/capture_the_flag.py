import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import matplotlib.animation as animation
import matplotlib.patches as patches

import numpy as np
import time
import copy
import math


class RobotState(object):
    def __init__(self, initial_pose, vel_cmd, is_team_purple, robot_number):
        assert len(initial_pose) == 3, "Pose should be 3 elements (x, y, theta)"
        assert len(vel_cmd) == 2, "Vel cmd should be 2 elements (linear, angular)"
        self.initial_pose = initial_pose
        self._pose = initial_pose
        self.last_vel_cmd = vel_cmd

        # If false, is team orange.
        self.is_team_purple = is_team_purple
        self.robot_number = robot_number
        
        # Has this robot been tagged. If true, commands to this robot are ignored.
        self.is_tagged = False

        # Can this robot tag another bot / the flag.
        self.can_tag = True
        self.tag_cooldown = 0.0

        # Has this robot captured a flag
        self.has_flag = False

    def update_pose(self, pose):
        '''
        Updates the robot pose after a simulation step has been taken.
        '''
        assert len(pose) == 3, "Pose should be 3 elements (x, y, theta)"
        self._pose = pose

    def update_last_vel(self, vel_cmd):
        '''
        Updates the last velocity command sent to this robot.
        '''
        assert len(vel_cmd) == 2, "Vel cmd should be 2 elements (linear, angular)"
        self._last_vel_cmd = vel_cmd


class RobotPolicy(object):
    def __init__(self, attempt_tag):
        self.attempt_tag_fn = attempt_tag

    def policy(self, state, observation):
        # Linear and angular velocity
        vel_cmd = [0., 0.]

        return vel_cmd


class RobotObservation(object):
    def __init__(self):
        self.left_encoder = 0  # rads
        self.right_encoder = 0  # rads

        self.flags = list()
        self.robots = list()

    def populate_flags(self, ego_state, flag_loc, active, color):
        flag = dict()
        angle = np.arctan2(ego_state._pose[1] - flag_loc[1], ego_state._pose[0] - flag_loc[0]) - ego_state._pose[2]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        flag['angle'] = angle
        dist = np.linalg.norm((ego_state._pose[:2] - flag_loc))
        flag['dist'] = dist
        flag['color'] = color
        flag['active'] = active
        self.flags.append(flag)
        
    def populate_robots(self, ego_state, bot):
        bot_obs = dict()
        angle = np.arctan2(ego_state._pose[1] - bot._pose[1], ego_state._pose[0] - bot._pose[0]) - ego_state._pose[2]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        bot_obs['angle'] = angle
        dist = np.linalg.norm((ego_state._pose[:2] - bot._pose[:2]))
        bot_obs['dist'] = dist
        bot_obs['color'] = 'purple' if bot.is_team_purple else 'orange'
        self.robots.append(bot_obs)


class CTFGame(object):
    def __init__(self, num_robots=12, store_state_history=True, max_iterations=1000, use_barriers=True):
        # Instantiate Robotarium object
        self.num_robots = num_robots
        assert self.num_robots == 12 or self.num_robots == 4 or self.num_robots == 2
        initial_conditions = np.array([0.])
        if self.num_robots == 12:
            initial_conditions = np.array([[-1.3, -1.3, -1.3, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.3, 1.3, 1.3],
                                           [ 0.5,  0.0, -0.5,  0.5,  0.0, -0.5, 0.5, 0.0,-0.5, 0.5, 0.0,-0.5],
                                           [ 0., 0., 0., 0., 0., 0., np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]])
        elif self.num_robots == 4:
            initial_conditions = np.array([[-1.0, -1.0, 1.0, 1.0],
                                           [0.5, -0.5, 0.5, -0.5],
                                           [0., 0., np.pi, np.pi]])
        elif self.num_robots == 2:
            initial_conditions = np.array([[-1.0, 1.0],
                                           [0.0, 0.0],
                                           [0., np.pi]])

        self.robo_inst = robotarium.Robotarium(number_of_robots=num_robots, 
                                               show_figure=False, 
                                               initial_conditions=initial_conditions, 
                                               sim_in_real_time=False)

        # Create barrier certificates to avoid collision
        self.uni_barrier_cert = create_unicycle_barrier_certificate()

        # Flags, true if present
        self.purple_flags = [True, True]
        self.purple_flag_locs = [(-1.45, -0.85), (-1.45, 0.85)]

        self.orange_flags = [True, True]
        self.orange_flag_locs = [(1.45, -0.85), (1.45, 0.85)]

        self.purple_home = [-1.5, 0.0, 0.0]
        self.orange_home = [1.5, 0.0, np.pi]

        self.purple_points = 0
        self.orange_points = 0

        # Robot tag cooldowns
        self.tag_cooldowns = np.zeros(self.num_robots)

        self.robot_states = [None]*self.num_robots
        self.robot_policies = [None]*self.num_robots

        self.store_state_history = store_state_history
        self.state_history = list()

        # Create barrier certificates to avoid collision
        self.uni_barrier_cert = create_unicycle_barrier_certificate()
        self.use_barriers = use_barriers

        self.max_iterations = max_iterations
        self.iterations = 0

        self.flag_states = list()
        self.flag_states.append(copy.deepcopy(self.purple_flags + self.orange_flags))

        self.score_history = list()
        self.score_history.append([0., 0.])

    def register_purple_team(self, states, policies):
        assert len(states) == self.num_robots/2, "You must provide {} state classes for your team".format(self.num_robots/2)
        assert len(policies) == self.num_robots/2, "You must provide {} policy objects for your team".format(self.num_robots/2)
        self.robot_states[:int(self.num_robots/2)] = states
        self.robot_policies[:int(self.num_robots/2)] = policies

    def register_orange_team(self, states, policies):
        assert len(states) == self.num_robots/2, "You must provide {} state classes for your team".format(self.num_robots/2)
        assert len(policies) == self.num_robots/2, "You must provide {} policy objects for your team".format(self.num_robots/2)
        self.robot_states[int(self.num_robots/2):] = states
        self.robot_policies[int(self.num_robots/2):] = policies
    
    @staticmethod
    def is_offsides(state):
        if state.is_team_purple and state._pose[0] > 0.0:
            return True
        elif not state.is_team_purple and state._pose[0] < 0.0:
            return True
        else:
            return False

    @staticmethod
    def replace_one_flag(flags):
        if flags[0] == False:
            flags[0] = True
            return
        if flags[1] == False:
            flags[1] = True
            return

    def attempt_tag(self, tagger_robot_state):
        '''
        Returns true if the tag attempt was successful, false otherwise.
        '''
        tag_radius = 0.25

        if self.tag_cooldowns[tagger_robot_state.robot_number] > 0.0:
            return False

        self.tag_cooldowns[tagger_robot_state.robot_number] = 3.0
        tagger_robot_state.tag_cooldown = 3.0
        tagger_robot_state.can_tag = False

        if tagger_robot_state.is_team_purple:
            for i, loc in enumerate(self.orange_flag_locs):
                if np.linalg.norm((tagger_robot_state._pose[:2] - np.array(loc))) < tag_radius:
                    tagger_robot_state.has_flag = True
                    self.orange_flags[i] = False
                    return True
        else:
            for i, loc in enumerate(self.purple_flag_locs):
                if np.linalg.norm((tagger_robot_state._pose[:2] - np.array(loc))) < tag_radius:
                    tagger_robot_state.has_flag = True
                    self.purple_flags[i] = False
                    return True

        for state in self.robot_states:
            # Opposing teams
            if state.is_team_purple != tagger_robot_state.is_team_purple:
                if np.linalg.norm((tagger_robot_state._pose[:2] - state._pose[:2])) < tag_radius:
                    if self.is_offsides(state):
                        state.is_tagged = True
                        # If enemy is tagged, they lose and replace flag
                        if state.has_flag:
                            state.has_flag = False
                            if state.is_team_purple:
                                self.replace_one_flag(self.orange_flags)
                            else:
                                self.replace_one_flag(self.purple_flags)

                        return True

        return False

    def setup(self):
        poses = self.robo_inst.get_poses()
        self.robo_inst.step()

        for i, pose in enumerate(poses.copy().T):
            assert self.robot_states[i] != None and self.robot_policies[i] != None
            is_team_purple = i < self.num_robots/2
            # Replace classes with instantiations of those classes.
            self.robot_states[i] = self.robot_states[i](pose, [0., 0.], is_team_purple, i)
            self.robot_policies[i] = self.robot_policies[i](self.attempt_tag)

        if self.store_state_history:
            self.state_history.append(copy.deepcopy(self.robot_states))

    def run(self):
        while self.iterations < self.max_iterations:
            poses = self.robo_inst.get_poses()
            for i, pose in enumerate(poses.copy().T):
                self.robot_states[i].update_pose(pose)

            vel_cmds = np.array([[0., 0.]]*self.num_robots)
            for i in range(self.num_robots):
                # Simulate sensors
                obs = RobotObservation()
                for flag_loc, active in zip(self.purple_flag_locs, self.purple_flags):
                    obs.populate_flags(self.robot_states[i], flag_loc, active, 'purple')
                for flag_loc, active in zip(self.orange_flag_locs, self.orange_flags):
                    obs.populate_flags(self.robot_states[i], flag_loc, active, 'orange')
                for j in range(self.num_robots):
                    if i == j:
                        continue
                    obs.populate_robots(self.robot_states[i], self.robot_states[j])

                # Run policy
                vel_cmds[i] = self.robot_policies[i].policy(self.robot_states[i], obs)

            # Send tagged robots back home
            for i, state in enumerate(self.robot_states):
                if state.is_tagged:
                    unicycle_pose_controller = create_hybrid_unicycle_pose_controller()
                    goal = [0., 0., 0.]
                    if state.is_team_purple:
                        goal = self.purple_home
                    else:
                        goal = self.orange_home

                    if np.linalg.norm((state._pose[:2] - goal[:2])) < 0.15:
                        state.is_tagged = False
                    else:
                        vel_cmds[i] = unicycle_pose_controller(np.array([state._pose]).T, np.array([goal]).T).T[0]

            # Cooldown tag times
            for i in range(self.num_robots):
                self.tag_cooldowns[i] = max(0.0, self.tag_cooldowns[i] - self.robo_inst.time_step)
                self.robot_states[i].tag_cooldown = self.tag_cooldowns[i]
                if self.tag_cooldowns[i] == 0.0:
                    self.robot_states[i].can_tag = True

            # Get successful captures
            for i, state in enumerate(self.robot_states):
                if state.has_flag and not self.is_offsides(state):
                    if state.is_team_purple:
                        self.purple_points += 1
                        self.replace_one_flag(self.orange_flags)
                    else:
                        self.orange_points += 1
                        self.replace_one_flag(self.purple_flags)
                    
                    state.has_flag = False

            # Robotarium code uses different matrix form
            vel_cmds = vel_cmds.T
            # Create safe control inputs (i.e., no collisions)
            if self.use_barriers:            
                vel_cmds = self.uni_barrier_cert(vel_cmds, poses)

            # Set the velocities
            self.robo_inst.set_velocities(np.arange(self.num_robots), vel_cmds)

            for i, cmd in enumerate(vel_cmds.copy().T):
                self.robot_states[i].update_last_vel(cmd)

            if self.store_state_history:
                self.flag_states.append(copy.deepcopy(self.purple_flags + self.orange_flags))
                self.score_history.append([self.purple_points, self.orange_points])
                self.state_history.append(copy.deepcopy(self.robot_states))

            self.iterations += 1
            self.robo_inst.step()
        
        self.robo_inst.call_at_scripts_end()


def plot_game(game, fidelity=1):
    state_history = game.state_history
    robo_inst = game.robo_inst

    fig, ax = plt.subplots(figsize=(14, 10), dpi=60)

    fig.tight_layout()
    plt.title("Simulation")
    plt.axis('equal')
    ax.set_xlim(robo_inst.boundaries[0]-0.1, robo_inst.boundaries[0]+robo_inst.boundaries[2]+0.1)
    ax.set_ylim(robo_inst.boundaries[1]-0.1, robo_inst.boundaries[1]+robo_inst.boundaries[3]+0.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_axis_off()

    flag_patches = list()

    for i, loc in enumerate(game.purple_flag_locs):
        flag = patches.Circle(loc, 0.15, facecolor='#886FE3')
        flag_patches.append(flag)
        ax.add_patch(flag)

    for i, loc in enumerate(game.orange_flag_locs):
        flag = patches.Circle(loc, 0.15, facecolor='#E37F41')
        flag_patches.append(flag)
        ax.add_patch(flag)

    score = plt.text(-0.5, 1.05, "Purple: 0                Orange: 0", size=20)

    chassis_patches = list()
    right_wheel_patches = list()
    left_wheel_patches = list()
    robot_labels = list()

    for i, r in enumerate(state_history[0]):
        color = ''
        if r.is_team_purple:
            color = '#886FE3'
        else:
            color = '#E37F41'
        p = patches.RegularPolygon(r._pose[:2], 4, math.sqrt(2)*robo_inst.robot_radius, r._pose[2]+math.pi/4, facecolor=color, edgecolor='k')
        rw = patches.Circle(r._pose[:2]+robo_inst.robot_radius*np.array((np.cos(r._pose[2]+math.pi/2), np.sin(r._pose[2]+math.pi/2)))+\
                            0.04*np.array((-np.sin(r._pose[2]+math.pi/2), np.cos(r._pose[2]+math.pi/2))),\
                            0.02, facecolor='k')
        lw = patches.Circle(r._pose[:2]+robo_inst.robot_radius*np.array((np.cos(r._pose[2]-math.pi/2), np.sin(r._pose[2]-math.pi/2)))+\
                            0.04*np.array((-np.sin(r._pose[2]+math.pi/2))),\
                            0.02, facecolor='k')
        label = plt.text(*r._pose[:2], str(r.robot_number), rotation=r._pose[2])

        chassis_patches.append(p)
        right_wheel_patches.append(rw)
        left_wheel_patches.append(lw)
        robot_labels.append(label)
        ax.add_patch(p)
        ax.add_patch(rw)
        ax.add_patch(lw)

    # Plot arena
    ax.add_patch(patches.Rectangle(robo_inst.boundaries[:2], robo_inst.boundaries[2], robo_inst.boundaries[3], fill=False))
    ax.plot([0., 0.], [robo_inst.boundaries[1], robo_inst.boundaries[1]+robo_inst.boundaries[3]], 'k')

    def frame(step):
        idx = min(int(step * fidelity), len(state_history)-1)  # max index is final pose
        states = state_history[idx]

        for j in range(len(game.flag_states[idx])):
            if j < 2:
                if game.flag_states[idx][j]:
                    flag_patches[j].set_color('#886FE3')
                else:
                    flag_patches[j].set_color('#858585')
            else:
                if game.flag_states[idx][j]:
                    flag_patches[j].set_color('#E37F41')
                else:
                    flag_patches[j].set_color('#858585')
        score.set_text("Purple: {}                Orange: {}".format(int(game.score_history[idx][0]), int(game.score_history[idx][1])))

        for i, r in enumerate(states):
            chassis_patches[i].xy = r._pose[:2]
            chassis_patches[i].orientation = r._pose[2] + math.pi/4

            right_wheel_patches[i].center = r._pose[:2] + robo_inst.robot_radius*np.array( (np.cos(r._pose[2]+math.pi/2), np.sin(r._pose[2]+math.pi/2)) )+\
                                            0.04*np.array( (-np.sin(r._pose[2]+math.pi/2), np.cos(r._pose[2]+math.pi/2)) )
            right_wheel_patches[i].orientation = r._pose[2] + math.pi/4

            left_wheel_patches[i].center = r._pose[:2] + robo_inst.robot_radius*np.array( (np.cos(r._pose[2]-math.pi/2), np.sin(r._pose[2]-math.pi/2)) )+\
                                        0.04*np.array( (-np.sin(r._pose[2]+math.pi/2), np.cos(r._pose[2]+math.pi/2)) )
            left_wheel_patches[i].orientation = r._pose[2] + math.pi/4

            robot_labels[i].set_position((r._pose[0]-0.02, r._pose[1]-0.02))
            robot_labels[i].set_rotation(r._pose[2])

        # Returns a list of all Artists
        return chassis_patches + right_wheel_patches + left_wheel_patches + robot_labels

    anim = animation.FuncAnimation(fig, frame, 
                                frames=math.ceil(len(state_history) / fidelity), 
                                interval=(robo_inst.time_step * 1000 * fidelity), 
                                blit=True, repeat=False)
    plt.close()
    return anim
