from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from isaacgym.torch_utils import *
import numpy as np
import torch

class Go2JumpCfg( LeggedRobotCfg ):
    task_name = 'go2_gap'
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.32] # x,y,z [m]
        rel_foot_pos = [[0.1881,0.1881,-0.1881,-0.1881], # x
                        [0.12675,-0.12675,0.12675,-0.12675], # y
                        [-0.32,-0.32,-0.32,-0.32]] # z
        
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0

            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.0,   # [rad]

            'FL_thigh_joint': 0.7220,     # [rad]
            'RL_thigh_joint': 0.7220,   # [rad]
            'FR_thigh_joint': 0.7220,     # [rad]
            'RR_thigh_joint': 0.7220,   # [rad]

            'FL_calf_joint': -1.4441,   # [rad]
            'RL_calf_joint': -1.4441,    # [rad]
            'FR_calf_joint': -1.4441,  # [rad]
            'RR_calf_joint': -1.4441,    # [rad]
        }

        K_HIP = 10.0
        K_THIGH = 16.0
        K_CALF = 20.0
        # damping
        D_HIP = 0.1
        D_THIGH = 0.1
        D_CALF = 0.1
       
        DEFAULT_HIP_ANGLE = 0
        DEFAULT_THIGH_ANGLE = 0.7220
        DEFAULT_CALF_ANGLE = -1.4441

        spring_stiffness = torch.tensor([K_HIP,K_THIGH,K_CALF]).repeat(1,4)
        spring_damping = torch.tensor([D_HIP,D_THIGH,D_CALF]).repeat(1,4)
        spring_rest_pos = torch.tensor([DEFAULT_HIP_ANGLE,DEFAULT_THIGH_ANGLE,DEFAULT_CALF_ANGLE]).repeat(1,4)

    class viewer:
        camera_track_robot = False
        ref_env = 0
        pos = [5.0, -4.0, 1.5]  # [m]
        lookat = [5.0, 0.0, 0.4]  # [m]
        simulate_camera = False

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0 # scales other values
        distribution = "uniform" # "uniform" or "gaussian"
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_vel = 1.0
            contacts_noise_prob =  0.05

    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        episode_length_s = 3 # episode length in seconds
        use_state_history = True
        state_history_length = 20
        state_stored_length = 5

        jump_type = "forward_with_obstacles" # "upward" or "forward" or "forward_with_obstacles"

        continuous_jumping = False # if true: the robot states are not reset after each jump.
        continuous_jumping_reset_probability = 0.9
        use_springs = False
        gap_start_margin = 0.4 # [m] base distance before the near edge of the gap
        gap_landing_margin = 0.4 # [m] desired landing distance after the far edge of the gap
        reset_height = 0.10 # [m]
        reset_landing_error = 0.2 # [in m]

        debug_draw = False
        debug_landing_direction = True
        debug_gap_spawn = True
        throttle_to_real_time = True

        reset_orientation_error = 0.8 # [rad]

        known_contact_feet = True
        known_height = False
        jumping_target = True
        pass_remaining_time = False
        pass_has_jumped = True
        known_quaternion = True
        known_ori_error = False   
        known_error_quaternion = False     
        object_information = True

        num_observations = 0
        if not use_state_history:
            state_history_length = 1

        # Num observations is lin and ang vel (each 3*state_history_length), joint angles (12*state_history_length), 
        # and joint velocities (12*state_history_length) 
        # and action (12*state_history_length) 
        num_observations += 2*3*state_history_length  + 3*12*state_history_length
        
        # + orientation (quaternion*state_history_length)
        if known_quaternion:
            num_observations += 4*state_history_length
        if known_ori_error:
            num_observations += 1*state_history_length
        if known_error_quaternion:
            num_observations += 4*state_history_length
        # And and height if it's known (1*state_history_length)
        if known_height:
            num_observations += 1*state_history_length
        # And desired jumping position (relative) and quat if added:
        if jumping_target:
            num_observations += 7 + 6

        if pass_remaining_time: # Pass remaining time in the episode (one scalar value)
            num_observations += 1

        if pass_has_jumped: # Pass has_jumped bool (one scalar value)
            num_observations += 1
        
        if known_contact_feet: # Pass contact state at the feet (4*state_history_length)
            num_observations += 4*state_history_length
        
        # num_observations += 187 # If measuring height of terrain


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'  
        # control_type = 'actuator_net'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5 # Reduce the hip action scaling factor
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        use_action_filter = True
        filter_freq = 5.0 # [Hz]

        # action_filter_alpha = 0.5
        filter_type = "EMA" # or "EMA" or "butterworth"
        butterworth_order = 2

        safety_clip_actions = True # Clip ALL actions whenever ANY joint approaches its limit.

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "Head", "thigh", "calf", "hip"]
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
        fix_base_link = False

        armature = 0.0
        use_physx_armature = False
  
    class domain_rand ( LeggedRobotCfg.domain_rand ):
        push_robots = False
        push_interval_s = 1.
        max_push_vel_xy = 1.
        randomize_friction = True

        # Randomise some of the initial robot states:
        pos_vel_random_prob = 0.0

        push_upwards = False
        push_upwards_prob = 0.5
        
        randomize_robot_pos = False
        randomize_robot_vel = False
        randomize_robot_ori = True
        randomize_dof_pos = True
        dof_pos_foot_xy_range = [-0.01, 0.01]
        dof_pos_foot_z_range = [-0.02, 0.02]
        dof_pos_base_xy_range = [-0.005, 0.005]
        randomize_spring_params = False
        randomize_motor_strength = True
        randomize_PD_gains = True

        randomize_has_jumped = False # Disable RSI-style has_jumped randomization at reset
        has_jumped_random_prob = 0.8
        reset_has_jumped = True # Whether to reset the has_jumped to False at a random point of the episode
        manual_has_jumped_reset_time = 0 # Manual time at which to reset has_jumped (in steps)
        # Curriculum:
        curriculum = False # Enable initial state randomisation curriculum
        rand_vel_interval_ep = 5 # How many episodes until curriculum is changed
        # At the first flight step, set the velocity of the agent to the desired one
        push_towards_goal = False
        sim_latency = False
        base_latency = 0 # in ms
        sim_pd_latency = False

        lag_timesteps = 6
        randomize_lag_timesteps = False

        randomize_motor_offset = False

        randomize_base_mass = True
        randomize_com = True
        randomize_restitution = False
        randomize_link_mass = False

        randomize_joint_friction = True
        randomize_joint_damping = True

        randomize_joint_armature = False

        # Virtual jump assist force with curriculum decay.
        virtual_force_assist = True
        vf_apply_as_external_force = True

        # Jump assist: track a desired torso/base-height trajectory.
        vf_trajectory_type = "bezier"    # "bezier" or "parabola"
        vf_force_mode = "weight_error"   # "weight_error" or "trajectory_pd"
        vf_desired_height = 0.7          # [m] desired apex/base height
        vf_parabola_duration = 0.8       # [s] fallback full up-and-down trajectory duration
        vf_parabola_duration_range = [0.6, 1.0] # [s] randomized duration range for training
        vf_height_kp = 600.0              # [1/s^2] height tracking gain
        vf_height_kd = 60.0               # [1/s] vertical velocity tracking gain
        vf_height_force_kp = 900.0        # [N/m] height tracking force gain
        vf_height_force_kd = 120.0        # [N/(m/s)] vertical velocity tracking force gain
        vf_stance_max = 2500.0            # [N] max upward force applied to the base body
        vf_x_force_kp = 500.0             # [N/m] forward trajectory tracking force gain
        vf_x_force_kd = 30.0              # [N/(m/s)] forward velocity tracking force gain
        vf_x_force_max = 600.0            # [N] max forward/backward force applied to the base body
        vf_trigger_mode = "upward_velocity" # "height" or "upward_velocity"
        vf_jump_trigger_vz = 0.15         # [m/s] start assist when the robot initiates takeoff
        vf_stance_h_threshold = 0.34     # [m] start the parabola from normal stance contact

        # Landing assist: vertical damping.
        vf_landing_kd = 200.0            # [N*s/m] damping coefficient
        vf_landing_max = 500.0           # [N] max landing assist force
        vf_landing_duration = 35         # [steps] assist window after first contact

        vf_probability = 1.0             # fraction of envs receiving virtual-force assist

        # Curriculum: hold force until success reaches zeta, then decay alpha.
        vf_curriculum = True
        vf_curriculum_mode = "success_gated"
        vf_curriculum_steps_per_iter = 24
        vf_curriculum_decay_start = 0
        vf_curriculum_decay_end = 2500
        vf_adaptive_success_threshold = 0.8  # zeta
        vf_adaptive_decay_step = 0.015      
          # epsilon: 50 successful stages -> zero
        vf_adaptive_check_interval = 50
        vf_adaptive_min_valid_envs = 256

        # Optional orientation assist: roll + yaw to zero, pitch stays free.
        vf_torque_assist = True
        vf_torque_kp_roll = 100.0
        vf_torque_kd_roll = 20.0
        vf_torque_max_roll = 5.0
        vf_torque_kp_yaw = 100.0
        vf_torque_kd_yaw = 20.0
        vf_torque_max_yaw = 5.0

        randomize_gravity = False
        gravity_rand_interval_s = 4.0 # in seconds
        gravity_impulse_duration = 0.99 # in seconds

        # Initial state randomisation ranges:
        class ranges():
            min_robot_vel = [-0.0, -0.0,-0.5] # min velocity [m/s] 
            max_robot_vel = [0.0,0.0,3.0] # max velocity [m/s] 

            min_robot_pos = [-0.0, 0.0, 0.0] # min robot pos [m] (relative to base height 0.32)
            max_robot_pos = [-0.0, 0.0, 0.7] # min robot pos [m] (relative to base height 0.32)

            # At each curriculum step the range changes by the increment step:
            pos_variation_increment = [0.0,0.005]
            vel_variation_increment = [0.01,0.01]
            min_ori_euler = [-0.0,-0.0, -0.1]
            max_ori_euler = [ 0.0, 0.0, 0.1]
            ori_variation_increment = [0.001,0.001,0.01]

            spring_stiffness_percentage = 0.3 # in percentages
            spring_damping_percentage = 0.3 # in percentages
            spring_rest_pos_range = [-0.1,0.1] # in rads

            motor_strength_ranges = [0.999, 1.001] # in percentages of nominal values
            p_gains_range = [0.9,1.1] # percentages of nominal values
            d_gains_range = [0.9,1.1] # percentages of nominal values

            added_mass_range = [-1.0,3.0]

            # Latency to simulate (sampled at start of episode):
            latency_range = [0.0,40.0] # Latency to simulate at the observations -> policy level (in ms)
            pd_latency_range = [0.0,0.0] # Latency to simulate at the low-level PD controller (in ms)

            additional_latency_range = [-5.0,5.0] # Additional random latency to add to the latency at each step

            motor_offset_range = [-0.02, 0.02] # Offset to add to the motor angles

            restitution_range = [0.0, 0.4]
            friction_range = [0.01, 3.0]
            
            com_displacement_range = [-0.1, 0.1]

            joint_friction_range = [0.0, 0.02]
            joint_damping_range = [0.0, 0.005]

            added_link_mass_range = [0.7,1.3] # Factor

            joint_armature_range = [0.0,0.0] # Factor

            gravity_range = [-1.,1.]

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False
        only_positive_rewards_ji22_style = True
        class scales():
            #---------- Task rewards (once per episode): ----------- #

            # Rewards for reaching desired pose upon landing:
            task_pos = 500.0 # Final reward is scale * (dt*decimation)
            task_ori = 200.0
            task_max_height = 100.0 # Reward for maximum height (minus minimum height) achieved
            
            termination = -20.
            jumping = 50.
            takeoff = 15.
            upward_velocity = 8.

            #---------- Continuous rewards (at every time step): ----------- #

            # Rewards for maintaining desired pose AFTER landing:
            post_landing_pos = 10. # Reward for maintaining landing position
            post_landing_ori = 10. # Reward for returning to desired orientation after landing
            post_landing_pitch = 10. # Reward for keeping pitch flat after landing
            post_landing_front_clearance = 20. # Reward for keeping the front body/head off the ground after landing
            landing_feet_far = 25.0 # Reward for putting all feet clearly past the far edge of the gap after landing

            base_height_flight = 20. # Reward for being in the air, only active the first jump
            base_height_stance = 20. # Reward fo
            torso_trajectory = 20.0 # Reward for tracking the virtual-force torso/base height trajectory

            forward_velocity = 20.0 # Reward for building velocity toward the landing target before landing
            tracking_lin_vel = 20.0 # Reward for tracking desired linear velocity
            tracking_ang_vel = 2.0 # Reward for tracking desired angular velocity
            symmetric_joints = -3. # Reward for symmetric joint angles b/w left and right legs
            default_pose = 6. # Reward for staying close to default pose post landing
            feet_distance = -20.0 # Reward for keeping feet close to the body in flight

            #---------- Regularisation rewards: ----------- #

            energy_usage_actuators = -1e-2 # Additional energy usage penalty for the actuators.
            torques_actuators = -0.0 # Penalty for large torques

            base_acc = -1e-6 # Penalty for large base acceleration
            change_of_contact = 0.0 # Reward for maintaining contact state
            early_contact = 5.0 # Reward for maintaining contact with ground early in the episode
            feet_contact_forces = -5.0# Penalty for large contact forces at the feet
            action_rate = -0.2 # Penalty for large change of actions
            action_rate_second_order = -0.0 # Penalty for large change of action rate

            dof_vel = -0.0
            dof_acc = -1e-6
            dof_jerk = -0.0

            dof_pos_limits = -10.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            
            
            
        # Exponential kernel sigma coefficients
        command_pos_tracking_sigma = 0.05
        post_landing_pos_tracking_sigma = 0.001
        command_ori_tracking_sigma = 0.05
        post_landing_pitch_sigma = 0.03
        post_landing_front_offset_x = 0.28
        post_landing_front_clearance_min = 0.16
        post_landing_front_clearance_tolerance = 0.10

        flight_reward_sigma = 0.1
        takeoff_height_target = 0.45
        upward_velocity_clip = 2.0
        torso_trajectory_sigma = 0.01
        torso_trajectory_x_weight = 1.0
        torso_trajectory_z_weight = 1.0
        forward_velocity_clip = 2.0
        landing_feet_far_margin = 0.05
        landing_feet_far_tolerance = 0.25
        base_height_stance_target = 0.36
        max_height_reward_sigma = 0.05
        squat_reward_sigma = 0.001
        stance_reward_sigma = 0.005
        dof_pos_sigma = 0.1

        vel_tracking_sigma = 0.1

        # penalty_sigma = 0.5
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 1.0

        sigma_rew_neg = 0.2
        sigma_neg_rew_curriculum = False
        sigma_neg_rew_curriculum_duration = 1000 # in update steps (= 5*episodes)
        sigma_neg_rew_initial_duration = 1000 # Keep at initial value for this many update steps

        max_contact_force = 150.0

    class normalization( LeggedRobotCfg.normalization ):
        clip_actions = 100.

    class commands():
        jump_over_box = False
        num_commands = 13 # default: relative x,y,z for jump and desired quaternion (euler angles use xyz notation)
        # and 6 for centre of object and its dimensions.
        upward_jump_probability = 0.1
        curriculum = False
        curriculum_type = "time-based"
        randomize_commands = False
        curriculum_interval = 5
        max_curriculum = 1.

        num_levels = 11
        randomize_yaw = False
        
        class ranges(): 
            # The command distances are relative to the initial agent position and are sampled from
            # the ranges below:

            # This is the min/maximum ranges in the jump's distance curriculum (x_des = dx~pos_dx + x)
            pos_dx_lim = [-0.0,0.0]
            pos_dy_lim = [-0.0,0.0]
            pos_dz_lim = [-0.0,0.0]
            # These are the starting ranges for the jump's distances (i.e. if curriculum 
            # if off, these stay the same for the whole training.)
            pos_dx_ini = [0.0,0.0]
            pos_dy_ini = [-0.0,0.0]
            pos_dz_ini = [0.0,0.0]
            # These are the steps for the jump distance changes every curriculum update.
            pos_variation_increment = [0.01,0.01,0.01]

        class distances(): # If you want to pass a fixed command distance, use these values:
            x = 0.0
            y = 0.0
            z = 0.0
            # Specify in Euler 'xyz' notation (order is important as it gets converted to quat later):
            # des_angles_euler = [0.0,0.0,0.0]
            des_yaw = 0.0

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        selected = True
        curriculum = False
        make_terrain_uneven = False
        sloped_terrain_number = 5
        slope_range = [0.0,0.2]

        # terrain_kwargs =  { 'type': 'discrete_obstacles_terrain', 'max_height': 0.1, 'min_size': 0.2,\
        #     'max_size':0.5,'num_rects':10, 'platform_size': 3.}
        # terrain_kwargs = { 'type':'stairs_terrain', 'step_width': 1, 'step_height':0.2}
        terrain_kwargs = { 'type':'gap_terrain', 'gap_size':0.6, 'depth':1.0, 'platform_size':3.0 }

        num_zero_height_terrains = 10
        terrain_difficulty_height_range = [0.0, 0.3]
        terrain_difficulty_width_range = [0.6,1.2]
        
        measure_heights = False
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0]
        terrain_length = 4.
        terrain_width = 4.
        num_rows = 32
        num_cols = 64
        horizontal_scale = 0.2 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 2 # [m]
        # max_init_terrain_level = 4

        static_friction = 1.
        dynamic_friction = 1.
        
    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1

        class physx(LeggedRobotCfg.sim.physx):
            solver_type = 1 # 0: pgs

class Go2JumpCfgPPO( LeggedRobotCfgPPO ):

    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid


        # actor_hidden_dims = [24]#[256, 128, 64]
        # critic_hidden_dims = [24]#[256, 128, 64]
        # activation = 'lrelu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # rnn_type = 'lstm'
        # rnn_hidden_size = 128 # 512
        # rnn_num_layers = 2 # 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        max_grad_norm = 1.0
        clip_param = 0.2 # PPO clip parameter
        # Bounds for the adaptive-KL learning-rate controller. Stated explicitly so this
        # task does not depend on PPO's defaults; these are the values that were hard-coded
        # in ppo.py before they became configurable, so gap behaviour is unchanged.
        # The box task deliberately uses a lower ceiling -- see go2_box_config.py.
        max_learning_rate = 1.e-2
        min_learning_rate = 1.e-5
    class runner( LeggedRobotCfgPPO.runner ):
        # policy_class_name = 'ActorCriticRecurrent'
        # policy_class_name = 'ActorCritic'
        run_name = ''
        experiment_name = 'go2_gap'
        num_steps_per_env = 24 # Try 30?
