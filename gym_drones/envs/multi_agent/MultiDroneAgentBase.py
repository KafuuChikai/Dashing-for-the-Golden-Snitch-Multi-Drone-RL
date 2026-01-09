import os
import numpy as np
from gymnasium import spaces

from gym_drones.envs.DroneBase import DroneBase
from gym_drones.utils.enums import (
    DroneModel,
    ActionType,
    ObservationType,
    SimulationDim,
)

from typing import Optional, Tuple, Union


class MultiDroneAgentBase(DroneBase):
    """Base multi-agent environment class for reinforcement learning."""

    ################################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.MULTI_TEST_DRONE,
        num_drones: int = 2,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
        sim_freq: int = 100,
        ctrl_freq: int = 100,
        obs: ObservationType = ObservationType.RACE_MULTI,
        act: ActionType = ActionType.RATE,
        dim: SimulationDim = SimulationDim.DIM_3,
        episode_len_sec: int = 15,
        use_mappo: bool = False,
        eval_mode: bool = False,
        sensor_sigma: float = 0.01,
        domain_rand: bool = False,
        drone_model_path: Optional[Union[str, os.PathLike]] = None,
        num_obstacles: int = 0,
        obstacle_size: float = 0.5,
        ray_length: float = 5.0,
        num_rays: int = 12,
        obstacle_safe_dist: float = 0.3,
    ):
        """Initialization of a generic multi-agent RL environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        initial_xyzs : ndarray[float] | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys : ndarray[float] | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        sim_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)
        dim : SimulationDim, optional
            The dimension of the simulation (2D or 3D).
        episode_len_sec : int, optional
            The length of the episode in seconds.
        use_mappo : bool, optional
            Whether to use the MAPPO algorithm. if False, uses PPO(stable-baseline 3), else uses MAPPO(on-policy).
            default is False.
        eval_mode : bool, optional
            Whether to use the evaluation mode. If True, the environment is in evaluation mode.
            default is False.
        sensor_sigma : float, optional
            The standard deviation of the sensor noise. If 0, no noise is added.
            default is 0.01.
        domain_rand : bool, optional
            Whether to use domain randomization. If True, the environment is in domain randomization mode.
            default is False.
        drone_model_path : str | os.PathLike, optional
            The path to the drone model file. If None, uses the default drone model.
            default is None.

        """
        if num_drones < 2:
            print("[ERROR] in MultiDroneAgentBase.__init__(), num_drones should be >= 2")
            exit()
        if obs != ObservationType.RACE and obs != ObservationType.RACE_MULTI:
            print("[ERROR] in MultiDroneAgentBase.__init__(): Wrong self.ObservationType!")
            exit()
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.DIM = dim
        self.use_mappo = use_mappo
        #### Set max sim time (per episode) ########################
        self.EPISODE_LEN_SEC = episode_len_sec
        self.EVAL_MODE = eval_mode
        self.sensor_sigma = sensor_sigma
        #### Initialize DroneBase class ############################
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            domain_rand=domain_rand,
            drone_model_path=drone_model_path,
            num_obstacles=num_obstacles,
            obstacle_size=obstacle_size,
            ray_length=ray_length,
            num_rays=num_rays,
            obstacle_safe_dist=obstacle_safe_dist,
        )
        #### Set observation space size ############################
        self._getObsSize()
        #### Store obstacle parameters for reward computation #######
        self.obstacle_reward_weight = 1.0  # lambda_obs in the reward equation
        #### Create action and observation spaces ##################
        # self.share_observation_space = self._shareObservationSpace()
        self.share_observation_space = self.observation_space
        self.Collisions_alert = np.full(self.NUM_DRONES, False)

    ################################################################################

    def step(self, action: np.ndarray) -> Tuple:
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray[float] : [-1, 1]
            (NUM_DRONES, action_dim)-shaped array for multiple drones.

            The input action for one or more drones, translated into real inputs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        obs : ndarray[float]
            (NUM_DRONES, obs_dim)-shaped array of observations.
        share_obs : ndarray[float]
            (NUM_DRONES, share_obs_dim)-shaped array of shared observations.
        state : ndarray[float]
            (NUM_DRONES, state_dim)-shaped array of states.
        reward : ndarray[float]
            (NUM_DRONES,)-shaped array of rewards.
        terminated : ndarray[bool]
            (NUM_DRONES,)-shaped array of booleans indicating if the episode
            has terminated for each drone.
        truncated : ndarray[bool]
            (NUM_DRONES,)-shaped array of booleans indicating if the episode
            has been truncated for each drone.
        info : dict
            A dictionary containing additional information about the
            environment, such as the current state of each drone.

        """
        # NOTE: This step() function is capable for on-policy (MAPPO), with
        # share obs ouputs.

        #### Save, preprocess, and clip the action to the max value #
        self.clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the physics steps #####
        valid_mask = ~(self.crashed | self.finished)
        valid_indices = np.where(valid_mask)[0]
        for _ in range(self.SIM_STEPS_PER_CTRL):
            self._dynamics_vectorized(self.clipped_action, drone_indices=valid_indices)
        #### Prepare the return values #############################
        obs, share_obs = self._computeObs()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        reward = self._computeReward(terminated, truncated)
        dones = self._computeDones(terminated, truncated)
        info = self._computeInfo()
        #### Save the last step (e.g. to compute drag) #############
        self._saveLastStates()
        self._saveLastAction(action)
        self.last_clipped_action = self.clipped_action
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + self.SIM_STEPS_PER_CTRL
        self._computeExtra()

        if self.use_mappo:
            truncated = np.all(self.crashed)
            # return obs, share_obs, reward, terminated, truncated, info
            return obs, obs, reward, terminated, truncated, info
        else:
            # for SB3, change truncated condition to Bool
            truncated = np.all(self.crashed)
            # truncated = False
            return obs, reward, terminated, truncated, info

    ################################################################################

    def reset(self, seed: int = None, options: dict = None) -> Tuple:
        """Resets the environment to an initial state.

        Parameters
        ----------
        seed : int, optional
            The random seed for the environment. The default is None.
        options : dict, optional
            Additional options for resetting the environment. The default is None.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, dict]
            A tuple containing the initial observation, the shared observation,
            and the initial info dictionary.
        The observation is a (NUM_DRONES, obs_size)-shaped array of floats
        containing the normalized state of each drone, and the shared observation
        is a (NUM_DRONES, share_obs_size)-shaped array of floats containing
        the normalized state of all drones.
        info : dict
            A dictionary containing additional information about the
            environment, such as the current state of each drone.
        -----------
        NOTE: This reset() function is capable for on-policy (MAPPO), with
        share obs outputs.
        -----------

        """
        # NOTE: This reset() function is capable for on-policy (MAPPO), with
        # share obs ouputs.

        #### Set initial states ####################################
        super(DroneBase, self).reset(seed=seed)
        self.step_counter = 0
        self.last_saved_states = []

        # Positions
        xy_indices = np.arange(self.NUM_DRONES)
        xy_coords = xy_indices * 4 * self.L
        self.INIT_XYZS = np.column_stack([xy_coords, xy_coords, np.ones(self.NUM_DRONES)])
        self.TARGET = self.INIT_XYZS.copy()
        self.next_TARGET = self.INIT_XYZS.copy()
        # Attitudes
        self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        # Crashed
        self.crashed = np.full(self.NUM_DRONES, False)
        # Finished
        self.finished = np.full(self.NUM_DRONES, False)
        # success
        self.success_rate = 0

        #### Random Init ###########################################
        self._randomInit()
        #### Generate Obstacles ######################################
        if self.NUM_OBSTACLES > 0:
            self._generateObstacles(min_dist_from_drones=2.0)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Return the initial observation ########################
        initial_obs, initial_share_obs = self._computeObs()
        #### Return the initial info ###############################
        initial_info = self._computeInfo()
        # run the track
        self.run_track = False

        if self.use_mappo:
            # return initial_obs, initial_share_obs, initial_info
            return initial_obs, initial_obs, initial_info
        else:
            # for SB3
            return initial_obs, initial_info

    ################################################################################

    def _actionSpace(self) -> spaces.Box:
        """Returns the action space of the environment.

        Returns
        -------
        Box
            The size of Box() depending on the action type.

        """
        if self.ACT_TYPE == ActionType.RATE:
            if self.DIM == SimulationDim.DIM_2:
                size = 2
            elif self.DIM == SimulationDim.DIM_3:
                size = 4
            else:
                print("[ERROR] in MultiDroneAgentBase._actionSpace(): Wrong self.DIM!")
        else:
            print("[ERROR] in MultiDroneAgentBase._actionSpace(): Wrong self.ACT_TYPE!")
        agent_action_space = spaces.Box(
            low=np.float32(-1 * np.ones(size)),
            high=np.float32(np.ones(size)),
            dtype=np.float32,
        )

        action_space = agent_action_space
        return action_space

    ################################################################################

    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """Pre-processes the action passed to `.step()` into real inputs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone.

        Parameters
        ----------
        action : ndarray[float]
            (NUM_DRONES, action_dim)-shaped array for multiple drones.
            The input action for drones, to be translated into real inputs.

        Returns
        -------
        ndarray[float] : [-1, 1]
            (NUM_DRONES, action_dim)-shaped array of inputs(Thrust and Rates etc.)
             for each drone.

        """
        if self.ACT_TYPE == ActionType.RATE:
            if self.DIM == SimulationDim.DIM_2:
                thrust = self.TWR_MAX * self.GRAVITY * (1 + action[:, 0]) / 2
                rate_x = self.MAX_RATE_XY * action[:, 1]
                zeros = np.zeros((self.NUM_DRONES,), dtype=np.float32)
                preprocess_action = np.column_stack([thrust, rate_x, zeros, zeros])
            elif self.DIM == SimulationDim.DIM_3:
                thrust = self.TWR_MAX * self.GRAVITY * (1 + action[:, 0]) / 2
                rate_xy = self.MAX_RATE_XY * action[:, 1:3]
                rate_z = self.MAX_RATE_Z * action[:, 3].reshape(-1, 1)
                preprocess_action = np.hstack([thrust.reshape(-1, 1), rate_xy, rate_z])
            else:
                print("[ERROR] in MultiDroneAgentBase._preprocessAction(): Wrong self.DIM!")
        else:
            print("[ERROR] in MultiDroneAgentBase._preprocessAction(): Wrong self.ACT_TYPE!")
        return preprocess_action

    ################################################################################

    def _observationSpace(self) -> spaces.Box:
        """Returns the observation space of the environment.

        Returns
        -------
        Box
            The size of Box() depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RACE:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 22 + NUM_RAYS (Relative pos 1,2 + absolute vel #
                    ########################## + rot matrix + last action + ray distances) #####
                    #### Observation vector ### X1 Y1 Z1 X2 Y2 Z2 Vx Vy Vz
                    #### Observation vector ### rot_x rot_y rot_z at ap aq ar
                    #### Observation vector ### ray_dist_1 ... ray_dist_N
                    obs_lower_bound = -1
                    obs_upper_bound = 1
                    ray_obs_size = self.NUM_RAYS if self.NUM_OBSTACLES > 0 else 0
                    agent_obs_space = spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        shape=(22 + ray_obs_size,),
                        dtype=np.float32,
                    )
                    ############################################################
                else:
                    print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.DIM!")
            else:
                print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.RACE_MULTI:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    # Self OBS SPACE OF SIZE 22 + NUM_RAYS (Relative pos 1,2 + absolute vel
                    ########################## + rot matrix + last action + ray distances) #####
                    # Other OBS SPACE OF SIZE 7*(NUM_DRONE - 1) (Rel pos + vel)
                    ########################## + drone_life ####################
                    #### Observation vector ### X1 Y1 Z1 X2 Y2 Z2 Vx Vy Vz
                    #### Observation vector ### rot_x rot_y rot_z at ap aq ar
                    #### Observation vector ### ray_dist_1 ... ray_dist_N
                    #### Observation vector ### (r_x r_y r_z v_x v_y v_z) * NUM_DRONE
                    obs_lower_bound = -1
                    obs_upper_bound = 1
                    ray_obs_size = self.NUM_RAYS if self.NUM_OBSTACLES > 0 else 0
                    obs_size = 22 + ray_obs_size + 7 * (self.NUM_DRONES - 1)
                    # obs_size = 22 + 8*(self.NUM_DRONES - 1)
                    agent_obs_space = spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        shape=(obs_size,),
                        dtype=np.float32,
                    )
                    ############################################################
                else:
                    print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.DIM!")
            else:
                print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.ACT_TYPE!")
        else:
            print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.OBS_TYPE!")

        obs_space = agent_obs_space
        return obs_space

    ################################################################################

    def _shareObservationSpace(self) -> spaces.Box:
        """Returns the share observation space of the environment.

        Returns
        -------
        Box
            The size of Box() depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RACE:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### OBS SPACE OF SIZE 22 (Relative pos 1,2 + absolute vel #
                    ########################## + rot matrix + last action) #####
                    #### Observation vector ### X1 Y1 Z1 X2 Y2 Z2 Vx Vy Vz
                    #### Observation vector ### rot_x rot_y rot_z at ap aq ar
                    obs_lower_bound = -1
                    obs_upper_bound = 1
                    share_obs_space = spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        shape=(self.self_obs_size,),
                        dtype=np.float32,
                    )
                    ############################################################
                else:
                    print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.DIM!")
            else:
                print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.RACE_MULTI:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    ############################################################
                    #### LOCAL OBS SPACE OF SIZE (22+7*(NUM_DRONE-1))*NUM_DRONES
                    #### GLOBAL POS SPACE OF SIZE 3*NUM_DRONE ##################
                    ############################################################
                    #### SHARE OBS SPACE OF SIZE (25+7*(NUM_DRONE-1))*NUM_DRONES
                    obs_lower_bound = -1
                    obs_upper_bound = 1
                    share_obs_space = spaces.Box(
                        low=np.float32(obs_lower_bound),
                        high=np.float32(obs_upper_bound),
                        shape=(self.share_obs_size,),
                        dtype=np.float32,
                    )
                    ############################################################
                else:
                    print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.DIM!")
            else:
                print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.ACT_TYPE!")
        else:
            print("[ERROR] in MultiDroneAgentBase._observationSpace(): Wrong self.OBS_TYPE!")
        return share_obs_space

    ################################################################################

    def _computeObs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the current observation of the environment.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the observation and the shared observation.
            The observation is a (NUM_DRONES, obs_size)-shaped array of floats
            containing the normalized state of each drone, and the shared observation
            is a (NUM_DRONES, share_obs_size)-shaped array of floats containing
            the normalized state of all drones.

        """
        state = self._clipAndNormalizeState()
        if self.OBS_TYPE == ObservationType.RACE:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    obs = state[:, : self.self_obs_size].reshape(self.NUM_DRONES, self.self_obs_size)
                    share_obs = obs
                else:
                    print("[ERROR] in MultiDroneAgentBase._computeObs(): Wrong self.DIM!")
            else:
                print("[ERROR] in MultiDroneAgentBase._computeObs(): Wrong self.ACT_TYPE!")
        elif self.OBS_TYPE == ObservationType.RACE_MULTI:
            if self.ACT_TYPE == ActionType.RATE:
                if self.DIM == SimulationDim.DIM_3:
                    obs = state[:, : self.obs_size].reshape(self.NUM_DRONES, self.obs_size)

                    # Get the shared observation
                    state_by_agent = state.reshape(self.NUM_DRONES, self.state_size)
                    share_obs = np.zeros((self.NUM_DRONES, self.share_obs_size), dtype=np.float32)

                    for agent_id in range(self.NUM_DRONES):
                        all_agents_state = state_by_agent.copy()
                        # change the order of the last agent and the current agent
                        if agent_id != self.NUM_DRONES - 1:
                            all_agents_state[[agent_id, self.NUM_DRONES - 1]] = all_agents_state[
                                [self.NUM_DRONES - 1, agent_id]
                            ]
                        # flatten the state of all agents
                        share_obs[agent_id] = all_agents_state.flatten()
                else:
                    print("[ERROR] in MultiDroneAgentBase._computeObs(): Wrong self.DIM!")
            else:
                print("[ERROR] in MultiDroneAgentBase._computeObs(): Wrong self.ACT_TYPE!")
        else:
            print("[ERROR] in MultiDroneAgentBase._computeObs(): Wrong self.OBS_TYPE!")

        # Add sensor noise if in evaluation mode
        if self.EVAL_MODE:
            obs = self._add_sensor_noise(obs, self.sensor_sigma)

        return obs.astype("float32"), share_obs.astype("float32")

    ################################################################################

    def _checkCollision(self, distance: Optional[float] = None) -> np.ndarray:
        """Returns the current collisions of the environment.
           To compute the truncated state of the environment.

        Parameters
        ----------
        distance : float, optional
            The distance to check for collisions. If None, use the default value.
            The default is None.

        Returns
        -------
        ndarray[bool]
            (NUM_DRONES,)-shaped array of collisions.True if drone is collision at this step,
            False otherwise.

            Only consider non-crashed drones. If a drone has crashed before this step,
            the state remains False, but after or with self.crashed,
            the state is True.

        """
        safe_threshold = self.COLLISION_R
        safe_distance = self.COLLISION_R * 2 + safe_threshold
        if distance is not None:
            safe_distance = distance  # user defined distance
        # Check for collisions
        collisions = np.full(self.NUM_DRONES, False)

        # filter out crashed and finished drones
        valid_mask = ~(self.crashed | self.finished)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            return collisions

        # get valid positions
        valid_positions = self.pos[valid_indices]

        # calculate distances between drones
        diff = valid_positions[:, np.newaxis, :] - valid_positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(distances, np.inf)

        # find pairs of drones that are within the safe distance
        collision_pairs = np.where(distances <= safe_distance)

        # unique pairs of drones
        if collision_pairs[0].size > 0:
            colliding_drones = np.unique(
                np.concatenate([valid_indices[collision_pairs[0]], valid_indices[collision_pairs[1]]])
            )
            collisions[colliding_drones] = True

        return collisions

    ################################################################################

    def _getObsSize(self) -> None:
        """Get the observation size of the environment."""
        # Ray distances are added to self_obs_size
        ray_obs_size = self.NUM_RAYS if self.NUM_OBSTACLES > 0 else 0
        if self.OBS_TYPE == ObservationType.RACE:
            self.self_obs_size = 22 + ray_obs_size
            self.other_obs_size = 0
            self.global_obs_size = 0
        elif self.OBS_TYPE == ObservationType.RACE_MULTI:
            self.self_obs_size = 22 + ray_obs_size
            self.other_obs_size = 7 * (self.NUM_DRONES - 1)  # 7 states per drone (rel pos + rel vel + rel dist)
            self.global_obs_size = 3
        else:
            print("[ERROR] in MultiDroneAgentBase._getObsSize(): Wrong self.OBS_TYPE!")
        self.obs_size = self.self_obs_size + self.other_obs_size
        self.state_size = self.self_obs_size + self.other_obs_size + self.global_obs_size
        self.share_obs_size = self.state_size * self.NUM_DRONES

    ################################################################################

    def _clipAndNormalizeState(self):
        """Normalizes all states to the [-1,1] range.

        example: Returns
        -------
        ndarray[float] : [-1, 1]
            (self.NUM_DRONES, state_size)-shaped array of floats containing
            the normalized state.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _stepNextTarget(self):
        """Change the goal, if env is not terminated

        example: generate random targets or gates

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _randomInit(self):
        """Change the initial state & target etc.

        example: generate random initial state

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeDones(self, Terminated: bool, Truncated: bool):
        """Computes the done state of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeExtra(self):
        """Computes the extra functions of the environment.

        Can be implemented in a subclass.

        """
        pass
