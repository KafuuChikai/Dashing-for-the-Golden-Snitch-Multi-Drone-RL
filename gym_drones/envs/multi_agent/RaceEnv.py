import numpy as np

from gym_drones.utils.enums import DroneModel
from gym_drones.envs.multi_agent.MultiDroneAgentBase import (
    ActionType,
    ObservationType,
    SimulationDim,
    MultiDroneAgentBase,
)

from scipy.spatial.transform import Rotation
from typing import Tuple, Union, Optional
import os


class RaceEnv(MultiDroneAgentBase):
    """Multi agent RL problem: waypoints."""

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
        alert_dist: Optional[float] = None,
        collision_radius: Optional[float] = None,
        num_obstacles: int = 0,
        obstacle_size: float = 0.5,
        ray_length: float = 5.0,
        num_rays: int = 12,
        obstacle_safe_dist: float = 0.3,
        obstacle_reward_weight: float = 1.0,
    ):
        """Initialization of a multi agent RL environment.

        Using the generic multi agent RL superclass.

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
            The frequency at which simulator steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        dim : SimulationDim, optional
            The dimension of the simulation (2D or 3D).
        episode_len_sec : float, optional
            The length of the episode in seconds.
        use_mappo : bool, optional
            Whether to use the MAPPO algorithm, if False, uses PPO(stable-baseline 3), else uses MAPPO(on-policy).
            default is False.
        eval_mode : bool, optional
            Whether the environment is in evaluation mode.
            If True, the environment will not truncate the episode when a drone crashes.
            default is False.
        sensor_sigma : float, optional
            The standard deviation of the sensor noise.
            default is 0.01.
        domain_rand : bool, optional
            Whether to use domain randomization.
            If True, the environment will randomize the drone model and the initial position of the drones.
            default is False.
        drone_model_path : str | os.PathLike, optional
            The path to the drone model file (.urdf).
            If None, the default drone model will be used.
            default is None.
        alert_dist : float | None, optional
            The distance at which the drone will be alerted of a potential collision.
            If None, it will be set to 3 times the collision radius.
            default is None.
        collision_radius : float | None, optional
            The radius at which the drone will be considered as collided.
            If None, it will be set to 2 times the collision radius.
            default is None.

        """
        if obs != ObservationType.RACE and obs != ObservationType.RACE_MULTI:
            print("[ERROR] in RaceEnv.__init__(): Wrong self.ObservationType!")
            exit()
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            obs=obs,
            act=act,
            dim=dim,
            episode_len_sec=episode_len_sec,
            use_mappo=use_mappo,
            eval_mode=eval_mode,
            sensor_sigma=sensor_sigma,
            domain_rand=domain_rand,
            drone_model_path=drone_model_path,
            num_obstacles=num_obstacles,
            obstacle_size=obstacle_size,
            ray_length=ray_length,
            num_rays=num_rays,
            obstacle_safe_dist=obstacle_safe_dist,
        )
        self.at_goal = np.full(self.NUM_DRONES, False)
        self.crashed_step = np.zeros(self.NUM_DRONES, dtype=int)
        self.finished_step = np.zeros(self.NUM_DRONES, dtype=int)
        self.Truncated = np.full(self.NUM_DRONES, False)
        self.active_masks = np.full(self.NUM_DRONES, True)
        self.alert_dist = 3 * self.COLLISION_R if alert_dist is None else alert_dist  # alert distance
        self.collision_dist = 2 * self.COLLISION_R if collision_radius is None else collision_radius  # collision radius
        self.obstacle_reward_weight = obstacle_reward_weight  # lambda_obs in reward equation
        # Beta parameter for exponential decay in obstacle reward (similar to drone safe reward)
        self.obstacle_beta = 15.0  # decay rate for obstacle proximity penalty

        # Debug: Print obstacle parameters
        if self.NUM_OBSTACLES > 0:
            print(
                f"[OBSTACLE DEBUG] num_obstacles={self.NUM_OBSTACLES}, num_rays={self.NUM_RAYS}, "
                f"obstacle_size={self.OBSTACLE_SIZE}, ray_length={self.RAY_LENGTH}, "
                f"obstacle_safe_dist={self.OBSTACLE_SAFE_DIST}, obstacle_reward_weight={self.obstacle_reward_weight}"
            )

    ################################################################################

    def _computeReward(self, Terminated, Truncated):
        """Computes the current reward value.

        Parameters
        ----------
        Terminated : bool
            Whether the current episode is terminated.
        Truncated : ndarray[bool]
            (self.NUM_DRONES,)-shaped array. Whether each drone is crashed.

        Returns
        -------
        ndarray
            (self.NUM_DRONES,)-shaped array. The reward.

        """

        ########### compute progress reward ############################
        # progress reward 1: distance to target
        weights = np.array([1.0, 1.0, 1.0])  # can be tuned
        target_dist_last = np.sqrt(np.sum(np.square((self.TARGET - self.last_states[:, 0:3]) * weights), axis=1))
        target_dist_current = np.sqrt(np.sum(np.square((self.TARGET - self.pos) * weights), axis=1))
        last_dist_rew = np.sqrt(np.clip(np.square(target_dist_last) - np.square(self.WAYPOINT_R * 0.75), 0, None))
        current_dist_rew = np.sqrt(
            np.clip(
                np.square(target_dist_current) - np.square(self.WAYPOINT_R * 0.75),
                0,
                None,
            )
        )
        prog_reward = last_dist_rew - current_dist_rew

        # progress reward 2: reward for reaching a waypoint
        prog_reward += 5 * (self.num_waypoints - self.last_num_waypoints)

        # reset prog_reward for crashed and finished drones
        self.finished_step[self.at_goal] = self.step_counter
        if Terminated or np.all(Truncated):
            self.finished_step[~np.logical_or(self.finished, self.at_goal)] = self.step_counter
        prog_reward[self.crashed | self.finished] = 0
        self.prog_reward = prog_reward  # for logging

        ########### compute command reward ##############################
        # command reward: rate of change of action
        rate = self.clipped_action[:, 1:4]
        action_diff = self.clipped_action - self.last_clipped_action
        command_reward = -2e-4 * np.linalg.norm(rate, axis=1) - 1e-4 * np.linalg.norm(action_diff, axis=1) ** 2
        # reset command_reward for crashed and finished drones
        command_reward[self.crashed | self.finished] = 0
        self.command_reward = command_reward  # for logging

        ########### compute drone safe reward ###########################
        # drone safe reward 1: according to relative velocity
        mask = ~np.eye(self.NUM_DRONES, dtype=bool)
        drone_rel_pos = self.pos[None, :, :] - self.pos[:, None, :]
        drone_rel_vel = self.vel[None, :, :] - self.vel[:, None, :]
        drone_coll_dist = np.linalg.norm(drone_rel_pos, axis=2)
        np.clip(drone_coll_dist, 1e-2, None, out=drone_coll_dist)
        rel_vel_norm = np.linalg.norm(drone_rel_vel, axis=2)
        np.clip(rel_vel_norm, 1e-2, None, out=rel_vel_norm)
        rel_close_vel = np.einsum("ijk,ijk->ij", drone_rel_pos, drone_rel_vel) / drone_coll_dist

        drone_safe_reward_vel = np.clip(
            rel_close_vel
            * np.clip(
                1 - (drone_coll_dist - self.alert_dist) / self.WAYPOINT_R,
                0,
                1,
            )
            ** 2,
            None,
            0,
        )
        drone_safe_reward_vel[:, self.crashed | self.finished] = 0  # reset for crashed and finished drones
        drone_safe_reward_vel = np.sum(
            drone_safe_reward_vel[mask].reshape(self.NUM_DRONES, self.NUM_DRONES - 1),
            axis=1,
        )

        # drone safe reward 2: according to relative distance
        weights = np.array([1.0, 1.0, 1.0])  # can be tuned
        drone_coll_dist_weighted = np.sqrt(np.sum(np.square(drone_rel_pos * weights), axis=2))

        vel_coef = np.clip((rel_close_vel / rel_vel_norm), -1, 0)
        vel_coef = vel_coef[mask].reshape(self.NUM_DRONES, self.NUM_DRONES - 1)
        drone_safe_reward_dist = np.clip(np.exp(-15 * (drone_coll_dist_weighted - self.collision_dist)), 0, 1)
        drone_safe_reward_dist[:, self.crashed | self.finished] = 0  # reset for crashed and finished drones
        drone_safe_reward_dist = np.sum(
            vel_coef * drone_safe_reward_dist[mask].reshape(self.NUM_DRONES, self.NUM_DRONES - 1),
            axis=1,
        )

        # combine the two drone safe rewards
        drone_safe_reward = 0.5 * drone_safe_reward_vel + 2.4 * drone_safe_reward_dist

        # reset drone safe reward for crashed and finished drones
        drone_safe_reward[self.crashed | self.finished] = 0
        self.drone_safe_reward = drone_safe_reward  # for logging

        ########### compute crash reward ################################
        crash_reward = np.zeros(self.NUM_DRONES)
        crash_reward[Truncated] = -30
        crash_reward[self.Collisions_alert] -= 0.5
        self.crash_reward = crash_reward  # for logging

        ########### compute obstacle avoidance reward ####################
        obstacle_reward = np.zeros(self.NUM_DRONES)
        if self.NUM_OBSTACLES > 0 and self.NUM_RAYS > 0:
            # Cast rays for each drone to detect obstacles
            for drone_idx in range(self.NUM_DRONES):
                if self.crashed[drone_idx] or self.finished[drone_idx]:
                    continue

                drone_pos = self.pos[drone_idx]
                drone_rot = self.rot[drone_idx]
                ray_distances = self._castRays(drone_pos, drone_rot)

                # Compute obstacle avoidance reward using exponential decay
                # r_obs_dist = sum_k min(exp(-beta * (d_k - d_safe)), 1)
                # where d_k is ray distance, d_safe is safety margin
                safe_dist = self.OBSTACLE_SAFE_DIST
                obs_reward_per_ray = np.minimum(np.exp(-self.obstacle_beta * (ray_distances - safe_dist)), 1.0)
                # Sum over all rays and normalize
                obstacle_reward[drone_idx] = -np.sum(obs_reward_per_ray) / self.NUM_RAYS

        # Reset obstacle reward for crashed and finished drones
        obstacle_reward[self.crashed | self.finished] = 0
        self.obstacle_reward = obstacle_reward  # for logging

        # compute total reward
        reward = (
            prog_reward
            + command_reward
            + drone_safe_reward
            + crash_reward
            + self.obstacle_reward_weight * obstacle_reward
        )

        return reward

    ################################################################################

    def _computeTerminated(self) -> bool:
        """Computes the current terminated value.

        Returns
        -------
        bool
            Whether the current episode is terminated.
            A flag to indicate the end of the episode,
            so it's bool instead of array.

        """
        return self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Returns
        -------
        ndarray[bool]
            (self.NUM_DRONES,)-shaped array. Whether each drone is crashed.

            Truncted is True if and only if the drone is crashed at current time step.
            And use self.crashed to record if the drone is crashed at any time step.

        """

        # compute collision alert conditions
        self.Collisions_alert = self._checkCollision(self.alert_dist)

        # count the number of real collisions
        real_collisions = self._checkCollision(self.COLLISION_R * 2)
        self.collisions_num += real_collisions

        # compute boundary conditions and combine with collision conditions
        MAX_Z = 3 * self.MAX_POS_Z
        target_z = self.TARGET[:, 2]
        current_z = self.pos[:, 2]

        # Check for obstacle collisions
        obstacle_collisions = np.zeros(self.NUM_DRONES, dtype=bool)
        if self.NUM_OBSTACLES > 0:
            for drone_idx in range(self.NUM_DRONES):
                if not (self.crashed[drone_idx] or self.finished[drone_idx]):
                    obstacle_collisions[drone_idx] = self._checkObstacleCollision(self.pos[drone_idx])

        # Collisions do not count as truncation during training
        if self.EVAL_MODE:
            Truncated = np.logical_or(
                np.logical_or(self._checkCollision(self.COLLISION_R * 2), obstacle_collisions),
                (np.abs(target_z - current_z) >= MAX_Z),
            )
        else:
            Truncated = np.logical_or(obstacle_collisions, (np.abs(target_z - current_z) >= MAX_Z))

        Truncated[self.crashed] = False  # crashed drones don't get truncated
        Truncated[self.finished] = False  # finished drones don't get truncated
        self.Truncated = Truncated
        return Truncated

    ################################################################################

    def _computeInfo(self) -> dict:
        """Computes the current info dict(s).

        Returns
        -------
        dict[...]

        """
        return {
            "target": self.TARGET.copy(),
            "prog_reward": np.mean(self.prog_reward.copy()),
            "command_reward": np.mean(self.command_reward.copy()),
            "crash_reward": np.mean(self.crash_reward.copy()),
            "drone_safe_reward": np.mean(self.drone_safe_reward.copy()),
            "obstacle_reward": np.mean(self.obstacle_reward.copy()) if self.NUM_OBSTACLES > 0 else 0.0,
            "num_waypoints": self.num_waypoints.copy(),
            "crash_rate": np.mean(self.crashed.copy()),
            # "finish_rate": np.mean(self.finished.copy()),
            "finish_rate": np.mean(~self.crashed.copy()),
            # "collisions_num": np.mean(self.collisions_num.copy()),
            "success_rate": self.success_rate,
            "active_mask": self.active_masks.copy(),
            "crashed": self.Truncated.copy(),
            "finished_step": self.finished_step.copy(),
            "crashed_step": self.crashed_step.copy(),
        }

    ################################################################################

    def _clipAndNormalizeState(self) -> np.ndarray:
        """Normalizes all states to the [-1,1] range.

        Returns
        -------
        ndarray[float] : [-1, 1]
            (self.NUM_DRONES, state_size)-shaped array of floats containing
            the normalized state.

        """
        # Constants for normalization
        MAX_XY = self.MAX_POS_XY
        MAX_Z = self.MAX_POS_Z
        MAX_LIN_VEL_XY = self.MAX_LIN_VEL_XY
        MAX_LIN_VEL_Z = self.MAX_LIN_VEL_Z
        MAX_DRONE_XY = self.MAX_POS_XY / 2
        MAX_DRONE_Z = self.MAX_POS_Z
        MAX_DRONE_D = self.MAX_POS_XY / 4
        MAX_DRONE_LIN_VEL_XY = self.MAX_LIN_VEL_XY
        MAX_DRONE_LIN_VEL_Z = self.MAX_LIN_VEL_Z

        # compute all relative positions and velocities
        mask = ~np.eye(self.NUM_DRONES, dtype=bool)
        target_rel_pos = self.TARGET - self.pos
        next_target_rel_pos = self.next_TARGET - self.TARGET
        drone_rel_pos = self.pos[None, :, :] - self.pos[:, None, :]
        drone_rel_vel = self.vel[None, :, :] - self.vel[:, None, :]

        # normalize all relative positions and velocities
        normalized_rel_xy = np.clip(target_rel_pos[:, :2] / MAX_XY, -1, 1)
        normalized_rel_z = np.clip(target_rel_pos[:, 2:] / MAX_Z, -1, 1)
        normalized_rel_xy_next = np.clip(next_target_rel_pos[:, :2] / MAX_XY, -1, 1)
        normalized_rel_z_next = np.clip(next_target_rel_pos[:, 2:] / MAX_Z, -1, 1)
        normalized_vel_xy = np.clip(self.vel[:, :2] / MAX_LIN_VEL_XY, -1, 1)
        normalized_vel_z = np.clip(self.vel[:, 2:] / MAX_LIN_VEL_Z, -1, 1)
        normalized_drone_pos_xy = np.clip(drone_rel_pos[..., :2] / MAX_DRONE_XY, -1, 1)
        normalized_drone_pos_z = np.clip(drone_rel_pos[..., 2:] / MAX_DRONE_Z, -1, 1)
        normalized_drone_vel_xy = np.clip(drone_rel_vel[..., :2] / MAX_DRONE_LIN_VEL_XY, -1, 1)
        normalized_drone_vel_z = np.clip(drone_rel_vel[..., 2:] / MAX_DRONE_LIN_VEL_Z, -1, 1)
        normalized_global_pos_xy = np.clip(self.pos[:, :2] / MAX_XY, -1, 1)
        normalized_global_pos_z = np.clip((self.pos[:, 2:] - self.INIT_HEIGHT) / MAX_Z / 2, -1, 1)
        normalized_drone_dist = np.clip(np.linalg.norm(drone_rel_pos, axis=2, keepdims=True) / MAX_DRONE_D, 0, 1)

        # Compute and normalize ray distances for obstacle detection
        normalized_ray_distances = np.zeros((self.NUM_DRONES, self.NUM_RAYS))
        if self.NUM_OBSTACLES > 0 and self.NUM_RAYS > 0:
            for drone_idx in range(self.NUM_DRONES):
                if not (self.crashed[drone_idx] or self.finished[drone_idx]):
                    ray_distances = self._castRays(self.pos[drone_idx], self.rot[drone_idx])
                    # Normalize ray distances to [0, 1] range (0 = obstacle at safe_dist, 1 = no obstacle within ray_length)
                    # Closer distances get lower normalized values
                    normalized_ray_distances[drone_idx] = np.clip(ray_distances / self.RAY_LENGTH, 0, 1)
                else:
                    normalized_ray_distances[drone_idx] = 1.0  # No obstacles detected for crashed/finished drones

        # stack all normalized self observations
        obs_components = [
            normalized_rel_xy,
            normalized_rel_z,
            normalized_rel_xy_next,
            normalized_rel_z_next,
            normalized_vel_xy,
            normalized_vel_z,
            self.rot.reshape(self.NUM_DRONES, 9),
            self.last_action,
        ]
        # Add ray distances if obstacles are enabled
        if self.NUM_OBSTACLES > 0 and self.NUM_RAYS > 0:
            obs_components.append(normalized_ray_distances)

        self_obs = np.hstack(obs_components).reshape(self.NUM_DRONES, self.self_obs_size)

        if self.OBS_TYPE == ObservationType.RACE_MULTI:
            # stack all normalized other observations

            # None used drone life
            # drone_life = np.ones((self.NUM_DRONES, self.NUM_DRONES, 1))
            # other_obs = np.concatenate((normalized_drone_pos_xy,
            #                             normalized_drone_pos_z,
            #                             normalized_drone_vel_xy,
            #                             normalized_drone_vel_z,
            #                             drone_life), axis=2)    # (NUM_DRONES, NUM_DRONES, 7)

            other_obs = np.concatenate(
                (
                    normalized_drone_pos_xy,
                    normalized_drone_pos_z,
                    normalized_drone_vel_xy,
                    normalized_drone_vel_z,
                    normalized_drone_dist,
                ),
                axis=2,
            )  # (NUM_DRONES, NUM_DRONES, 7)

            # None used drone life
            # for i, crashed in enumerate(self.crashed):
            #     if crashed:
            #         # other_obs[:, i, :] = 0
            #         other_obs[:, i, 7] = 0

            # for i, finished in enumerate(self.finished):
            #     if finished:
            #         # other_obs[:, i, :] = 0
            #         other_obs[:, i, 7] = 0

            other_obs = other_obs[mask].reshape(self.NUM_DRONES, -1)

            # stack all normalized global observations
            global_obs = np.hstack([normalized_global_pos_xy, normalized_global_pos_z]).reshape(
                self.NUM_DRONES, self.global_obs_size
            )

            # stack all normalized states
            state = np.hstack([self_obs, other_obs, global_obs]).reshape(self.NUM_DRONES, self.state_size)
        else:
            state = self_obs

        for i, crashed in enumerate(self.crashed):
            if crashed:
                state[i, :] = 0

        for i, finished in enumerate(self.finished):
            if finished:
                state[i, :] = 0

        return state

    ################################################################################

    def _stepNextTarget(self) -> None:
        """Updates the next target for each drone based on the current position and waypoints.

        This method is called at each step of the environment to update the target position"""
        distance_to_target = np.linalg.norm(self.TARGET - self.pos, axis=1)
        self.at_goal = np.full(self.NUM_DRONES, False)

        # Check if any drone has reached the target
        reached_target = (distance_to_target <= self.WAYPOINT_R) & ~self.crashed & ~self.finished
        if not np.any(reached_target):
            return

        # Update the number of waypoints for each drone that reached the target
        self.num_waypoints[reached_target] += 1

        if not self.run_track:
            self._updateTargetsCircular(reached_target)
        else:
            self._updateTargetsLinear(reached_target)

    ################################################################################

    def _updateTargetsCircular(self, reached_target: np.ndarray) -> None:
        """Updates the targets for each drone in a circular manner.

        This method is called when the drones are running in a circular track."""
        reached_indices = np.where(reached_target)[0]

        if len(reached_indices) == 0:
            return

        if self.same_waypoints:
            # vectorized calculation of target and next target indices
            target_indices = self.num_waypoints[reached_indices] % len(self.waypoints)
            next_target_indices = (self.num_waypoints[reached_indices] + 1) % len(self.waypoints)

            # vectorized update of targets
            self.TARGET[reached_indices] = self.waypoints[target_indices]
            self.next_TARGET[reached_indices] = self.waypoints[next_target_indices]
        else:
            # vectorized calculation of target and next target indices
            target_indices = self.num_waypoints[reached_indices] % self.waypoints.shape[1]
            next_target_indices = (self.num_waypoints[reached_indices] + 1) % self.waypoints.shape[1]

            # vectorized update of targets
            self.TARGET[reached_indices] = self.waypoints[reached_indices, target_indices, :]
            self.next_TARGET[reached_indices] = self.waypoints[reached_indices, next_target_indices, :]

    ################################################################################

    def _updateTargetsLinear(self, reached_target: np.ndarray) -> None:
        """Updates the targets for each drone in a linear manner.

        This method is called when the drones are running in a linear track."""
        reached_indices = np.where(reached_target)[0]

        if len(reached_indices) == 0:
            return

        if self.same_waypoints:
            max_waypoints = len(self.waypoints)
        else:
            max_waypoints = self.waypoints.shape[1]

        # check which drones have completed all waypoints
        completed_mask = self.num_waypoints[reached_indices] >= max_waypoints
        completed_indices = reached_indices[completed_mask]
        active_indices = reached_indices[~completed_mask]

        # mark drones that have reached their goal
        self.at_goal[completed_indices] = True

        # update targets for active drones
        if len(active_indices) > 0:
            if self.same_waypoints:
                target_indices = self.num_waypoints[active_indices]
                self.TARGET[active_indices] = self.waypoints[target_indices]

                # handle next_target: check if it's the last waypoint
                last_waypoint_mask = (self.num_waypoints[active_indices] + 1) >= max_waypoints
                next_target_indices = np.where(
                    last_waypoint_mask,
                    self.num_waypoints[active_indices],  # if it's the last waypoint, use current target
                    self.num_waypoints[active_indices] + 1,  # otherwise, use the next waypoint
                )
                self.next_TARGET[active_indices] = self.waypoints[next_target_indices]
            else:
                target_indices = self.num_waypoints[active_indices]
                self.TARGET[active_indices] = self.waypoints[active_indices, target_indices, :]

                # handle next_target: check if it's the last waypoint
                last_waypoint_mask = (self.num_waypoints[active_indices] + 1) >= max_waypoints
                next_target_indices = np.where(
                    last_waypoint_mask,
                    self.num_waypoints[active_indices],  # if it's the last waypoint, use current target
                    self.num_waypoints[active_indices] + 1,  # otherwise, use the next waypoint
                )
                self.next_TARGET[active_indices] = self.waypoints[active_indices, next_target_indices, :]

    ################################################################################

    def _randomInit(self) -> None:
        """Initializes the environment with random values.

        This method is called at the beginning of each episode."""
        #### initial reward check callback #########################
        self.prog_reward = np.zeros(self.NUM_DRONES)
        self.command_reward = np.zeros(self.NUM_DRONES)
        self.crash_reward = np.zeros(self.NUM_DRONES)
        self.drone_safe_reward = np.zeros(self.NUM_DRONES)
        self.drone_race_reward = np.zeros(self.NUM_DRONES)
        self.obstacle_reward = np.zeros(self.NUM_DRONES)

        if self.DIM == SimulationDim.DIM_3:
            safe_distance = self.WAYPOINT_R
            positions = []

            # Random initialization of the drones
            for _ in range(self.NUM_DRONES):
                while True:
                    position = np.array(
                        np.random.uniform(
                            low=-self.NUM_DRONES * safe_distance / 2,
                            high=self.NUM_DRONES * safe_distance / 2,
                            size=(2,),
                        ).tolist()
                        + [self.INIT_HEIGHT]
                    )
                    # Check if the drone is not too close to the others
                    if all(np.linalg.norm(position - p) >= safe_distance for p in positions):
                        positions.append(position)
                        break

            # set the initial positions and orientations
            self.INIT_XYZS = np.array(positions)
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))

            # Crashed
            self.crashed = np.full(self.NUM_DRONES, False)
            self.finished = np.full(self.NUM_DRONES, False)
            self.crashed_step = np.zeros(self.NUM_DRONES, dtype=int)
            self.finished_step = np.zeros(self.NUM_DRONES, dtype=int)
            self.collisions_num = np.zeros(self.NUM_DRONES)

            # set the initial waypoints and targets
            self.same_waypoints = True
            self.num_waypoints = np.zeros(self.NUM_DRONES, dtype=int)
            self.last_num_waypoints = np.zeros(self.NUM_DRONES, dtype=int)

            waypoints = [
                np.hstack(
                    [
                        self.np_random.uniform(low=-self.MAX_POS_XY / 2, high=self.MAX_POS_XY / 2, size=(2,)),
                        self.np_random.uniform(
                            low=self.INIT_HEIGHT - self.MAX_POS_Z / 2,
                            high=self.INIT_HEIGHT + self.MAX_POS_Z / 2,
                            size=(1,),
                        ),
                    ]
                )
            ]

            for i in range(30):
                temp_waypoint = np.hstack(
                    [
                        self.np_random.uniform(low=-self.MAX_POS_XY / 2, high=self.MAX_POS_XY / 2, size=(2,)),
                        self.np_random.uniform(
                            low=self.INIT_HEIGHT - self.MAX_POS_Z / 2,
                            high=self.INIT_HEIGHT + self.MAX_POS_Z / 2,
                            size=(1,),
                        ),
                    ]
                )
                waypoints.append(temp_waypoint)
            self.waypoints = np.array(waypoints)

            self.TARGET = np.tile(self.waypoints[0], (self.NUM_DRONES, 1))
            self.next_TARGET = np.tile(self.waypoints[1], (self.NUM_DRONES, 1))

            # random_init_rpys = np.hstack([self.np_random.uniform(low=-np.pi, high=np.pi, size=(2,)),
            #                               self.np_random.uniform(low=0, high=2*np.pi, size=(1,))])
            # random_init_vels = np.hstack([self.np_random.uniform(low=-15, high=15, size=(2,)),
            #                               self.np_random.uniform(low=-3, high=3, size=(1,))])
            # self.INIT_RPYS = np.tile(random_init_rpys, (self.NUM_DRONES, 1))
            # self.INIT_VELS = np.tile(random_init_vels, (self.NUM_DRONES, 1))

        else:
            print("[ERROR] in RaceEnv.reset(): Wrong self.DIM!")

    ################################################################################

    def _computeDones(self, Terminated: bool, Truncated: np.ndarray) -> np.ndarray:
        """Computes the current done value(s).

        Parameters
        ----------
        Terminated : bool
            Whether the current episode is terminated.
        Truncated : ndarray[bool]
            (self.NUM_DRONES,)-shaped array. Whether each drone is crashed.

        Returns
        -------
        ndarray[bool]
            (self.NUM_DRONES,)-shaped array.
            Whether each drone is done (input of MAPPO).

        """
        # update active_masks, before crash and finished, because they are used in the reward computation
        self.active_masks = ~np.logical_or(self.crashed, self.finished)
        # update last number of waypoints
        for i in range(self.NUM_DRONES):
            if self.num_waypoints[i] > self.last_num_waypoints[i]:
                self.last_num_waypoints[i] = self.num_waypoints[i]
        # update crashed drones
        if Truncated.any():
            self.pos[Truncated, :] = 0
            self.vel[Truncated, :] = 0
            self.TARGET[Truncated, :] = 0
            self.quat[Truncated, :] = Rotation.from_euler("xyz", [0, 0, 0]).as_quat()
            self.thrust[Truncated] = 0
            self.rate[Truncated, :] = 0
            self.clipped_action[Truncated, :] = 0
        self.crashed_step[Truncated] = self.step_counter
        self.crashed = np.logical_or(self.crashed, Truncated)
        # update finished drones
        if self.at_goal.any():
            self.pos[self.at_goal, :] = 0
            self.vel[self.at_goal, :] = 0
            self.TARGET[self.at_goal, :] = 0
            self.quat[self.at_goal, :] = Rotation.from_euler("xyz", [0, 0, 0]).as_quat()
            self.thrust[self.at_goal] = 0
            self.rate[self.at_goal, :] = 0
            self.clipped_action[self.at_goal, :] = 0
        self.finished = np.logical_or(self.finished, self.at_goal)
        # success rate
        # if Terminated and np.all(self.finished):
        if Terminated and ~np.any(self.crashed):
            self.success_rate = 1
        return np.full(self.NUM_DRONES, True) if Terminated else self.crashed

    ################################################################################

    def _computeExtra(self) -> None:
        """Computes the extra functions of the environment."""
        #### Change to next waypoint ###############################
        self._stepNextTarget()

    ################################################################################

    def _setWaypoints_reset(
        self,
        waypoints: np.ndarray,
        waypoints_radius: float = None,
        seed: int = None,
        options: dict = None,
    ) -> Tuple:
        """reset method for setting specific waypoints

        parameters
        ----------
        waypoints : ndarray
            (num_waypoints, 3) or (num_drones, num_waypoints, 3)-shaped array containing the desired XYZ position of the drone.
        waypoints_radius : float
            the radius of the waypoints.

        returns
        -------
        initial_obs : ndarray
            (NUM_DRONES, OBS_DIM)-shaped array containing the initial observation of the drone.
        initial_info : dict
            Dictionary containing the initial info of the drone.

        """
        super().reset(seed=seed, options=options)

        self.run_track = True
        # get waypoints infos
        if waypoints.ndim == 2:
            self.same_waypoints = True
        elif waypoints.ndim == 3:
            self.same_waypoints = False

        # update waypoints
        self.waypoints = waypoints
        if waypoints_radius is not None:
            self.WAYPOINT_R = waypoints_radius

        # set initial target
        if self.same_waypoints:
            self.TARGET = np.tile(self.waypoints[0], (self.NUM_DRONES, 1)).copy()
            self.next_TARGET = np.tile(self.waypoints[1], (self.NUM_DRONES, 1)).copy()
        else:
            # reset initial position
            self.INIT_XYZS = self.waypoints[:, 0, :].reshape(self.NUM_DRONES, 3).copy()
            self.TARGET = self.waypoints[:, 0, :].reshape(self.NUM_DRONES, 3).copy()
            self.next_TARGET = self.waypoints[:, 1, :].reshape(self.NUM_DRONES, 3).copy()
        self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        self.INIT_VELS = np.zeros((self.NUM_DRONES, 3))
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Return the initial observation ########################
        initial_obs, initial_share_obs = self._computeObs()
        #### Return the initial info ###############################
        initial_info = self._computeInfo()

        if self.use_mappo:
            # return initial_obs, initial_share_obs, initial_info
            return initial_obs, initial_obs, initial_info
            # return initial_share_obs, initial_share_obs, initial_info
        else:
            # for SB3
            return initial_obs, initial_info

    ################################################################################

    def _setWaypoints(self, waypoints: np.ndarray) -> None:
        """Setting specific waypoints

        Parameters
        ----------
        waypoints : ndarray
            (num_waypoints, 3) or (num_drones, num_waypoints, 3)-shaped array containing the desired XYZ position of the drone.

        """
        # get waypoints infos
        if waypoints.ndim == 2:
            self.same_waypoints = True
        elif waypoints.ndim == 3:
            self.same_waypoints = False

        # update waypoints
        self.waypoints = waypoints

        # set initial target
        if self.same_waypoints:
            for nth_drone in range(self.NUM_DRONES):
                self.TARGET[nth_drone] = self.waypoints[self.num_waypoints[nth_drone] % len(self.waypoints)]
                self.next_TARGET[nth_drone] = self.waypoints[(self.num_waypoints[nth_drone] + 1) % len(self.waypoints)]
        else:
            for nth_drone in range(self.NUM_DRONES):
                self.TARGET[nth_drone] = self.waypoints[
                    nth_drone,
                    self.num_waypoints[nth_drone] % self.waypoints.shape[1],
                    :,
                ]
                self.next_TARGET[nth_drone] = self.waypoints[
                    nth_drone,
                    (self.num_waypoints[nth_drone] + 1) % self.waypoints.shape[1],
                    :,
                ]
