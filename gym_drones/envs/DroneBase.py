import importlib.resources as pkg_resources
import numpy as np
import gymnasium as gym
from gym_drones.utils.enums import DroneModel, SimulationDim
import yaml
from scipy.spatial.transform import Rotation
from scipy.stats import truncnorm
import os
from typing import Optional, Union


class DroneBase(gym.Env):
    """Base class for Drone Gymnasium env."""

    #### Set render mode ###########################################
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.SINGLE_TEST_DRONE,
        num_drones: int = 1,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
        sim_freq: int = 100,
        ctrl_freq: int = 100,
        domain_rand: bool = False,
        drone_model_path: Optional[Union[str, os.PathLike]] = None,
        num_obstacles: int = 0,
        obstacle_size: float = 0.5,
        ray_length: float = 5.0,
        num_rays: int = 12,
        obstacle_safe_dist: float = 0.3,
    ):
        """
        Initialization of a single agent drone env.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone param (detailed in an .yaml file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the environment.
        initial_xyzs : ndarray[float] | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys : ndarray[float] | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones
            in radians.
        sim_freq : int, optional
            The frequency at which simulator steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the controller steps.
        domain_rand : bool, optional
            Whether to apply domain randomization.
        drone_model_path : str | os.PathLike, optional
            The path to the folder containing the YAML file with drone parameters.
            If not provided, the default path will be used.
            The default path is `gym_drones/assets/<DRONE_MODEL>/<YAML>`.

        """

        #### Constants #############################################
        self.G = 9.8
        self.CTRL_FREQ = ctrl_freq
        self.SIM_FREQ = sim_freq

        if self.SIM_FREQ % self.CTRL_FREQ != 0:
            raise ValueError("[ERROR] in DroneBase.__init__(), " "sim_freq is not divisible by ctrl_freq.")

        self.SIM_STEPS_PER_CTRL = int(self.SIM_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1.0 / self.CTRL_FREQ
        self.SIM_TIMESTEP = 1.0 / self.SIM_FREQ

        self.domain_rand = domain_rand

        #### Drone Parameters ######################################
        self.NUM_DRONES = num_drones
        self.DRONE_MODEL = drone_model
        self.YAML = f"{self.DRONE_MODEL.value}.yaml"

        #### Load the drone properties from the .yaml file #########
        (
            self.M,
            self.L,
            self.TWR_MAX,
            self.J,
            self.J_INV,
            self.COLLISION_H,
            self.COLLISION_R,
            self.DRAG_COEFF_L,
            self.DRAG_COEFF_H,
            self.DELAY_W,
            self.DELAY_T,
            self.MAX_POS_XY,
            self.MAX_POS_Z,
            self.MAX_LIN_VEL_XY,
            self.MAX_LIN_VEL_Z,
            self.MAX_RATE_XY,
            self.MAX_RATE_Z,
            self.WAYPOINT_R,
            self.INIT_HEIGHT,
        ) = self._readYAMLParameters(drone_model_path)

        #### Compute constants #####################################
        self.GRAVITY = self.G * self.M
        self.MAX_THRUST = self.GRAVITY * self.TWR_MAX

        #### Obstacle Parameters ####################################
        self.NUM_OBSTACLES = num_obstacles
        self.OBSTACLE_SIZE = obstacle_size
        self.RAY_LENGTH = ray_length
        self.NUM_RAYS = num_rays
        self.OBSTACLE_SAFE_DIST = obstacle_safe_dist
        # Obstacles stored as (NUM_OBSTACLES, 3) array: [x, y, z] positions
        self.obstacles = np.zeros((self.NUM_OBSTACLES, 3))
        # Ray directions (NUM_RAYS, 3) - normalized direction vectors
        self._initRayDirections()

        #### Set initial states ####################################
        # Targets
        self.TARGET = np.zeros((self.NUM_DRONES, 3))
        self.next_TARGET = np.zeros((self.NUM_DRONES, 3))

        # Positions
        if initial_xyzs is None:
            self.INIT_XYZS = np.column_stack(
                (
                    np.arange(self.NUM_DRONES) * 4 * self.L,
                    np.arange(self.NUM_DRONES) * 4 * self.L,
                    np.ones(self.NUM_DRONES),
                )
            )
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES, 3):
            self.INIT_XYZS = initial_xyzs
        else:
            print("[ERROR] invalid initial_xyzs in DroneBase.__init__(), " "try initial_xyzs.reshape(NUM_DRONES,3)")

        # Attitudes
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in DroneBase.__init__(), " "try initial_rpys.reshape(NUM_DRONES,3)")

        # Velocities
        self.INIT_VELS = np.zeros((self.NUM_DRONES, 3))

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

    ################################################################################

    def step(
        self,
        action: np.ndarray,
    ) -> None:
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray[float] : [-1, 1]
            (action_dim,)-shaped array for single drone
            or (NUM_DRONES, action_dim)-shaped array for multiple drones.

            The input action for one or more drones, translated into real inputs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> None:
        """Reset the environment when episodes end.

        Parameters
        ----------
        seed : int, optional
            random seeds.
        options : dict[..], optional
            extra options.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def render(
        self,
        mode: str = "human",
        close: bool = False,
    ) -> None:
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        raise NotImplementedError

    ################################################################################

    def _housekeeping(self) -> None:
        """Housekeeping function.

        Allocation and zero-ing of the variables in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.last_action = np.hstack((np.ones((self.NUM_DRONES, 1)), np.zeros((self.NUM_DRONES, 3))))
        self.last_states = np.zeros((self.NUM_DRONES, 10))
        self.clipped_action = self.last_action.copy() * self.GRAVITY
        self.last_clipped_action = self.clipped_action.copy()
        #### Initialize the drones kinemaatic information ##########
        self.last_states[:, 0:3] = self.INIT_XYZS
        self.pos = self.INIT_XYZS
        self.rpy = self.INIT_RPYS
        self.quat = np.array([Rotation.from_euler("xyz", rpy, degrees=False).as_quat() for rpy in self.INIT_RPYS])
        self.rot = np.array([Rotation.from_quat(quat).as_matrix() for quat in self.quat])
        self.vel = self.INIT_VELS
        self.rate = np.zeros((self.NUM_DRONES, 3))
        self.thrust = np.full(self.NUM_DRONES, self.GRAVITY)

    ################################################################################

    def _getDroneStateVector(
        self,
        nth_drone: int,
    ) -> np.ndarray:
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The index of the drone for which to get the state vector.

        Returns
        -------
        ndarray[float]
            (29,)-shaped array of floats containing the state vector of the n-th drone.

        """
        state = np.hstack(
            [
                self.pos[nth_drone, :],
                self.quat[nth_drone, :],
                self.rpy[nth_drone, :],
                self.vel[nth_drone, :],
                self.rate[nth_drone, :],
                self.thrust[nth_drone],
                self.clipped_action[nth_drone, :],
                self.last_clipped_action[nth_drone, :],
                self.last_action[nth_drone, :],
            ]
        )
        return state.reshape(
            29,
        )

    ################################################################################

    def _getDroneTarget(
        self,
        nth_drone: int,
    ) -> np.ndarray:
        """Returns the target of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The index of the drone for which to get the target position.

        Returns
        -------
        ndarray[float]
            (3,)-shaped array of floats containing the target position.

        """
        return self.TARGET[nth_drone, :].reshape(
            3,
        )

    ################################################################################

    def _getDroneNextTarget(
        self,
        nth_drone: int,
    ) -> np.ndarray:
        """Returns the next target of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The index of the drone for which to get the next target position.

        Returns
        -------
        ndarray[float]
            (3,)-shaped array of floats containing the next target position.

        """
        return self.next_TARGET[nth_drone, :].reshape(
            3,
        )

    ################################################################################

    def _integrateQ(
        self,
        quat: np.ndarray,
        omega: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Returns quaternion after integrating the angular velocity.

        Parameters
        ----------
        quat : ndarray[float]
            (4,)-shaped array of floats containing the quaternion.
        omega : ndarray[float]
            (3,)-shaped array of floats containing the angular velocity.
        dt : float
            The time step.

        Returns
        -------
        ndarray[float]
            (4,)-shaped array of floats containing the new quaternion.

        Notes
        -----
        for quat [x, y, z, w] format.

        if you use [w, x, y, z] format, please change:

        lambda_ = np.array([
            [0, -p, -q, -r],
            [p,  0,  r, -q],
            [q, -r,  0,  p],
            [r,  q, -p,  0]
        ])

        """
        ##################################################
        ############### Formula derivation ###############
        # Quaternion multiplication for [w, x, y, z]
        # We can use either [p]_L·q or [q]_R·p
        # In this function, we choose the [q]_R·p format

        # Angular velocity vector:
        #   ω_n = [p, q, r]

        # Quaternion derivative:
        #   d_q = (1/2) * q_n × [0, ω_n]
        #       = (1/2) * [[0, ω_n]]_R · q_n
        #       = (1/2) * Λ · q_n
        # where [[0, ω_n]]_R is defined as Λ (lambda)

        # Integration Step:
        #   θ = ||ω_n|| * dt / 2

        # Quaternion Increment:
        #   q_increment = [cos(θ), (ω_n / ||ω_n||) * sin(θ)]

        # Quaternion Update:
        #   q_new = q_n × q_increment
        #         = q_n × [cos(θ), 0, 0, 0] + q_n × [0, (ω_n / ||ω_n||) * sin(θ)]
        #         = [[cos(θ), 0, 0, 0]]_R · q_n + [[0, (ω_n / ||ω_n||) * sin(θ)]] · q_n
        #         = (I * cos(θ) + (sin(θ) / ||ω_n||) * Λ) · q_n
        ############### Formula derivation ###############
        ##################################################

        omega_norm = np.linalg.norm(omega)
        p, q, r = omega
        if np.isclose(omega_norm, 0):
            return quat

        lambda_ = np.array([[0, r, -q, p], [-r, 0, p, q], [q, -p, 0, r], [-p, -q, -r, 0]])
        theta = omega_norm * dt / 2
        quat = np.dot(np.eye(4) * np.cos(theta) + 1 / omega_norm * np.sin(theta) * lambda_, quat)
        return quat

    ################################################################################

    def _integrateQ_vectorized(
        self,
        quats: np.ndarray,  # 形状为 (N, 4)
        omegas: np.ndarray,  # 形状为 (N, 3)
        dt: float,
    ) -> np.ndarray:
        """Returns quaternions after integrating the angular velocities.

        Parameters
        ----------
        quats : ndarray[float]
            (N, 4)-shaped array of floats containing the quaternions.
        omegas : ndarray[float]
            (N, 3)-shaped array of floats containing the angular velocities.
        dt : float
            The time step.

        Returns
        -------
        ndarray[float]
            (N, 4)-shaped array of floats containing the new quaternions.

        Notes
        -----
        for quat [x, y, z, w] format.
        """
        # compute the norms of the angular velocities
        omega_norms = np.linalg.norm(omegas, axis=1)  # shape (N,)

        # handle the case where all angular velocities are zero
        zero_mask = np.isclose(omega_norms, 0)
        if np.all(zero_mask):
            return quats.copy()  # if all are zero, return the original quats

        # get the components of the angular velocities
        p, q, r = omegas[:, 0], omegas[:, 1], omegas[:, 2]  # each shape (N,)

        # compute the thetas, cosines, and sines
        thetas = omega_norms * dt / 2  # shape (N,)
        cos_thetas = np.cos(thetas)  # shape (N,)
        sin_thetas = np.sin(thetas)  # shape (N,)

        # initialize the result array
        result_quats = np.zeros_like(quats)

        # handle the case where angular velocities are zero
        result_quats[zero_mask] = quats[zero_mask]

        # handle the case where angular velocities are non-zero
        non_zero_mask = ~zero_mask
        if np.any(non_zero_mask):
            # abstract the non-zero components
            nz_p, nz_q, nz_r = p[non_zero_mask], q[non_zero_mask], r[non_zero_mask]
            nz_quats = quats[non_zero_mask]
            nz_omega_norms = omega_norms[non_zero_mask]
            nz_cos_thetas = cos_thetas[non_zero_mask]
            nz_sin_thetas = sin_thetas[non_zero_mask]

            # create the lambda matrices batch
            batch_size = np.sum(non_zero_mask)
            lambdas = np.zeros((batch_size, 4, 4))

            # fill the lambda matrices
            lambdas[:, 0, 1] = nz_r
            lambdas[:, 0, 2] = -nz_q
            lambdas[:, 0, 3] = nz_p
            lambdas[:, 1, 0] = -nz_r
            lambdas[:, 1, 2] = nz_p
            lambdas[:, 1, 3] = nz_q
            lambdas[:, 2, 0] = nz_q
            lambdas[:, 2, 1] = -nz_p
            lambdas[:, 2, 3] = nz_r
            lambdas[:, 3, 0] = -nz_p
            lambdas[:, 3, 1] = -nz_q
            lambdas[:, 3, 2] = -nz_r

            # create the identity matrix batch
            identity_batch = np.tile(np.eye(4), (batch_size, 1, 1))

            # compute the rotation matrices
            rotation_matrices = identity_batch * nz_cos_thetas[:, None, None]
            rotation_matrices += np.einsum("ijk,i->ijk", lambdas, nz_sin_thetas / nz_omega_norms)

            # apply the rotation matrices to the quaternions
            result_quats[non_zero_mask] = np.einsum("ijk,ik->ij", rotation_matrices, nz_quats)

        return result_quats

    ################################################################################

    def _saveLastAction(
        self,
        action: np.ndarray,
    ) -> None:
        """Stores the most recent action into attribute `self.last_action`.

        The last action can be used to compute aerodynamic effects.

        Parameters
        ----------
        action : ndarray[float]
            (NUM_DRONES, action_dim)-shaped array containing the current input for each drone.

        """
        if self.DIM == SimulationDim.DIM_2:
            self.last_action = np.hstack((action.reshape(self.NUM_DRONES, 2), np.zeros((self.NUM_DRONES, 2))))
        elif self.DIM == SimulationDim.DIM_3:
            self.last_action = np.reshape(action, (self.NUM_DRONES, 4))

    ################################################################################

    def _saveLastStates(self) -> None:
        """Stores the most recent states into attribute `self.last_states`."""
        self.last_states = np.reshape(np.hstack([self.pos, self.quat, self.vel]), (self.NUM_DRONES, 10))

    ################################################################################

    def _readYAMLParameters(self, file_path: Optional[Union[str, os.PathLike]] = None) -> tuple:
        """Read drone parameters from a YAML file.

        This method reads various parameters related to the drone from a specified YAML file.
        The parameters include physical properties, drag coefficients, delays, and limits.

        Parameters
        ----------
        file_path : str | os.PathLike, optional
            The path to the folder containing the YAML file. If not provided,
            the default path will be used.
            The default path is `gym_drones/assets/<DRONE_MODEL>/<YAML>`.

        Returns
        -------

        tuple
            A tuple containing the following parameters:
            - M: float, mass of the drone
            - L: float, arm length of the drone
            - TWR_MAX: float, thrust-to-weight ratio maximum
            - J: ndarray, inertia matrix
            - J_INV: ndarray, inverse of the inertia matrix
            - COLLISION_H: float, collision height
            - COLLISION_R: float, collision radius
            - DRAG_COEFF_L: ndarray, drag coefficients for x, y, z axes
            - DRAG_COEFF_H: float, drag coefficient for height (z axes)
            - DELAY_W: float, delay for weight
            - DELAY_T: float, delay for thrust
            - MAX_POS_XY: float, maximum position in XY plane
            - MAX_POS_Z: float, maximum position in Z axis
            - MAX_LIN_VEL_XY: float, maximum linear velocity in XY plane
            - MAX_LIN_VEL_Z: float, maximum linear velocity in Z axis
            - MAX_RATE_XY: float, maximum rate in XY plane
            - MAX_RATE_Z: float, maximum rate in Z axis
            - WAYPOINT_R: float, waypoint radius
            - INIT_HEIGHT: float, initial height
        """
        if file_path is not None:
            yaml_file_path = os.path.join(file_path, "assets", self.DRONE_MODEL.value, self.YAML)
            if not os.path.exists(yaml_file_path):  # check if the file exists
                print(f"Warning: {yaml_file_path} not found. Using default path.")
                yaml_file_path = pkg_resources.files("gym_drones").joinpath("assets", self.DRONE_MODEL.value, self.YAML)
        else:
            yaml_file_path = pkg_resources.files("gym_drones").joinpath("assets", self.DRONE_MODEL.value, self.YAML)
        self.yaml_file_path = yaml_file_path
        with open(yaml_file_path, "r") as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)

            # Physical properties
            M, L = data["m"], data["arm"]
            TWR_MAX = data["twr_max"]  # thrust2weight_max
            IXX, IYY, IZZ = data["ixx"], data["iyy"], data["izz"]
            J = np.diag([IXX, IYY, IZZ])
            J_INV = np.linalg.inv(J)
            COLLISION_H, COLLISION_R = data["height"], data["radius"]

            # Drag coefficients
            DRAG_COEFF_X = data["drag_coeff_x"]
            DRAG_COEFF_Y = data["drag_coeff_y"]
            DRAG_COEFF_Z = data["drag_coeff_z"]
            DRAG_COEFF_H = data["drag_coeff_h"]
            DRAG_COEFF_L = np.array([DRAG_COEFF_X, DRAG_COEFF_Y, DRAG_COEFF_Z])

            # Delays
            DELAY_W, DELAY_T = data["delay_w"], data["delay_T"]

            # Limits
            MAX_POS_XY, MAX_POS_Z = data["max_pos_xy"], data["max_pos_z"]
            MAX_LIN_VEL_XY, MAX_LIN_VEL_Z = (
                data["max_lin_vel_xy"],
                data["max_lin_vel_z"],
            )
            MAX_RATE_XY, MAX_RATE_Z = data["max_rate_xy"], data["max_rate_z"]
            WAYPOINT_R = data["waypoint_radius"]
            INIT_HEIGHT = data["init_height"]

        return (
            M,
            L,
            TWR_MAX,
            J,
            J_INV,
            COLLISION_H,
            COLLISION_R,
            DRAG_COEFF_L,
            DRAG_COEFF_H,
            DELAY_W,
            DELAY_T,
            MAX_POS_XY,
            MAX_POS_Z,
            MAX_LIN_VEL_XY,
            MAX_LIN_VEL_Z,
            MAX_RATE_XY,
            MAX_RATE_Z,
            WAYPOINT_R,
            INIT_HEIGHT,
        )

    ################################################################################

    def saveYAMLParameters(self, save_path: Union[str, os.PathLike], verbose: int = 0) -> None:
        """Save the drone parameters to a YAML file.

        This method saves the current drone parameters to a specified YAML file.

        Parameters
        ----------
        save_path : str | os.PathLike
            The path where the YAML file will be saved.
        verbose : int, optional
            Verbosity level. If greater than 0, prints success message.
            Default is 0 (no message).
        """
        with open(self.yaml_file_path, "r") as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
        save_file_path = os.path.join(save_path, "assets", self.DRONE_MODEL.value, self.YAML)
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        with open(save_file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        if verbose > 0:
            print(f"Parameters successfully saved to {save_file_path}")

    ################################################################################

    def getDroneParameters(self) -> None:
        """Prints the drone parameters.

        Show the drone parameters read from the .yaml file.

        """
        params = (
            f"Loaded parameters from {self.yaml_file_path}:\n"
            f"  [PARAM] m: {self.M:.2f} kg, L: {self.L:.2f} m\n"
            f"  [PARAM] ixx: {self.J[0, 0]:.5f}, iyy: {self.J[1, 1]:.5f}, izz: {self.J[2, 2]:.5f}\n"
            f"  [PARAM] drag_x: {self.DRAG_COEFF_L[0]:.4f}, drag_y: {self.DRAG_COEFF_L[1]:.4f}, drag_z: {self.DRAG_COEFF_L[2]:.4f}\n"
            f"  [PARAM] drag_h: {self.DRAG_COEFF_H:.5f}, t2w: {self.TWR_MAX:.2f}"
        )
        print("-" * 50)
        print("[DRONE PARAMETERS]")
        print(params)

    ################################################################################

    def _dynamics(
        self,
        inputs: np.ndarray,
        nth_drone: int,
    ) -> None:
        """Dynamics implementation.

        Parameters
        ----------
        inputs : ndarray[float]
            (4,)-shaped array containing the current input for each drone.
        nth_drone : int
            The index of the drone for which to compute the dynamics.

        """
        #### Current state #########################################
        vel = self.vel[nth_drone, :]
        quat = self.quat[nth_drone, :]
        rate = self.rate[nth_drone, :]
        T = self.thrust[nth_drone]
        R_B_E = self.rot[nth_drone, :, :]  # rot body to earth
        DRAG_COEFF_L = self.DRAG_COEFF_L
        DRAG_COEFF_H = self.DRAG_COEFF_H

        #### domain randomization ##################################
        if self.domain_rand:
            scale_T = truncnorm(-1, 1, loc=1.0, scale=0.1).rvs()
            T = T * scale_T
            # DRAG_COEFF_L = DRAG_COEFF_L * np.random.uniform(0.5, 1.5, size=DRAG_COEFF_L.shape)
            DRAG_COEFF_L = DRAG_COEFF_L * np.random.uniform(0.8, 1.2, size=DRAG_COEFF_L.shape)

        #### state derivative ######################################
        d_position = vel
        body_vel = R_B_E.transpose() @ vel
        z_drag_velocity = np.array([0.0, 0.0, body_vel[0] ** 2 + body_vel[1] ** 2])
        d_velocity = (
            -1 * np.array([0.0, 0.0, self.G])
            + R_B_E  # gravity
            @ (
                np.array([0.0, 0.0, T])
                + DRAG_COEFF_L * body_vel  # thrust
                + DRAG_COEFF_H * z_drag_velocity  # linear drag
            )  # nonlinear drag
            / self.M
        )
        d_T = (inputs[0] - T) / self.DELAY_T  # throttle delay
        d_rate = (inputs[1:4] - rate) / self.DELAY_W  # rate delay

        self.pos[nth_drone, :] += d_position * self.SIM_TIMESTEP
        self.vel[nth_drone, :] += d_velocity * self.SIM_TIMESTEP
        self.quat[nth_drone, :] = self._integrateQ(quat, rate, self.SIM_TIMESTEP)
        # use ZYX-order, but output order is rpy
        self.rpy[nth_drone, :] = Rotation.from_quat(self.quat[nth_drone, :]).as_euler("xyz", degrees=False)
        self.rot[nth_drone, :, :] = Rotation.from_quat(self.quat[nth_drone, :]).as_matrix()
        self.rate[nth_drone, :] += d_rate * self.SIM_TIMESTEP
        self.thrust[nth_drone] += d_T * self.SIM_TIMESTEP

    ################################################################################

    def _dynamics_vectorized(
        self,
        inputs: np.ndarray,  # (NUM_DRONES, 4) 形状
        drone_indices: Optional[np.ndarray] = None,  # (NUM_DRONES,) 形状
    ) -> None:
        """Vectorized dynamics implementation.
        This method computes the dynamics for all drones in a vectorized manner.

        Parameters
        ----------
        inputs : ndarray[float]
            (NUM_DRONES, 4)-shaped array containing the current inputs for all drones.
        drone_indices : ndarray[float], optional
            (NUM_DRONES,)-shaped array containing the indices of the drones to be updated.
            If None, all drones will be updated.
        """
        if drone_indices is None:
            drone_indices = np.arange(self.NUM_DRONES)
        num_drones = len(drone_indices)
        if num_drones == 0:
            Warning("[WARNING] in DroneBase._dynamics_vectorized(), " "no drones to update.")
            return
        #### current state #########################################
        # select the drones to be updated
        inputs = inputs[drone_indices]
        vel = self.vel[drone_indices]
        quat = self.quat[drone_indices]
        rate = self.rate[drone_indices]
        T = self.thrust[drone_indices]
        R_B_E = self.rot[drone_indices]
        DRAG_COEFF_L = self.DRAG_COEFF_L
        DRAG_COEFF_H = self.DRAG_COEFF_H

        #### domain randomization ##################################
        if self.domain_rand:
            # create a truncnorm distribution for scaling thrust
            scale_T = truncnorm(-1, 1, loc=1.0, scale=0.1).rvs(size=num_drones)
            T = T * scale_T
            # create a uniform distribution for scaling drag coefficients
            random_factors = np.random.uniform(0.8, 1.2, size=(num_drones,) + DRAG_COEFF_L.shape)
            DRAG_COEFF_L = DRAG_COEFF_L * random_factors

        #### state derivative ######################################
        # position derivative
        d_position = vel

        # calculate body velocity
        body_vel = np.einsum("ijk,ik->ij", np.transpose(R_B_E, (0, 2, 1)), vel)

        # calculate nonlinear drag velocity, only in z direction
        z_drag_velocity = np.zeros((num_drones, 3))
        z_drag_velocity[:, 2] = body_vel[:, 0] ** 2 + body_vel[:, 1] ** 2

        # prepare gravity vector
        gravity = np.tile(np.array([0.0, 0.0, -self.G]), (num_drones, 1))

        # prepare thrust vectors
        thrust_vectors = np.zeros((num_drones, 3))
        thrust_vectors[:, 2] = T

        # compute linear drag
        linear_drag = np.multiply(body_vel, DRAG_COEFF_L)

        # compute nonlinear drag
        nonlinear_drag = DRAG_COEFF_H * z_drag_velocity

        # compute total force in body frame
        total_force_body = thrust_vectors + linear_drag + nonlinear_drag

        # rotate total body force to earth frame
        total_force_earth = np.einsum("ijk,ik->ij", R_B_E, total_force_body)

        # velocity derivative (acceleration)
        d_velocity = gravity + total_force_earth / self.M

        # inputs derivative
        d_T = (inputs[:, 0] - T) / self.DELAY_T  # thrust delay
        d_rate = (inputs[:, 1:4] - rate) / self.DELAY_W  # rate delay

        #### update states ######################################
        # update positions and velocities
        self.pos[drone_indices] += d_position * self.SIM_TIMESTEP
        self.vel[drone_indices] += d_velocity * self.SIM_TIMESTEP
        self.quat[drone_indices] = self._integrateQ_vectorized(quat, rate, self.SIM_TIMESTEP)
        # update euler angles and rotation matrices
        rotations = Rotation.from_quat(quat)
        self.rpy[drone_indices] = rotations.as_euler("xyz", degrees=False)
        self.rot[drone_indices] = rotations.as_matrix()
        # update rates and thrusts
        self.rate[drone_indices] += d_rate * self.SIM_TIMESTEP
        self.thrust[drone_indices] += d_T * self.SIM_TIMESTEP

    ################################################################################

    def _add_sensor_noise(
        self,
        x_real: np.ndarray,
        sensor_sigma: float,
    ) -> np.ndarray:
        """Adds Gaussian noise to the observation.

        Parameters
        ----------
        x_real : ndarray
            The real observation.

        Returns
        -------
        noisy_x : ndarray
            The noisy observation.

        """
        noise = np.random.normal(0, sensor_sigma, x_real.shape)
        noisy_x = x_real + noise
        return noisy_x

    ################################################################################

    def _initRayDirections(self) -> None:
        """Initialize ray directions for obstacle detection.

        Creates NUM_RAYS rays distributed in a horizontal circle around the drone.
        """
        if self.NUM_RAYS > 0:
            angles = np.linspace(0, 2 * np.pi, self.NUM_RAYS, endpoint=False)
            # Create rays in horizontal plane (xy) with slight vertical variation
            self.ray_directions = np.zeros((self.NUM_RAYS, 3))
            self.ray_directions[:, 0] = np.cos(angles)  # x component
            self.ray_directions[:, 1] = np.sin(angles)  # y component
            self.ray_directions[:, 2] = 0.0  # z component (horizontal plane)
            # Normalize
            norms = np.linalg.norm(self.ray_directions, axis=1, keepdims=True)
            self.ray_directions = self.ray_directions / (norms + 1e-8)
        else:
            self.ray_directions = np.zeros((0, 3))

    ################################################################################

    def _castRays(self, drone_pos: np.ndarray, drone_rot: np.ndarray) -> np.ndarray:
        """Cast rays from drone position and return distances to obstacles.

        Parameters
        ----------
        drone_pos : ndarray[float]
            (3,)-shaped array containing the drone position [x, y, z].
        drone_rot : ndarray[float]
            (3, 3)-shaped array containing the rotation matrix.

        Returns
        -------
        ndarray[float]
            (NUM_RAYS,)-shaped array containing the distance to nearest obstacle
            for each ray, or RAY_LENGTH if no obstacle is hit.
        """
        if self.NUM_OBSTACLES == 0 or self.NUM_RAYS == 0:
            return np.full(self.NUM_RAYS, self.RAY_LENGTH)

        # Transform ray directions to world frame using drone rotation
        ray_dirs_world = (drone_rot @ self.ray_directions.T).T  # (NUM_RAYS, 3)

        # Initialize distances to max ray length
        ray_distances = np.full(self.NUM_RAYS, self.RAY_LENGTH)

        # For each ray, find the closest intersection with any obstacle
        for ray_idx in range(self.NUM_RAYS):
            ray_dir = ray_dirs_world[ray_idx]
            ray_start = drone_pos

            # Check intersection with each obstacle (treated as spheres)
            for obs_idx in range(self.NUM_OBSTACLES):
                obs_center = self.obstacles[obs_idx]
                obs_radius = self.OBSTACLE_SIZE / 2.0

                # Vector from ray start to obstacle center
                to_obs = obs_center - ray_start

                # Project to_obs onto ray direction
                proj_length = np.dot(to_obs, ray_dir)

                # If projection is negative, obstacle is behind the ray start
                if proj_length < 0:
                    continue

                # Closest point on ray to obstacle center
                closest_point = ray_start + proj_length * ray_dir

                # Distance from closest point to obstacle center
                dist_to_center = np.linalg.norm(closest_point - obs_center)

                # If distance is less than obstacle radius, ray intersects
                if dist_to_center <= obs_radius:
                    # Calculate intersection distance
                    # Use Pythagorean theorem: proj_length^2 - (dist_to_center^2 - radius^2)
                    if dist_to_center < obs_radius:
                        # Ray starts inside obstacle
                        intersection_dist = 0.0
                    else:
                        # Calculate distance along ray to intersection point
                        half_chord = np.sqrt(obs_radius**2 - dist_to_center**2)
                        intersection_dist = proj_length - half_chord

                    # Update if this is the closest intersection so far
                    if intersection_dist >= 0 and intersection_dist < ray_distances[ray_idx]:
                        ray_distances[ray_idx] = intersection_dist

        return ray_distances

    ################################################################################

    def _generateObstacles(self, min_dist_from_drones: float = 2.0) -> None:
        """Generate random obstacles in the environment.

        Parameters
        ----------
        min_dist_from_drones : float, optional
            Minimum distance obstacles should be from initial drone positions.
        """
        if self.NUM_OBSTACLES == 0:
            return

        obstacles = []
        max_attempts = 100 * self.NUM_OBSTACLES

        for _ in range(self.NUM_OBSTACLES):
            attempt = 0
            while attempt < max_attempts:
                # Generate random position within bounds
                obs_pos = np.array(
                    [
                        self.np_random.uniform(-self.MAX_POS_XY / 2, self.MAX_POS_XY / 2),
                        self.np_random.uniform(-self.MAX_POS_XY / 2, self.MAX_POS_XY / 2),
                        self.np_random.uniform(
                            self.INIT_HEIGHT - self.MAX_POS_Z / 2, self.INIT_HEIGHT + self.MAX_POS_Z / 2
                        ),
                    ]
                )

                # Check distance from all drones
                too_close = False
                for drone_pos in self.INIT_XYZS:
                    dist = np.linalg.norm(obs_pos - drone_pos)
                    if dist < min_dist_from_drones:
                        too_close = True
                        break

                # Check distance from other obstacles
                for existing_obs in obstacles:
                    dist = np.linalg.norm(obs_pos - existing_obs)
                    if dist < self.OBSTACLE_SIZE:
                        too_close = True
                        break

                if not too_close:
                    obstacles.append(obs_pos)
                    break

                attempt += 1

            # If we couldn't place it, place it anyway (might overlap)
            if attempt >= max_attempts:
                obs_pos = np.array(
                    [
                        self.np_random.uniform(-self.MAX_POS_XY / 2, self.MAX_POS_XY / 2),
                        self.np_random.uniform(-self.MAX_POS_XY / 2, self.MAX_POS_XY / 2),
                        self.np_random.uniform(
                            self.INIT_HEIGHT - self.MAX_POS_Z / 2, self.INIT_HEIGHT + self.MAX_POS_Z / 2
                        ),
                    ]
                )
                obstacles.append(obs_pos)

        self.obstacles = np.array(obstacles)
        # print(f"[OBSTACLE DEBUG] Generated {len(self.obstacles)} obstacles at positions:\n{self.obstacles}")

    ################################################################################

    def _checkObstacleCollision(self, drone_pos: np.ndarray) -> bool:
        """Check if drone collides with any obstacle.

        Parameters
        ----------
        drone_pos : ndarray[float]
            (3,)-shaped array containing the drone position.

        Returns
        -------
        bool
            True if drone collides with an obstacle, False otherwise.
        """
        if self.NUM_OBSTACLES == 0:
            return False

        collision_radius = self.COLLISION_R + self.OBSTACLE_SIZE / 2.0

        for obs_pos in self.obstacles:
            dist = np.linalg.norm(drone_pos - obs_pos)
            if dist < collision_radius:
                return True

        return False

    ################################################################################

    def _actionSpace(self) -> None:
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _observationSpace(self) -> None:
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeObs(self) -> None:
        """Returns the current observation of drones.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _preprocessAction(self, action: np.ndarray) -> None:
        """Pre-processes the action passed to `.step()` into real inputs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray[float]
            (action_dim,)-shaped array for single drone
            or (NUM_DRONES, action_dim)-shaped array for multiple drones.

            The input action for one or more drones, to be real inputs.

        """
        raise NotImplementedError

    ################################################################################

    def _computeReward(
        self,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Computes the current reward value(s).

        Parameters
        ----------
        terminated : bool
            Whether the episode is terminated.
        truncated : bool | ndarray[bool]
            Whether the episode/drone is truncated.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeTerminated(self) -> None:
        """Computes the current terminated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeTruncated(self) -> None:
        """Computes the current truncated value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeInfo(self) -> None:
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _stepNextTarget(self) -> None:
        """Change the goal, if env is not terminated

        example: generate random targets or gates

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _randomInit(self) -> None:
        """Change the initial state & target etc.

        example: generate random initial state

        Must be implemented in a subclass.

        """
        raise NotImplementedError
