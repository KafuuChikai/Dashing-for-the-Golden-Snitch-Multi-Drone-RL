import os, yaml
import numpy as np
import torch as th
import gymnasium as gym
from typing import Union, Optional, Callable, Tuple

from stable_baselines3 import PPO

from gym_drones.utils.Logger import Logger
from gym_drones.utils.enums import SimulationDim
from gym_drones.utils.rl_manager.EvalCallback import (
    EvalBaseCallback,
    ConvertCallback,
    EvalCallbackList,
    EvalRewardCallback,
    EvalTimeCallback,
)
from gym_drones.utils.utils import get_latest_run_id
from gym_drones.utils.rl_manager.config import _save_config
from gym_drones.utils.motion_library import GateMotionPlanner


def _get_predict_fn(
    model: Union[PPO, th.nn.Module], config_dict: dict
) -> Callable[[np.ndarray], Union[np.ndarray, tuple]]:
    """Get the prediction function for the model.

    Parameters
    ----------
    model : Union[PPO, th.nn.Module]
        The model to be evaluated.
    config_dict : dict
        The configuration dictionary containing the evaluation parameters.

    Returns
    -------
    Callable[[np.ndarray], Union[np.ndarray, tuple]]
        The prediction function for the model.
        In the case of Stable-Baselines3 PPO, it returns a tuple (action, state).
        In the case of PyTorch model, it returns the output of the model.
    """
    if isinstance(model, PPO):
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print("Evaluating a Stable-Baselines3 PPO model. Starting evaluation...")
        predict_fn = lambda obs: model.predict(obs, deterministic=True)[0]
    elif isinstance(model, th.nn.Module):
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print("Evaluating a PyTorch model. Starting evaluation...")
        device = th.device("cuda" if config_dict["pyrl"]["use_cuda"] else "cpu")
        output_activation_fn = config_dict["agent"].get("output_activation_fn", None)
        if output_activation_fn is not None:
            output_activation_fn = output_activation_fn()
            predict_fn = lambda obs: output_activation_fn(model(th.from_numpy(obs).to(device))).detach().cpu().numpy()
        else:
            predict_fn = lambda obs: np.clip(model(th.from_numpy(obs).to(device)).detach().cpu().numpy(), -1, 1)
    else:
        raise TypeError("Unsupported model type. Expected PPO or th.nn.Module.")
    return predict_fn


def _get_save_path(config_dict: dict, current_dir: Union[str, os.PathLike]) -> os.PathLike:
    """Get the save path for the evaluation results.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary containing the evaluation parameters.
    current_dir : str or os.PathLike
        The current directory where the results will be saved.

    Returns
    -------
    os.PathLike
        The path where the evaluation results will be saved.

    """
    # get the save path
    model_name = config_dict["logging"]["save_model_name"]
    model_id = config_dict["logging"]["run_id"]
    eval_save_dirname = config_dict["logging"].get("save_eval_path", config_dict["logging"]["save_config_dirname"])
    eval_save_path = os.path.join(
        current_dir,
        eval_save_dirname,
        config_dict["pyrl"]["exp_name"],
        f"{model_name}_{model_id + 1}",
        "evals",
    )
    eval_name = config_dict["pyrl"].get("eval_name", config_dict["pyrl"]["exp_name"])
    eval_id = get_latest_run_id(eval_save_path, eval_name)
    if eval_id > 0 and config_dict["logging"]["eval_overwrite"]:
        eval_id -= 1
    config_dict["logging"]["eval_id"] = eval_id
    output_folder = os.path.join(
        eval_save_path,
        f"{eval_name}_{eval_id + 1}",
    )

    # show the evaluation details
    if config_dict["rl_hyperparams"]["verbose"] > 0:
        print("-" * 50)
        print("[EVALUATION DETAILS]")
        print(f"Evaluation name: {eval_name}_{eval_id + 1}")
        print(f"Evaluating {model_name}_{model_id + 1} on {config_dict['env']['env_id']}")
    return output_folder


def _init_callback(
    env: gym.Env,
    callback: Optional[EvalBaseCallback] = None,
) -> EvalBaseCallback:
    """Initialize the callback.

    Parameters
    ----------
    env : gym.Env
        The environment to be used for evaluation.
    callback : Optional[EvalBaseCallback], optional
        The callback to be used for evaluation, by default None
        If None, a default callback will be used.
        If a list, it will be converted to a CallbackList.

    Returns
    -------
    EvalBaseCallback
        The initialized callback.
    """
    # Convert a list of callbacks into a callback
    if isinstance(callback, list):
        callback = EvalCallbackList(callback)

    # Convert functional callback to object
    if not isinstance(callback, EvalBaseCallback):
        callback = ConvertCallback(callback)

    callback.init_callback(env)
    return callback


def _calculate_path_length(waypoints: np.ndarray) -> float:
    """calculate the path length of the waypoints.

    Parameters
    ----------
    waypoints : np.ndarray
        The waypoints of the track.

    Returns
    -------
    float
        The path length of the waypoints.

    """
    total_length = 0
    for i in range(1, len(waypoints)):
        p1, p2 = waypoints[i - 1], waypoints[i]
        segment_length = np.sqrt(np.sum((p2 - p1) ** 2))
        total_length += segment_length
    return total_length


def _output_track_info(
    track_name: str,
    track_num_drones: int,
    same_track: bool,
    repeat_lap: int,
    WPs: np.ndarray,
    start_points: np.ndarray,
    end_points: np.ndarray,
    verbose: int = 0,
) -> None:
    """Output the track information.

    Parameters
    ----------
    track_name : str
        The name of the track.
    track_num_drones : int
        The number of drones in the track.
    same_track : bool
        Whether the drones are flying the same track or not.
    repeat_lap : int
        The number of laps to be repeated.
    WPs : np.ndarray
        The waypoints of the track.
    start_points : np.ndarray
        The start points of the track.
    end_points : np.ndarray
        The end points of the track.
    verbose : int, optional
        The verbosity level, by default 0
        0: no output
        1: basic output
        2: detailed output

    """
    if verbose > 0:
        print("[TRACK INFO]")
        print(f"Track name: {track_name}")
        print(f"Same track: {same_track}")
        print(f"Repeat lap: {repeat_lap}")

        if verbose > 1:
            start_points_len = len(start_points[0])
            end_points_len = len(end_points[0])
            track_len = len(WPs[0])
            loop_len = (track_len - start_points_len - end_points_len) / repeat_lap

            for nth_drone in range(track_num_drones):
                # initialize the counters
                start_count = 0
                end_count = 0
                loop_count = 0
                # get the waypoints for the drone
                waypoints = WPs[nth_drone]
                if same_track:
                    header = f"{track_name} - Same track for all drones"
                else:
                    header = f"{track_name} - Drone {nth_drone + 1}/{track_num_drones}"
                print("\n" + "=" * 50)
                print(f"{header:^50}")
                print("=" * 50)
                print("{:<5} {:<15} {:<15} {:<15}".format("Pt#", "X-coord(m)", "Y-coord(m)", "Z-coord(m)"))
                print("-" * 50)

                for i, point in enumerate(waypoints):
                    loop_name = ""
                    if (
                        (i - start_points_len) % loop_len == 0
                        and i >= start_points_len
                        and i <= len(waypoints) - end_points_len
                    ):
                        print("-" * 50)
                        if i < len(waypoints) - end_points_len:
                            loop_count += 1
                            loop_wp_id = 1
                            loop_name = f" (Loop {loop_count}/{repeat_lap})"

                    point_name = ""
                    if i < start_points_len:
                        start_count += 1
                        point_name = f"Waypoint{start_count}"
                        if i == 0:
                            point_name += " (Start)"
                    elif i >= len(waypoints) - end_points_len:
                        end_count += 1
                        point_name = f"Waypoint{end_count}"
                        if i == len(waypoints) - end_points_len:
                            point_name += " (End)"
                    else:
                        point_name = f"Waypoint{loop_wp_id}" + loop_name
                        loop_wp_id += 1

                    print(
                        "{:<5} {:<15.4f} {:<15.4f} {:<15.2f} {}".format(i + 1, point[0], point[1], point[2], point_name)
                    )

                print("=" * 50)
                print(f"Total: {len(waypoints)} points, Path length: {_calculate_path_length(waypoints):.2f}m")
                if same_track and track_num_drones > 1:
                    for mth_drone in range(track_num_drones):
                        # print the start and end points for all drones
                        header = f"Drone {mth_drone + 1} specific points"
                        print("\n" + "=" * 50)
                        print(f"{header:^50}")
                        print("=" * 50)
                        print("{:<10} {:<12} {:<12} {:<12}".format("Type", "X-coord(m)", "Y-coord(m)", "Z-coord(m)"))
                        print("-" * 50)

                        # print the start points
                        for j, start_point in enumerate(start_points[mth_drone]):
                            print(
                                "{:<10} {:<12.4f} {:<12.4f} {:<12.2f}".format(
                                    f"Start {j+1}", start_point[0], start_point[1], start_point[2]
                                )
                            )

                        # print the end points
                        for j, end_point in enumerate(end_points[mth_drone]):
                            print(
                                "{:<10} {:<12.4f} {:<12.4f} {:<12.2f}".format(
                                    f"End   {j+1}", end_point[0], end_point[1], end_point[2]
                                )
                            )
                    break  # only print once for same track


def _read_track(
    current_dir: Union[str, os.PathLike], env: gym.Env, track_name: str, verbose: int = 0
) -> Tuple[np.ndarray, str, dict]:
    """Read the track information.

    Parameters
    ----------
    current_dir : str or os.PathLike
        The current directory where the track file is located.
    env : gym.Env
        The environment to be used for evaluation.
    track_name : Union[str, os.PathLike]
        The name of the track to be used for evaluation.
    verbose : int, optional
        The verbosity level, by default 0

    Returns
    -------
    np.ndarray
        The waypoints of the track.
    str
        The comment to be added to the saved results.
    dict
        The raw data of the track.

    """
    # get the track path
    track_path = os.path.join(
        current_dir,
        "gym_drones/assets/Tracks",
        track_name,
    )

    # set evaluation waypoints
    with open(track_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    # track settings
    comment = data["comment"]
    track_num_drones = data.get("num_drones", 1)
    same_track = data.get("same_track", True)
    repeat_lap = data.get("repeat_lap", 1)
    # track waypoints
    start_points = np.array(data["start_points"]).reshape((track_num_drones, -1, 3))
    end_points = np.array(data["end_points"]).reshape((track_num_drones, -1, 3))
    if same_track:
        waypoints = np.tile(np.array(data["waypoints"]), (track_num_drones, 1, 1))
    else:
        waypoints = np.array(data["waypoints"]).reshape((track_num_drones, -1, 3))
    num_waypoints_per_lap = waypoints.shape[1]

    # check if the track is valid
    if track_num_drones != env.NUM_DRONES:
        ValueError("num_drones in track file is different from the setting")
    if ~same_track and repeat_lap > 1:
        ValueError("same_track is required for repeat_lap > 1")

    # generate the waypoints
    if repeat_lap == 1:
        main_segments = waypoints
    elif repeat_lap > 1:
        main_segments = np.tile(waypoints, (1, repeat_lap, 1))
    WPs = np.concatenate([start_points, main_segments, end_points], axis=1)

    # output the track information
    _output_track_info(
        track_name=comment,
        track_num_drones=track_num_drones,
        same_track=same_track,
        repeat_lap=repeat_lap,
        WPs=WPs,
        start_points=start_points,
        end_points=end_points,
        verbose=verbose,
    )
    moving_gate = data.get("moving_gate", None)
    if moving_gate is not None:
        gate_planner = GateMotionPlanner(
            num_waypoints_per_lap=num_waypoints_per_lap, num_laps=repeat_lap, moving_gate_config=moving_gate
        )
    else:
        gate_planner = None
        # print(gate_planner.motion_plan)
    return WPs.reshape(-1, 3) if track_num_drones == 1 else WPs, comment, data, gate_planner


def _add_track_noise(
    WPs: np.ndarray,
    track_num_drones: int,
    noise_sigma: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """add noise to the track waypoints.

    Parameters
    ----------
    WPs : np.ndarray
        The waypoints of the track.
    track_num_drones : int
        The number of drones in the track.
    noise_sigma : float, optional
        The standard deviation of the noise, by default 0.2

    Returns
    -------
    np.ndarray
        The waypoints of the track with noise.
    np.ndarray
        The noise matrix added to the waypoints.

    """
    WPs = WPs.reshape(track_num_drones, -1, 3)
    waypoints = WPs[0, 1:-1, :]  # remove start and end points
    # create noise
    noise = np.random.normal(0, noise_sigma, waypoints.shape)
    noise_matrix = np.tile(noise, (track_num_drones, 1, 1))
    # add noise to the waypoints
    WPs[:, 1:-1, :] += noise_matrix
    return WPs.reshape(-1, 3) if track_num_drones == 1 else WPs, noise_matrix


def _eval_sim_loop(
    predict_fn: Callable[[np.ndarray], Union[np.ndarray, tuple]],
    env: gym.Env,
    config_dict: dict,
    current_dir: Union[str, os.PathLike],
    output_folder: Union[str, os.PathLike],
    track_name: Optional[str] = None,
    track_sigma: float = 0.2,
) -> Tuple[Logger, EvalBaseCallback, str, dict, np.ndarray]:
    """Evaluate the model in a loop.

    Parameters
    ----------
    predict_fn : Callable[[np.ndarray], Union[np.ndarray, tuple]]
        The prediction function for the model.
    env : gym.Env
        The environment to be used for evaluation.
    config_dict : dict
        The configuration dictionary containing the evaluation parameters.
    current_dir : str or os.PathLike
        The current directory where the results will be saved.
    output_folder : str or os.PathLike
        The path where the evaluation results will be saved.
    callback : Optional[EvalBaseCallback], optional
        The callback to be used for evaluation, by default None
    track_name : Optional[str], optional
        The name of the track to be used for evaluation, by default None
        If None, the environment will be reset with random waypoints.

    Returns
    -------
    Logger
        The logger object containing the evaluation results.
    callback : EvalBaseCallback
        The callback object used for evaluation.
    str
        The comment to be added to the saved results.
    dict
        The raw data of the track.
    np.ndarray
        The noise matrix added to the waypoints.

    """
    # initialize the evaluation
    eval_loop = config_dict["logging"].get("eval_loop", 1)
    comment = None
    gate_planner = None
    if track_name is not None:
        waypoints_track, comment, track_raw_data, gate_planner = _read_track(
            current_dir=current_dir, env=env, track_name=track_name, verbose=config_dict["rl_hyperparams"]["verbose"]
        )
        run_track = True
        start_len = np.array(track_raw_data["start_points"]).reshape((env.NUM_DRONES, -1, 3)).shape[1]
        end_len = np.array(track_raw_data["end_points"]).reshape((env.NUM_DRONES, -1, 3)).shape[1]
    else:
        run_track = False

    # create the callbacks
    rew_callback = EvalRewardCallback(verbose=config_dict["rl_hyperparams"]["verbose"])
    callback = [rew_callback]
    if track_name is not None:
        time_callback = EvalTimeCallback(verbose=config_dict["rl_hyperparams"]["verbose"], data=track_raw_data)
        callback.append(time_callback)
    callback = _init_callback(env, callback)

    # start the evaluation loop
    callback.on_eval_start(locals(), globals())
    for j in range(eval_loop):
        # create the logger and reset the environment
        logger = Logger(
            logging_freq_hz=int(env.CTRL_FREQ),
            num_drones=env.NUM_DRONES,
            output_folder=output_folder,
        )
        if run_track:
            waypoints, noise_matrix = _add_track_noise(
                WPs=waypoints_track,
                track_num_drones=env.NUM_DRONES,
                noise_sigma=track_sigma,
            )
            if hasattr(env, "use_mappo") and env.use_mappo:
                obs, _, infos = env._setWaypoints_reset(waypoints=waypoints.copy())
            else:
                obs, infos = env._setWaypoints_reset(waypoints=waypoints.copy())
        else:
            if hasattr(env, "use_mappo") and env.use_mappo:
                obs, _, infos = env.reset(options={})
            else:
                obs, infos = env.reset(options={})
            spawn_point = env.pos.copy()
        callback.on_episode_start()

        # Sim loop
        step_counter = 0
        if gate_planner is not None:
            gate_t_list = [0.0]
            gate_waypoints_list = [env.waypoints.copy()]
        else:
            moving_gate_data = None
        while True:
            # predict
            action = predict_fn(obs)

            # step
            if hasattr(env, "use_mappo") and env.use_mappo:
                obs, _, reward, terminated, truncated, infos = env.step(action)
            else:
                obs, reward, terminated, truncated, infos = env.step(action)
            target = infos["target"]
            callback.update_child_locals(locals())
            callback.on_step()

            # record state
            for nth_drone in range(env.NUM_DRONES):
                log_state = np.hstack(
                    [
                        env.pos[nth_drone],
                        env.vel[nth_drone],
                        env.rpy[nth_drone],
                        env.rate[nth_drone],
                        env.quat[nth_drone],
                        target[nth_drone],
                        env.thrust[nth_drone],
                    ]
                )
                action_to_log = action[nth_drone] if action.ndim > 1 else action
                if env.DIM == SimulationDim.DIM_2:
                    action_to_log = np.hstack((action_to_log, np.zeros(2)))
                log_control = np.hstack([action_to_log, np.zeros(8)])

                logger.log(
                    drone=nth_drone,
                    timestamp=step_counter / env.CTRL_FREQ,
                    state=log_state,
                    control=log_control,
                )

            # update the counter
            step_counter += 1

            # check if the gate planner is used
            if gate_planner is not None:
                dynamic_waypoints = waypoints.copy()
                dynamic_waypoints[:, start_len:-end_len, :] = gate_planner.compute_positions(
                    t=step_counter / env.CTRL_FREQ, initial_positions=dynamic_waypoints[:, start_len:-end_len, :].copy()
                )
                gate_t_list.append(step_counter / env.CTRL_FREQ)
                gate_waypoints_list.append(dynamic_waypoints.copy())
                env._setWaypoints(waypoints=dynamic_waypoints.copy())

            # reset for track or random waypoints
            if run_track:
                if np.all(np.logical_or(env.finished, env.crashed)):
                    episode_status = "Finish the track"
                    break
            else:
                if terminated:
                    episode_status = "Terminated"
                    break

            # check if the episode is truncated
            if truncated:
                episode_status = "Truncated"
                break

            # check if the episode is out of time
            if step_counter >= (env.EPISODE_LEN_SEC * env.CTRL_FREQ) * 6:
                episode_status = "Out of time"
                break

        # end of the episode
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print("    " + "=" * 42)
            print(f"    [Episode {j + 1}] {episode_status}")
            print(f"    Total timesteps:       {step_counter} ({step_counter / env.CTRL_FREQ}s)")
        callback.on_episode_end()

    if config_dict["rl_hyperparams"]["verbose"] > 0:
        print("\n" + "=" * 50)
        print(f"{'EVALUATION COMPLETE':^50}")
        print(f"{'Episodes Completed: ' + str(eval_loop):^50}")
        print("=" * 50)

    callback.on_eval_end()
    if not run_track:
        max_num_waypoint = min(np.max(env.num_waypoints) + 2, len(env.waypoints))
        max_num_waypoint = 2 if max_num_waypoint < 1 else max_num_waypoint
        output_waypoints = np.tile(env.waypoints, (env.NUM_DRONES, 1, 1))
        pass_wps = output_waypoints.reshape((env.NUM_DRONES, -1, 3))[:, 0:max_num_waypoint]
        track_raw_data = {
            "comment": "Random_waypoints",
            "num_drones": env.NUM_DRONES,
            "same_track": False,
            "repeat_lap": 1,
            "start_points": spawn_point.tolist(),
            "end_points": pass_wps[:, -1].tolist(),
            "waypoints": pass_wps[:, 0:-1].tolist(),
        }
        noise_matrix = np.zeros((env.NUM_DRONES, pass_wps.shape[1] - 1, 3))

    # update the logger with the final information
    crashed_step = env.crashed_step.copy() if hasattr(env, "crashed_step") else np.zeros(env.NUM_DRONES, dtype=int)
    finished_step = env.finished_step.copy() if hasattr(env, "finished_step") else np.zeros(env.NUM_DRONES, dtype=int)
    update_info = {
        "crashed_step": crashed_step,
        "finished_step": finished_step,
    }
    logger.update_info(**update_info)

    if gate_planner is not None:
        # 1. Convert lists to NumPy arrays once.
        gate_t_arr = np.array(gate_t_list)
        gate_waypoints_arr = np.array(gate_waypoints_list)

        # 2. Determine the final length and trim the source arrays ONCE.
        if hasattr(logger, "end_step") and len(logger.end_step) > 0:
            max_step = np.max(logger.end_step)
            gate_t_arr = gate_t_arr[:max_step]
            gate_waypoints_arr = gate_waypoints_arr[:max_step]

        moving_gate_data = [
            {
                "t": gate_t_arr,  # Use the already trimmed time array
                "p_x": gate_waypoints_arr[:, nth_drone, :, 0],
                "p_y": gate_waypoints_arr[:, nth_drone, :, 1],
                "p_z": gate_waypoints_arr[:, nth_drone, :, 2],
            }
            for nth_drone in range(env.NUM_DRONES)
        ]

    return logger, callback, comment, track_raw_data, noise_matrix, moving_gate_data


def eval_model(
    model: Union[PPO, th.nn.Module],
    env: gym.Env,
    config_dict: dict,
    current_dir: Union[str, os.PathLike],
    save_results: bool = True,
    save_timestamps: bool = False,
    comment: Optional[str] = None,
    track_name: Optional[str] = None,
    track_sigma: float = 0.2,
) -> Tuple[Logger, Optional[dict], np.ndarray, os.PathLike, str]:
    """Evaluate the model.

    Parameters
    ----------
    model : PPO
        The PPO model to be evaluated.
    env : gym.Env
        The environment to be used for evaluation.
    config_dict : dict
        The configuration dictionary containing the evaluation parameters.
    current_dir : str or os.PathLike
        The current directory where the results will be saved.
    save_results : bool, optional
        Whether to save the results or not, by default True
    comment : str, optional
        A comment to be added to the saved results, by default "results"

    Returns
    -------
    Logger
        The logger object containing the evaluation results.
    Optional[dict]
        The raw data of the track.
    np.ndarray
        The noise matrix added to the waypoints.
    os.PathLike
        The path where the evaluation results will be saved.
    str
        The comment to be added to the saved results.

    """
    # unwrap the environment
    if hasattr(env, "unwrapped"):
        env = env.unwrapped

    # get the save path
    output_folder = _get_save_path(config_dict, current_dir)

    # Check the model type
    predict_fn = _get_predict_fn(model, config_dict)

    # get the logger
    logger, callback, results_comment, track_raw_data, noise_matrix, moving_gate_data = _eval_sim_loop(
        predict_fn=predict_fn,
        env=env,
        config_dict=config_dict,
        current_dir=current_dir,
        output_folder=output_folder,
        track_name=track_name,
        track_sigma=track_sigma,
    )

    # save the results
    if save_results:
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print("-" * 50)
            print("[SAVING RESULTS]")
            print(f"Saving results to {output_folder}")
        if comment is None:
            comment = "results" if results_comment is None else results_comment
        # save the flight data
        save_dir = logger.save_as_csv(comment=comment, save_timestamps=save_timestamps)
        # save the callback results
        callback.save_results(save_dir=save_dir, comment=comment)
        # save the config
        _save_config(
            config_dict=config_dict,
            eval_mode=True,
            eval_save_dir=save_dir,
        )
        # save the environment parameters
        env.saveYAMLParameters(
            save_path=os.path.join(save_dir, "configs"), verbose=config_dict["rl_hyperparams"]["verbose"]
        )
        # save the track data if available
        if track_raw_data is not None:
            save_track_path = os.path.join(save_dir, "configs/Tracks", f"{comment}.yaml")
            os.makedirs(os.path.dirname(save_track_path), exist_ok=True)
            if config_dict["rl_hyperparams"]["verbose"] > 0:
                print(f"Saving track data to {save_track_path}")
            with open(save_track_path, "w", encoding="utf-8") as file:
                yaml.dump(track_raw_data, file, default_flow_style=False)

    # Extract obstacle data if available
    obstacles = None
    obstacle_size = 0.5
    if hasattr(env, "obstacles") and hasattr(env, "NUM_OBSTACLES") and env.NUM_OBSTACLES > 0:
        obstacles = env.obstacles.copy()
        obstacle_size = env.OBSTACLE_SIZE
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print(f"[OBSTACLES] Found {len(obstacles)} obstacles for visualization")

    # close the environment
    env.close()

    return logger, track_raw_data, moving_gate_data, noise_matrix, save_dir, comment, obstacles, obstacle_size
