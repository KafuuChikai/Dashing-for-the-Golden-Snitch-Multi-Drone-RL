import os
import numpy as np

from numpy.core.records import fromarrays
from typing import List, Optional, Union
from gym_drones.utils.Logger import Logger
from race_utils.RaceGenerator.RaceTrack import RaceTrack
from race_utils.RaceVisualizer.RacePlotter import RacePlotter, BasePlotterList
from race_utils.RaceGenerator.GenerationTools import create_state, create_gate


def _data_to_racetrack(data: dict, shape_kwargs: dict, noise_matrix: np.ndarray) -> List[RaceTrack]:
    """
    Convert the data to a racetrack list.

    Parameters
    ----------
    data : np.ndarray
        The data to convert.
    shape_kwargs : dict
        The shape kwargs to use.
    noise_matrix : np.ndarray
        The noise matrix to use.

    Returns
    -------
    List[RaceTrack]
        The racetrack list for all drones.

    """
    # track settings
    comment = data["comment"]
    track_num_drones = data.get("num_drones", 1)
    same_track = data.get("same_track", True)
    repeat_lap = data.get("repeat_lap", 1)
    # track waypoints
    start_points = np.array(data["start_points"], dtype=float).reshape((track_num_drones, -1, 3))
    end_points = np.array(data["end_points"], dtype=float).reshape((track_num_drones, -1, 3))
    if same_track:
        waypoints = np.tile(np.array(data["waypoints"], dtype=float), (track_num_drones, 1, 1))
    else:
        waypoints = np.array(data["waypoints"], dtype=float).reshape((track_num_drones, -1, 3))

    # check if the track is valid
    if ~same_track and repeat_lap > 1:
        ValueError("same_track is required for repeat_lap > 1")

    # generate the waypoints
    if repeat_lap == 1:
        main_segments = waypoints
    elif repeat_lap > 1:
        main_segments = np.tile(waypoints, (1, repeat_lap, 1))
    if start_points.shape[1] > 1:
        main_segments = np.concatenate((start_points[:, 1:], main_segments), axis=1)
    if end_points.shape[1] > 1:
        main_segments = np.concatenate((main_segments, end_points[:, :-1]), axis=1)
    main_segments += noise_matrix

    racetrack_list = []
    main_seg_len = main_segments.shape[1]
    for nth_drone in range(track_num_drones):
        # Define initial and end states
        init_state = create_state({"pos": start_points[nth_drone]})
        end_state = create_state({"pos": end_points[nth_drone]})

        # Create the RaceTrack object
        race_track = RaceTrack(init_state=init_state, end_state=end_state, race_name=f"{comment}_drone{nth_drone + 1}")

        for j in range(main_seg_len):
            # Create the gate
            gate = create_gate(
                gate_type="SingleBall",
                position=main_segments[nth_drone, j],
                stationary=True,
                shape_kwargs=shape_kwargs,
                name=f"{comment}_Gate_{j+1}",
            )
            race_track.add_gate(gate, gate.name)
        racetrack_list.append(race_track)
    return racetrack_list


def _logger_to_traj_data(logger: Logger) -> List[np.ndarray]:
    """
    Convert the logger data to a list.

    Parameters
    ----------
    logger : Logger
        The logger to convert.

    Returns
    -------
    List[np.ndarray]
        The data list for all drones.

    """
    states = logger.states.copy()

    # define the data type
    dtype = [
        ("t", "f8"),  # time
        ("p_x", "f8"),
        ("p_y", "f8"),
        ("p_z", "f8"),  # position
        ("q_x", "f8"),
        ("q_y", "f8"),
        ("q_z", "f8"),
        ("q_w", "f8"),  # quaternion
        ("v_x", "f8"),
        ("v_y", "f8"),
        ("v_z", "f8"),  # velocity
    ]

    data_list = []
    t = np.arange(0, logger.timestamps.shape[1]) / logger.LOGGING_FREQ_HZ
    for i in range(logger.NUM_DRONES):
        # concatenate the data
        data = np.concatenate(
            (
                t.reshape((1, -1)),
                states[i][0:3, :],
                states[i][12:16, :],
                states[i][3:6, :],
            ),
            axis=0,
        ).T
        arrays = [data[:, j] for j in range(data.shape[1])]
        data = fromarrays(arrays, dtype=dtype)

        if hasattr(logger, "end_step"):
            data = data[: logger.end_step[i]]

        # append the data to the list
        data_list.append(data)
    if hasattr(logger, "end_step"):
        end_time = np.max(logger.end_step.copy() / logger.LOGGING_FREQ_HZ)
    else:
        end_time = None
    crash_effect = logger.crashed_step > 0
    return data_list, end_time, crash_effect


def create_raceplotter(
    logger: Logger,
    track_data: dict,
    shape_kwargs: dict,
    noise_matrix: np.ndarray,
    moving_gate_data: Optional[np.ndarray] = None,
    obstacles: Optional[np.ndarray] = None,
    obstacle_size: float = 0.5,
) -> BasePlotterList:
    """
    Create a RacePlotter list.

    Parameters
    ----------
    logger : Logger
        The logger to convert.
    track_data : dict
        The track data to convert.
    shape_kwargs : dict
        The shape kwargs to use.
    noise_matrix : np.ndarray
        The noise matrix to use.

    Returns
    -------
    List[RacePlotter]
        The RacePlotter list for all drones.
    """
    # convert the logger data to a dictionary
    data_list, end_time, crash_effect = _logger_to_traj_data(logger)

    # convert the data to a racetrack
    racetrack_list = _data_to_racetrack(track_data, shape_kwargs, noise_matrix)

    # create the RacePlotter list
    raceplotter_list = []
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    if moving_gate_data is None:
        moving_gate_data = [None] * logger.NUM_DRONES
    for i in range(logger.NUM_DRONES):
        crash_kwargs = {"color": color_cycle[i % len(color_cycle)]}
        raceplotter = RacePlotter(
            traj_file=data_list[i],
            track_file=racetrack_list[i],
            end_time=end_time,
            crash_effect=crash_effect[i],
            crash_kwargs=crash_kwargs,
            moving_gate_data=moving_gate_data[i],
        )
        # Store obstacle data for visualization
        if obstacles is not None:
            raceplotter.obstacles = obstacles
            raceplotter.obstacle_size = obstacle_size
        raceplotter_list.append(raceplotter)
    plotter = BasePlotterList(plotters=raceplotter_list)
    return plotter


def load_plotter_track(
    current_dir: Union[str, os.PathLike],
    track_file: Union[str, os.PathLike, RaceTrack],
    plotter: Optional[RacePlotter] = None,
    index: Optional[list] = None,
    plot_track_once: bool = False,
) -> Union[RacePlotter, BasePlotterList]:
    """
    Load the track file.

    Parameters
    ----------
    track_file : str
        The track file to load.
    index : list, optional
        The index of the track file to load, by default None
        If None, load all the track files.

    Returns
    -------
    Union[RacePlotter, BasePlotterList]
        The loaded plotters.

    """
    track_file = os.path.join(current_dir, "gym_drones/assets/Tracks/RaceUtils", f"{track_file}.yaml")
    if isinstance(plotter, RacePlotter):
        plotter.load_track(track_file=track_file, index=index)
    else:
        plotter.load_track(track_file=track_file, index=index, plot_track_once=plot_track_once)
    return plotter
