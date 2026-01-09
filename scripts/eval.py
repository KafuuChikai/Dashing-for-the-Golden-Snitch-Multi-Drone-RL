import os, argparse
import torch as th

from gym_drones.utils.enums import ObservationType, SimulationDim, ActionType, DroneModel
from gym_drones.utils.rl_manager.config import process_config, process_vis_config
from gym_drones.utils.rl_manager.runner import pre_runner, build_env, load_model
from gym_drones.utils.rl_manager.eval_utils import eval_model
from gym_drones.utils.vis_utils import create_raceplotter, load_plotter_track

#### Set Constants #######################################
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

envkey = {
    "hover_race": "hover_race",
    "race_multi_2": "race_multi_2",
    "race_multi_3": "race_multi_3",
    "race_multi_5": "race_multi_5",
    "kin_2d": "kin_2d",
    "kin_3d": "kin_3d",
    "kin_rel_2d": "kin_rel_2d",
    "kin_rel_3d": "kin_rel_3d",
    "pos_rel": "pos_rel",
    "rot_rel": "rot_rel",
}
algkey = {
    "hover_race": "ppo",
    "race_multi_2": "ippo",
    "race_multi_3": "ippo",
    "race_multi_5": "ippo",
    "kin_2d": "ppo",
    "kin_3d": "ppo",
    "kin_rel_2d": "ppo",
    "kin_rel_3d": "ppo",
    "pos_rel": "ppo",
    "rot_rel": "ppo",
}
runkey = {
    "hover_race": "hover_race",
    "race_multi_2": "race_multi_2",
    "race_multi_3": "race_multi_3",
    "race_multi_5": "race_multi_5",
    "kin_2d": "kin_2d",
    "kin_3d": "kin_3d",
    "kin_rel_2d": "kin_rel_2d",
    "kin_rel_3d": "kin_rel_3d",
    "pos_rel": "pos_rel",
    "rot_rel": "rot_rel",
}

enum_mapping = {
    "drone_model": DroneModel,
    "obs": ObservationType,
    "act": ActionType,
    "dim": SimulationDim,
    "activation_fn": th.nn,
    "output_activation_fn": th.nn,
}


def run():
    #### Get the Training Parameters ########################
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Add arguments
    parser.add_argument("-e", "--env", type=str, required=True, help="Specify the environment.")
    parser.add_argument("-n", "--exp_name", type=str, required=False, help="Specify the experiment name.")
    parser.add_argument("-r", "--eval_name", type=str, required=False, help="Specify the evaluation name.")
    parser.add_argument("-m", "--load_model", type=str, required=False, help="Specify the model path.")
    parser.add_argument("-c", "--load_ckpt", type=str, required=False, help="Specify the checkpoint path.")
    parser.add_argument("-s", "--load_step", type=int, required=False, help="Specify the load step.")
    parser.add_argument("-S", "--save_eval", type=str, required=False, help="Specify the results save path.")
    parser.add_argument("-f", "--config", type=str, required=False, help="Specify the config file.")
    parser.add_argument("-v", "--verbose", type=int, required=False, help="Specify the verbosity level.")
    parser.add_argument("-k", "--no_ow", action="store_false", help="Do not overwrite the results.")
    parser.add_argument("-l", "--loop", type=int, required=False, help="Specify the number of evaluation loops.")
    parser.add_argument("--seed", type=int, required=False, help="Specify the random seed.")
    parser.add_argument("--comment", type=str, required=False, help="Specify the comment for the results.")
    parser.add_argument("--track", type=str, required=False, help="Specify the track name.")
    parser.add_argument("--track_sigma", type=float, default=0.0, help="Specify the track sigma.")
    parser.add_argument("--save_timestamps", action="store_true", help="Save timestamps.")
    parser.add_argument("--radius", type=float, required=False, help="Specify the radius for the waypoints.")
    parser.add_argument("--margin", type=float, required=False, help="Specify the margin for the waypoints.")
    parser.add_argument("--vis_config", type=str, default="waypoints", help="Specify the visualization config file.")
    parser.add_argument("--save_video", default=False, action="store_true", help="Save the video of the evaluation.")
    # Obstacle parameters
    parser.add_argument("--num_obstacles", type=int, required=False, help="Number of obstacles in the environment.")
    parser.add_argument(
        "--obstacle_size", type=float, required=False, help="Size (diameter) of each obstacle in meters."
    )
    parser.add_argument("--ray_length", type=float, required=False, help="Maximum ray detection distance in meters.")
    parser.add_argument("--num_rays", type=int, required=False, help="Number of rays cast by each drone.")
    parser.add_argument(
        "--obstacle_safe_dist", type=float, required=False, help="Safety margin distance for obstacles in meters."
    )
    parser.add_argument(
        "--obstacle_reward_weight", type=float, required=False, help="Weight for obstacle avoidance reward."
    )

    # Use the arguments
    args = parser.parse_args()
    if args.load_model is None and args.load_ckpt is None:
        raise ValueError("Please specify the model path using --load_model or --load_ckpt.")
    if args.load_model is not None and args.load_ckpt is not None:
        Warning("Both model and checkpoint paths are specified. The checkpoint path will be used.")
    if args.config is None:
        print(
            "No config file specified. Using default config, which may not be suitable for your environment or model."
        )
    if "race_multi" in args.env and args.vis_config == "waypoints":
        args.vis_config = "waypoints_multi"
    if "race_multi_5" in args.env and args.vis_config == "waypoints_multi":
        args.vis_config = "waypoints_multi_5"

    # Read the config file
    config_dict = process_config(
        args=args,
        current_dir=current_dir,
        envkey=envkey,
        algkey=algkey,
        runkey=runkey,
        enum_mapping=enum_mapping,
        eval_mode=True,
    )

    #### Start the Training ###############################
    # Prepare the runner
    pre_runner(config_dict=config_dict, eval_mode=True)

    # Build the env
    env = build_env(config_dict=config_dict, current_dir=current_dir, eval_mode=True)

    # Load the model
    model = load_model(config_dict=config_dict, current_dir=current_dir, eval_mode=True)

    # evaluate the model and get the logger
    logger, track_raw_data, moving_gate_data, noise_matrix, save_dir, comment, obstacles, obstacle_size = eval_model(
        model=model,
        env=env,
        config_dict=config_dict,
        current_dir=current_dir,
        save_results=True,
        save_timestamps=args.save_timestamps,
        comment=args.comment,
        track_name=args.track,
        track_sigma=args.track_sigma,
    )
    # update visualization config
    track_vis_config = track_raw_data.get("vis_config", None)
    if track_vis_config is not None:
        args.vis_config = track_vis_config

    # create the raceplotter
    vis_config_dict = process_vis_config(args=args, current_dir=current_dir, file_name=args.vis_config)
    raceplotter = create_raceplotter(
        logger=logger,
        track_data=track_raw_data,
        shape_kwargs=vis_config_dict["shape_kwargs"],
        noise_matrix=noise_matrix,
        moving_gate_data=moving_gate_data,
        obstacles=obstacles,
        obstacle_size=obstacle_size,
    )

    vis_config_dict["track_file"] = vis_config_dict.get("track_file", None)
    if vis_config_dict["track_file"] is not None:
        raceplotter = load_plotter_track(
            current_dir=current_dir,
            track_file=vis_config_dict["track_file"],
            plotter=raceplotter,
            plot_track_once=vis_config_dict.get("plot_track_once", False),
        )
    ############# 2D Visualization ##########################
    raceplotter.plot(
        save_fig=True,
        save_path=save_dir,
        fig_name=comment,
        fig_title=comment,
        **vis_config_dict["2d_kwargs"],
        **vis_config_dict["shape_kwargs"],
    )
    ############# 3D Visualization ##########################
    raceplotter.plot3d(
        save_fig=True,
        save_path=save_dir,
        fig_name=comment,
        fig_title=comment,
        **vis_config_dict["3d_kwargs"],
        **vis_config_dict["shape_kwargs"],
        **vis_config_dict["gate_kwargs"],
    )
    ############# Animation Visualization ###################
    ani_list = raceplotter.create_animation(
        drone_kwargs=vis_config_dict["drone_kargs"],
        track_kwargs=vis_config_dict["track_kwargs"],
        save_path=save_dir if args.save_video else None,
        video_name=comment,
        **vis_config_dict["ani_kwargs"],
    )
    logger.plot()


if __name__ == "__main__":
    run()
