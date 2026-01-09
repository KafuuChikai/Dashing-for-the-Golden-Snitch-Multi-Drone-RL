import os, argparse
import torch as th

from gym_drones.utils.enums import ObservationType, SimulationDim, ActionType, DroneModel
from gym_drones.utils.rl_manager.config import process_config
from gym_drones.utils.rl_manager.runner import pre_runner, build_env, load_model, train_model

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
    parser.add_argument("-B", "--num_envs", type=int, required=False, help="Specify the number of environments.")
    parser.add_argument("-M", "--max_steps", type=int, required=False, help="Specify the max steps.")
    parser.add_argument("-S", "--save_num", type=int, required=False, help="Specify the save frequency.")
    parser.add_argument("-m", "--load_model", type=str, required=False, help="Specify the model path.")
    parser.add_argument("-c", "--load_ckpt", type=str, required=False, help="Specify the checkpoint path.")
    parser.add_argument("-s", "--load_step", type=int, required=False, help="Specify the load step.")
    parser.add_argument("-f", "--config", type=str, required=False, help="Specify the config file.")
    parser.add_argument("-v", "--verbose", type=int, required=False, help="Specify the verbosity level.")
    parser.add_argument("--seed", type=int, required=False, help="Specify the random seed.")
    parser.add_argument(
        "--no_reset_t",
        action="store_false",
        help="Do not reset the number of timesteps (default: use the value from config.yaml).",
    )
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
    if args.load_model is not None and args.load_ckpt is not None:
        Warning("Both model and checkpoint paths are specified. The checkpoint path will be used.")

    # Read the config file
    config_dict = process_config(args, current_dir, envkey, algkey, runkey, enum_mapping)

    #### Start the Training ###############################
    # Prepare the runner
    device = pre_runner(config_dict)

    # Build the env
    env = build_env(config_dict, current_dir)

    # Load the model
    model = load_model(config_dict=config_dict, current_dir=current_dir, env=env, device=device)

    # train the model
    train_model(model=model, config_dict=config_dict, current_dir=current_dir)


if __name__ == "__main__":
    run()
