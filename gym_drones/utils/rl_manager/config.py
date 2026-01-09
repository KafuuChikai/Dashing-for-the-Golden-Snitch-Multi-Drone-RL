"""This module contains functions to process the configuration for the train and evaluate scripts."""

import argparse, os, yaml
from typing import Union, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from gym_drones.utils.utils import recursive_dict_update, recursive_enum_mapping, fix_none_values, get_latest_run_id


def _get_config(config_name: str, current_dir: Union[str, os.PathLike], subfolder: Union[str, os.PathLike]) -> dict:
    """Get the config dictionary from the YAML file.

    Parameters
    ----------
    config_name : str
        Name of the config file (without extension).
    current_dir : Union[str, os.PathLike]
        Current directory of the script.
    subfolder : Union[str, os.PathLike]
        Subfolder where the config file is located.

    Returns
    -------
    config_dict : dict
        Dictionary containing the configuration parameters.

    """
    with open(os.path.join(current_dir, "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict


def _read_all_config(
    args: argparse.Namespace, current_dir: Union[str, os.PathLike], envconfig: dict, algconfig: dict, runconfig: dict
) -> dict:
    """Read the default config file.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.
    envconfig : dict
        Environment configuration dictionary.
    algconfig : dict
        Algorithm configuration dictionary.
    runconfig : dict
        Runner configuration dictionary.

    Returns
    -------
    config_dict : dict
        Dictionary containing the configuration parameters.

    """
    if args.config is not None:
        # Load the config file
        with open(args.config, "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{} error: {}".format(args.config, exc)
        config_dict["logging"]["drone_model_path"] = os.path.join(current_dir, os.path.dirname(args.config))
    else:
        # Get the defaults from train_default.yaml
        with open(os.path.join(current_dir, "config", "train_default.yaml"), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "train_default.yaml error: {}".format(exc)

        # Update the config_dict with envconfig, algconfig, and runconfig
        config_dict = recursive_dict_update(config_dict, envconfig)
        config_dict = recursive_dict_update(config_dict, algconfig)
        config_dict = recursive_dict_update(config_dict, runconfig)
        config_dict["logging"]["drone_model_path"] = None

    # Update the seed in the config_dict
    seed = getattr(args, "seed", None)
    if seed is not None:
        config_dict["seed"] = seed

    # Update the pyrl config_dict with command line arguments
    exp_name = getattr(args, "exp_name", None)
    if exp_name is not None:
        config_dict["pyrl"]["exp_name"] = exp_name

    eval_name = getattr(args, "eval_name", None)
    if eval_name is not None:
        config_dict["pyrl"]["eval_name"] = eval_name

    num_envs = getattr(args, "num_envs", None)
    if num_envs is not None:
        config_dict["pyrl"]["n_envs"] = num_envs
        config_dict["pyrl"]["runner"] = "parallel" if num_envs > 1 else "episode"

    max_steps = getattr(args, "max_steps", None)
    if max_steps is not None:
        config_dict["pyrl"]["t_max"] = max_steps

    # Update the logging config_dict with command line arguments
    load_model_path = getattr(args, "load_model", None)
    if load_model_path is not None:
        config_dict["logging"]["load_model_path"] = load_model_path

    save_model_num = getattr(args, "save_num", None)
    if save_model_num is not None:
        config_dict["logging"]["save_model_num"] = save_model_num

    load_ckpt_path = getattr(args, "load_ckpt", None)
    if load_ckpt_path is not None:
        config_dict["logging"]["load_ckpt_path"] = load_ckpt_path

    load_step = getattr(args, "load_step", None)
    if load_step is not None:
        config_dict["logging"]["load_step"] = load_step

    save_eval_path = getattr(args, "save_eval", None)
    if save_eval_path is not None:
        config_dict["logging"]["save_eval_path"] = save_eval_path

    reset_num_timesteps = getattr(args, "no_reset_t", None)
    if reset_num_timesteps is not None:
        config_dict["logging"]["reset_num_timesteps"] = (
            reset_num_timesteps if not reset_num_timesteps else config_dict["logging"]["reset_num_timesteps"]
        )

    eval_overwrite = getattr(args, "no_ow", None)
    if eval_overwrite is not None:
        config_dict["logging"]["eval_overwrite"] = eval_overwrite

    eval_loop = getattr(args, "loop", None)
    if eval_loop is not None:
        config_dict["logging"]["eval_loop"] = eval_loop

    # Update the rl_hyperparams config_dict with command line arguments
    verbose = getattr(args, "verbose", None)
    if verbose is not None:
        config_dict["rl_hyperparams"]["verbose"] = verbose

    # Check if verbose is valid
    if config_dict["rl_hyperparams"]["verbose"] not in (0, 1, 2):
        raise ValueError("Invalid value for verbose. Allowed values are 0, 1 or 2.")

    # Update the env_kwargs config_dict with obstacle parameters from command line arguments
    num_obstacles = getattr(args, "num_obstacles", None)
    if num_obstacles is not None:
        if "env_kwargs" not in config_dict["env"]:
            config_dict["env"]["env_kwargs"] = {}
        config_dict["env"]["env_kwargs"]["num_obstacles"] = num_obstacles
        print(f"[CONFIG DEBUG] Setting num_obstacles={num_obstacles} in env_kwargs")

    obstacle_size = getattr(args, "obstacle_size", None)
    if obstacle_size is not None:
        if "env_kwargs" not in config_dict["env"]:
            config_dict["env"]["env_kwargs"] = {}
        config_dict["env"]["env_kwargs"]["obstacle_size"] = obstacle_size

    ray_length = getattr(args, "ray_length", None)
    if ray_length is not None:
        if "env_kwargs" not in config_dict["env"]:
            config_dict["env"]["env_kwargs"] = {}
        config_dict["env"]["env_kwargs"]["ray_length"] = ray_length

    num_rays = getattr(args, "num_rays", None)
    if num_rays is not None:
        if "env_kwargs" not in config_dict["env"]:
            config_dict["env"]["env_kwargs"] = {}
        config_dict["env"]["env_kwargs"]["num_rays"] = num_rays
        print(f"[CONFIG DEBUG] Setting num_rays={num_rays} in env_kwargs")

    obstacle_safe_dist = getattr(args, "obstacle_safe_dist", None)
    if obstacle_safe_dist is not None:
        if "env_kwargs" not in config_dict["env"]:
            config_dict["env"]["env_kwargs"] = {}
        config_dict["env"]["env_kwargs"]["obstacle_safe_dist"] = obstacle_safe_dist

    obstacle_reward_weight = getattr(args, "obstacle_reward_weight", None)
    if obstacle_reward_weight is not None:
        if "env_kwargs" not in config_dict["env"]:
            config_dict["env"]["env_kwargs"] = {}
        config_dict["env"]["env_kwargs"]["obstacle_reward_weight"] = obstacle_reward_weight

    # Set default experiment name if not provided
    if config_dict["pyrl"]["exp_name"] is None:
        config_dict["pyrl"]["exp_name"] = args.env
    return config_dict


def _save_config(
    config_dict: dict,
    current_dir: Optional[Union[str, os.PathLike]] = None,
    eval_mode: bool = False,
    eval_save_dir: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Save the config dictionary to a YAML file.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.
    current_dir : Optional[Union[str, os.PathLike]]
        Current directory of the script. Must be provided if eval_mode is False.
        Default is None.
    eval_mode : bool, optional
        If True, save the config for evaluation mode. Default is False.
    eval_save_dir : Optional[Union[str, os.PathLike]], optional
        Directory to save the evaluation config. Must be provided if eval_mode is True.
        Default is None.

    """
    # save the config_dict to a YAML file
    if config_dict["logging"]["save_config"]:
        model_name = config_dict["logging"]["save_model_name"]
        model_id = config_dict["logging"]["run_id"]
        if eval_mode:
            if eval_save_dir is None:
                raise ValueError("eval_save_dir must be provided when eval_mode is True.")
            config_save_path = os.path.join(
                eval_save_dir,
                "configs",
                "config.yaml",
            )
            if config_dict["rl_hyperparams"]["verbose"] > 0:
                print("Evaluation config saved to:", config_save_path)
        else:
            if current_dir is None:
                raise ValueError("current_dir must be provided when training.")
            config_save_path = os.path.join(
                current_dir,
                config_dict["logging"]["save_config_dirname"],
                config_dict["pyrl"]["exp_name"],
                f"{model_name}_{model_id + 1}",
                "configs",
                "config.yaml",
            )
            if config_dict["rl_hyperparams"]["verbose"] > 0:
                print("Train Config saved to:", config_save_path)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        with open(config_save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def process_run_id(config_dict: dict, current_dir: Union[str, os.PathLike]) -> None:
    """Process the run_id in the configuration.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.

    """
    log_save_path = os.path.join(
        current_dir,
        config_dict["logging"]["tb_dirname"],
        config_dict["pyrl"]["exp_name"],
    )
    log_save_name = config_dict["logging"]["save_model_name"]

    latest_run_id = get_latest_run_id(log_save_path, log_save_name)
    # If run_id is set, protect files from being overwritten
    if config_dict["logging"].get("run_id") is None or config_dict["logging"]["reset_num_timesteps"]:
        # Adjust run_id if reset_num_timesteps is False
        if config_dict["logging"].get("run_id") is None and not config_dict["logging"]["reset_num_timesteps"]:
            latest_run_id -= 1
        config_dict["logging"]["run_id"] = latest_run_id


def process_eval_config(config_dict: dict) -> None:
    """Process the evaluation configuration.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.

    """
    # process pyrl config
    config_dict["pyrl"]["runner"] = "episode"
    config_dict["pyrl"]["n_envs"] = 1
    # process logging config
    config_dict["logging"]["reset_num_timesteps"] = False
    file_ext = os.path.splitext(config_dict["logging"]["load_model_path"])[1].lower()
    if file_ext == ".pt":
        file_name = os.path.splitext(os.path.basename(config_dict["logging"]["load_model_path"]))[0]
        config_dict["logging"]["save_model_name"] = (
            file_name + "_pt" if config_dict["logging"]["save_model_name"] == file_name else file_name
        )
    # process env config
    config_dict["env"]["env_kwargs"]["eval_mode"] = True
    config_dict["env"]["env_kwargs"]["domain_rand"] = False


def process_config(
    args: argparse.Namespace,
    current_dir: Union[str, os.PathLike],
    envkey: dict,
    algkey: dict,
    runkey: dict,
    enum_mapping: dict,
    eval_mode: bool = False,
) -> dict:
    """load the config file and process it.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.
    envkey : dict
        Dictionary containing environment keys.
    algkey : dict
        Dictionary containing algorithm keys.
    runkey : dict
        Dictionary containing runner keys.
    enum_mapping : dict
        Dictionary containing enum mappings.
    eval_mode : bool, optional
        If True, process the configuration for evaluation mode. Default is False.

    Returns
    -------
    config_dict : dict
        Dictionary containing the configuration parameters.

    """
    # read the config file
    envconfig = _get_config(envkey[args.env], current_dir, "envs")  # read env config
    algconfig = _get_config(algkey[args.env], current_dir, f"algs/{envkey[args.env]}")  # read rl hyperparameters
    runconfig = _get_config(runkey[args.env], current_dir, "runners")  # read runner config

    # update the config_dict with envconfig, algconfig, and runconfig
    config_dict = _read_all_config(args, current_dir, envconfig, algconfig, runconfig)
    if eval_mode:
        process_eval_config(config_dict)

    # process the run_id
    process_run_id(config_dict, current_dir)

    # output the experiment info
    if config_dict["rl_hyperparams"]["verbose"] > 0:
        print("-" * 50)
        print("[EXPERIMENT INFO]")
        exp_name = config_dict["pyrl"]["exp_name"]
        print(f"Experiment name: {exp_name}")

    # Save the config_dict to a YAML file
    if not eval_mode:
        _save_config(config_dict, current_dir)

    # process the config_dict
    fix_none_values(config_dict)
    config_dict = recursive_enum_mapping(config_dict, enum_mapping)
    return config_dict


def process_vis_config(
    args: argparse.Namespace,
    current_dir: Union[str, os.PathLike],
    file_name: str,
) -> dict:
    """load the config file and process it.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.
    enum_mapping : dict
        Dictionary containing enum mappings.

    Returns
    -------
    config_dict : dict
        Dictionary containing the configuration parameters.

    """
    with open(os.path.join(current_dir, "config", "vis", f"{file_name}.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "{} error: {}".format(file_name, exc)
    config_dict["common_kwargs"] = config_dict.get("common_kwargs", {})

    radius = getattr(args, "radius", None)
    if radius is not None:
        config_dict["shape_kwargs"]["radius"] = radius

    margin = getattr(args, "margin", None)
    if margin is not None:
        config_dict["shape_kwargs"]["margin"] = margin

    config_dict["track_kwargs"] = {**config_dict["shape_kwargs"], **config_dict["gate_kwargs"]}

    _recursive_process_cmap(config_dict)

    fix_none_values(config_dict)
    return config_dict


def _recursive_process_cmap(data: Union[dict, list]) -> None:
    """
    Recursively traverses a dictionary or list to find and process 'cmap' keys.

    It converts string representations of colormaps or solid colors into
    actual Matplotlib Colormap objects, modifying the data structure in-place.

    Parameters
    ----------
    data : Union[dict, list]
        The dictionary or list to traverse.
    """
    if not isinstance(data, dict):
        return

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "cmap":
                if isinstance(value, str):
                    # Handle single string value
                    try:
                        data[key] = plt.get_cmap(value)
                    except ValueError:
                        data[key] = ListedColormap([value])
                elif isinstance(value, list):
                    # Handle list of strings
                    processed_list = []
                    for item in value:
                        if isinstance(item, str):
                            try:
                                processed_list.append(plt.get_cmap(item))
                            except ValueError:
                                processed_list.append(ListedColormap([item]))
                        else:
                            # Keep non-string items as they are
                            processed_list.append(item)
                    data[key] = processed_list
            elif isinstance(value, dict):
                # If the value is another dictionary, recurse into it
                _recursive_process_cmap(value)
            elif isinstance(value, list):
                # If the value is a list, recurse into its dictionary items
                for item in value:
                    if isinstance(item, dict):
                        _recursive_process_cmap(item)
