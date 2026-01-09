"""This module contains functions to prepare the runner, build the environment,
load the model, and train the model using Stable Baselines3.
"""

import os
import torch as th
import gymnasium as gym
from typing import Union, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, SubprocShareVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_drones.utils.utils import set_seed
from gym_drones.utils.RewardCheckCallback import RewardCheckCallback


def pre_runner(config_dict: dict, eval_mode: bool = False) -> th.device:
    """Prepare the runner before training.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.
    eval_mode : bool, optional
        If True, the runner is set to evaluation mode. Default is False.
        In evaluation mode, the number of environments is set to 1 and
        the runner is set to episode mode.

    Returns
    -------
    device : torch.device
        The device to be used for training (CPU or GPU).

    """
    # set the seed
    if config_dict["seed"] is not None:
        set_seed(config_dict["seed"])

    # if eval mode, set the runner to episode and n_envs to 1
    if eval_mode:
        config_dict["pyrl"]["runner"] == "episode"
        config_dict["pyrl"]["n_envs"] = 1
    else:
        if config_dict["pyrl"]["runner"] == "episode":
            if config_dict["pyrl"]["n_envs"] > 1:
                print("Warning: n_envs > 1, but runner is episode. Set n_envs = 1.")
            config_dict["pyrl"]["n_envs"] = 1

    # configure the model save interval
    num_cpu = config_dict["pyrl"]["n_envs"]
    total_timesteps = config_dict["pyrl"]["t_max"]
    save_num = config_dict["logging"]["save_model_num"]
    model_save_interval = int(total_timesteps / save_num / num_cpu / 100) * 100
    config_dict["logging"]["model_save_interval"] = model_save_interval

    # set the device
    device = th.device("cuda" if config_dict["pyrl"]["use_cuda"] and th.cuda.is_available() else "cpu")
    if device.type == "cpu" and config_dict["pyrl"]["use_cuda"]:
        print("Warning: CUDA is not available, using CPU instead.")
    if config_dict["rl_hyperparams"]["verbose"] > 0:
        print("Seed:", config_dict["seed"])
        if eval_mode:
            print("Runner Mode: Evaluation,", config_dict["pyrl"]["runner"])
        else:
            print("Runner Mode: Train,", config_dict["pyrl"]["runner"])
            print("Model Save Interval:", model_save_interval * num_cpu)
    return device


def build_env(
    config_dict: dict, current_dir: Union[str, os.PathLike], eval_mode: bool = False
) -> Union[gym.Env, SubprocVecEnv, SubprocShareVecEnv]:
    """Build the environment based on the configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.
    eval_mode : bool, optional
        If True, the environment is set to evaluation mode. Default is False.
        In evaluation mode, the number of environments is set to 1 and
        the runner is set to episode mode.

    Returns
    -------
    env : Union[gym.Env, SubprocVecEnv, SubprocShareVecEnv]
        The created environment.

    """
    # set the environment parameters
    env_params = {
        **config_dict["env"],
        "n_envs": config_dict["pyrl"]["n_envs"],
        "seed": config_dict["seed"],
    }
    if config_dict["logging"]["drone_model_path"] is not None:
        env_params["env_kwargs"]["drone_model_path"] = config_dict["logging"]["drone_model_path"]
    env_params["env_kwargs"]["eval_mode"] = eval_mode

    # create the environment
    if config_dict["pyrl"]["runner"] == "parallel":
        if config_dict.get("marl_hyperparams", None) is not None:
            if config_dict["marl_hyperparams"].get("num_agent", 1) > 1:
                # if MARL is used, set the share observation space
                env = make_vec_env(
                    **env_params,
                    vec_env_cls=SubprocShareVecEnv,
                )
            else:
                env = make_vec_env(
                    **env_params,
                    vec_env_cls=SubprocVecEnv,
                )
        else:
            env = make_vec_env(
                **env_params,
                vec_env_cls=SubprocVecEnv,
            )
    elif config_dict["pyrl"]["runner"] == "episode":
        if config_dict.get("marl_hyperparams", None) is not None and not eval_mode:
            raise ValueError("MARL is not supported in episode mode. Please set the runner to 'parallel'.")
        env_params = {
            "id": env_params["env_id"],
            **env_params["env_kwargs"],
        }
        env = gym.make(
            **env_params,
        )
    else:
        raise ValueError("Invalid runner type. Choose 'parallel' or 'episode'.")

    # output the environment details
    if config_dict["rl_hyperparams"]["verbose"] > 0:
        print("-" * 50)
        print("[ENVIRONMENT DETAILS]")
        print("Environment_id:", config_dict["env"]["env_id"])
        print("Observation Type:", config_dict["env"]["env_kwargs"]["obs"])
        print("Dimension:", config_dict["env"]["env_kwargs"]["dim"])
        print("Action Space:", env.action_space)
        print("Observation Space:", env.observation_space)
        if "num_obstacles" in config_dict["env"]["env_kwargs"]:
            print(
                f"[OBSTACLE CONFIG] num_obstacles={config_dict['env']['env_kwargs']['num_obstacles']}, "
                f"num_rays={config_dict['env']['env_kwargs'].get('num_rays', 'not set')}, "
                f"obstacle_size={config_dict['env']['env_kwargs'].get('obstacle_size', 'not set')}, "
                f"ray_length={config_dict['env']['env_kwargs'].get('ray_length', 'not set')}"
            )
        if eval_mode:
            print("Evaluation sensor noise:", config_dict["env"]["env_kwargs"]["sensor_sigma"])
        else:
            print("Parallel Environments:", config_dict["pyrl"]["n_envs"])
        if config_dict["pyrl"]["runner"] == "parallel":
            env.get_attr(attr_name="unwrapped", indices=0)[0].getDroneParameters()
        else:
            env.unwrapped.getDroneParameters()

    # save the environment parameters
    if config_dict["logging"]["save_config"] and not eval_mode:
        # get the save path
        model_name = config_dict["logging"]["save_model_name"]
        model_id = config_dict["logging"]["run_id"]
        config_save_path = os.path.join(
            current_dir,
            config_dict["logging"]["save_config_dirname"],
            config_dict["pyrl"]["exp_name"],
            f"{model_name}_{model_id + 1}",
            "configs",
        )
        # save the config file
        if config_dict["pyrl"]["runner"] == "parallel":
            env.get_attr(attr_name="unwrapped", indices=0)[0].saveYAMLParameters(
                save_path=config_save_path, verbose=config_dict["rl_hyperparams"]["verbose"]
            )
        else:
            env.unwrapped.saveYAMLParameters(
                save_path=config_save_path, verbose=config_dict["rl_hyperparams"]["verbose"]
            )
    return env


def _find_checkpoint(config_dict: dict, current_dir: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    """Find the checkpoint file based on the configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.

    Returns
    -------
    ckpt_path : Union[str, os.PathLike]
        The path to the checkpoint file.

    """
    ckpt_dir = os.path.join(current_dir, config_dict["logging"]["load_ckpt_path"])
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("rl_model_") and f.endswith("_steps.zip")]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    if config_dict["logging"]["load_step"] > 0:
        ckpt_path = os.path.join(ckpt_dir, "rl_model_{}_steps.zip".format(config_dict["logging"]["load_step"]))
    else:
        max_step_file = max(ckpt_files, key=lambda f: int(f.split("_")[2]))
        ckpt_path = os.path.join(ckpt_dir, max_step_file)
    return ckpt_path


def load_model(
    config_dict: dict,
    current_dir: Union[str, os.PathLike],
    env: Optional[Union[gym.Env, SubprocVecEnv, SubprocShareVecEnv]] = None,
    device: Union[th.device, str] = "auto",
    eval_mode: bool = False,
) -> Union[PPO, th.nn.Module]:
    """Load the model based on the configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Dictionary containing the configuration parameters.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.
    env : Optional[Union[gym.Env, SubprocVecEnv, SubprocShareVecEnv]], optional
        The created environment. Default is None.
    device : Union[th.device, str], optional
        The device to be used for training (CPU or GPU). Default is "auto".
    eval_mode : bool, optional
        If True, the model is set to evaluation mode. Default is False.

    Returns
    -------
    model : Union[PPO, th.nn.Module]
        The loaded or newly created model.

    """
    if env is None and not eval_mode:
        raise ValueError("Environment is not provided. Please provide the environment for training.")
    # get the group name
    group_name = config_dict["pyrl"]["exp_name"]
    # set the policy architecture
    config_dict["rl_hyperparams"]["policy_kwargs"] = config_dict["agent"]
    # get the tensorboard log path
    if config_dict["logging"]["use_tensorboard"]:
        model_name = config_dict["logging"]["save_model_name"]
        model_id = config_dict["logging"]["run_id"]
        tensorboard_log = os.path.join(
            current_dir,
            config_dict["logging"]["tb_dirname"],
            group_name,
            f"{model_name}_{model_id + 1}",
            "logs",
        )
    else:
        tensorboard_log = None

    # load pt file, only supports eval mode and load_model_path
    if config_dict["logging"]["load_model_path"] is not None:
        file_ext = os.path.splitext(config_dict["logging"]["load_model_path"])[1].lower()
        if file_ext == ".pt":
            if not eval_mode:
                raise ValueError("Please provide the zip file for training.")
            if device == "auto":
                device = th.device("cuda" if th.cuda.is_available() else "cpu")
            # save device info to config_dict
            if isinstance(device, th.device):
                use_cuda = device.type == "cuda"
            elif isinstance(device, str):
                use_cuda = device.lower() == "cuda"
            else:
                use_cuda = False
            config_dict["pyrl"]["use_cuda"] = use_cuda
            # load the model from the path
            if config_dict["rl_hyperparams"]["verbose"] > 0:
                print("Loading model from:", config_dict["logging"]["load_model_path"])
            model = th.jit.load(config_dict["logging"]["load_model_path"], map_location=device)

            return model

    # check if the checkpoint path is valid
    if config_dict["logging"]["load_ckpt_path"] is not None:
        file_ext = os.path.splitext(config_dict["logging"]["load_ckpt_path"])[1].lower()
        if file_ext == ".pt":
            raise ValueError(
                "The provided checkpoint file has a .pt extension, which is not supported. Please use --load_model_path to specify a .pt model file instead."
            )

    # load the model
    if config_dict["rl_hyperparams"]["verbose"] > 0:
        print("-" * 50)
        print("[DEVICE INFO]")
        if config_dict["rl_hyperparams"]["verbose"] == 1:
            print(f"Using {device} device")
    if config_dict["logging"]["load_model_path"] is not None:
        # load the model from the path
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print("Loading model from:", config_dict["logging"]["load_model_path"])
        model = PPO.load(path=config_dict["logging"]["load_model_path"], env=env, device=device)
    elif config_dict["logging"]["load_ckpt_path"] is not None:
        # load the model from the checkpoint path
        ckpt_path = _find_checkpoint(config_dict, current_dir)
        if config_dict["rl_hyperparams"]["verbose"] > 0:
            print("Loading checkpoint from:", ckpt_path)
        model = PPO.load(path=ckpt_path, env=env, device=device)
    else:
        # create a new model
        policy_params = {
            **config_dict["rl_hyperparams"],
            **(config_dict["marl_hyperparams"] if "marl_hyperparams" in config_dict else {}),
            "env": env,
            "tensorboard_log": tensorboard_log,
            "seed": config_dict["seed"],
            "device": device,
        }
        policy_params["verbose"] = 1 if policy_params["verbose"] > 1 else 0
        model = PPO(**policy_params)
    if config_dict["rl_hyperparams"]["verbose"] > 0 and not eval_mode:
        print("-" * 50)
        print("[MODEL ARCHITECTURE]")
        print(model.policy)
        print("-" * 50)
        print("[LOGGING INFO]")
        if config_dict["rl_hyperparams"]["verbose"] == 1:
            model_name = config_dict["logging"]["save_model_name"]
            model_id = config_dict["logging"]["run_id"]
            tb_dir = os.path.join(
                current_dir,
                config_dict["logging"]["tb_dirname"],
                group_name,
                f"{model_name}_{model_id + 1}",
                "logs",
            )
            print("Logging to", tb_dir)
    return model


def train_model(model: PPO, config_dict: dict, current_dir: Union[str, os.PathLike]) -> None:
    """Train the model based on the configuration dictionary.

    Parameters
    ----------
    model : PPO
        The PPO model to be trained.
    config_dict : dict
        Dictionary containing the configuration parameters.
    current_dir : Union[str, os.PathLike]
        Current directory of the script.

    """
    # set the callbacks
    group_name = config_dict["pyrl"]["exp_name"]
    save_freq = config_dict["logging"]["model_save_interval"]
    callback_list = []
    rewardcheck_callback = RewardCheckCallback(num_env=config_dict["pyrl"]["n_envs"])
    callback_list.append(rewardcheck_callback)
    if config_dict["logging"]["save_ckpt"]:
        model_name = config_dict["logging"]["save_model_name"]
        model_id = config_dict["logging"]["run_id"]
        checkpoint_save_path = os.path.join(
            current_dir,
            config_dict["logging"]["save_ckpt_dirname"],
            group_name,
            f"{model_name}_{model_id + 1}",
            "ckpts",
        )
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_save_path)
        callback_list.append(checkpoint_callback)
    # start the training
    model.learn(
        total_timesteps=config_dict["pyrl"]["t_max"],
        callback=callback_list,
        log_interval=config_dict["logging"]["log_interval"],
        tb_log_name="run",
        reset_num_timesteps=config_dict["logging"]["reset_num_timesteps"],
        progress_bar=config_dict["logging"]["progress_bar"],
    )

    # save the model
    if config_dict["logging"]["save_model"]:
        model_name = config_dict["logging"]["save_model_name"]
        model_id = config_dict["logging"]["run_id"]
        model_save_path = os.path.join(
            current_dir,
            config_dict["logging"]["save_model_dirname"],
            group_name,
            f"{model_name}_{model_id + 1}",  # directory name
            f"{model_name}_{model_id + 1}",  # model name
        )
        model.save(path=model_save_path)
