# import igibson
# from igibson.envs.igibson_factor_obs_env import iGibsonFactorObsEnv

import datetime
from pathlib import Path
import pprint
import logging
import os
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append('.')
print('SYSTEM PATH:', sys.path)
from baselines.utils import parse_args_and_config
from baselines.flatten_dict_observation_wrapper import FlattenDictObservation


def make_env(config: DictConfig, env_seed: int) -> Callable:

    def _init_env():
        env_config = config.env
        env_name = env_config.env_name
        mini_behavior_config = env_config.mini_behavior
        igibson_config = env_config.igibson

        if config.env.selected_suite == 'mini_behavior': 
            print("Creating mini-behavior env...")
            mb_task = config.env.env_name
            task_specific_config = mini_behavior_config[mb_task]
            env_id = "MiniGrid-" + env_name + "-v0"
            kwargs = {"evaluate_graph": config.env.evaluate_graph,
                    "discrete_obs": mini_behavior_config.discrete_obs,
                    "room_size": task_specific_config.room_size,
                    "max_steps": task_specific_config.max_steps,
                    "use_stage_reward": task_specific_config.use_stage_reward,
                    "random_obj_pose": task_specific_config.random_obj_pose,
                    "task_name": config.env.specific_task_name,
                    "seed": config.ppo.seed*1000 + env_seed,
                    }
            env = gym.make(env_id, **kwargs)

        elif config.env.selected_suite == 'igibson': 
            print("Creating igibson env...")
            igibson_config = config.env.igibson
            igibson_config = OmegaConf.to_container(igibson_config, resolve=True)
            env = iGibsonFactorObsEnv(
                config_file=igibson_config,
                mode="headless",
                action_timestep=1 / 10.0,
                physics_timestep=1 / 120.0,
            )
        
        env = FlattenDictObservation(env)
        return env 
    return _init_env
    

if __name__ == "__main__": 

    now = datetime.datetime.now().strftime("%Y%m%d-%H:%M")
    config = parse_args_and_config()
    args = config.ppo
    run = f'run-seed{args.seed}-{now}'

    model_save_dir = Path(args.model_checkpoints) / config.env.selected_suite

    run_dir = Path(args.log_dir) / args.run_dir / config.env.selected_suite
    tb_log_dir = Path(args.log_dir) / args.tb_log_dir / config.env.selected_suite
    eval_dir = Path(args.log_dir) / args.eval_dir / config.env.selected_suite
    
    env_config_to_log = dict(config.env[config.env.selected_suite])
    env_name = config.env.env_name
    
    if config.env.selected_suite == 'mini_behavior':
        
        # adjust model save and logging for mb
        run_dir = run_dir / env_name / config.env.specific_task_name
        tb_log_dir = tb_log_dir / env_name / config.env.specific_task_name
        model_save_dir = model_save_dir / env_name / config.env.specific_task_name
        env_config_to_log = env_config_to_log[config.env.env_name]
    
    else:
        run_dir = run_dir / env_name / config.env['igibson'].downstream_task
        tb_log_dir = tb_log_dir / env_name / config.env['igibson'].downstream_task
        model_save_dir = model_save_dir / env_name / config.env['igibson'].downstream_task

    
    model_save_dir = model_save_dir / run

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    logging.basicConfig(filename=run_dir/(run + '.log'), filemode='w', level=logging.INFO)

    ppo_config_to_log = dict(config.ppo)
    pretty_env = pprint.pformat(env_config_to_log, compact=False, indent=2, width=20)
    pretty_ppo = pprint.pformat(ppo_config_to_log, compact=False, indent=2, width=20)
    logging.info(f"\nENV CONFIG: {config.env.env_name}\n{pretty_env}")
    logging.info(f"\nPPO CONFIG: {pretty_ppo}")

    print(f"Setting seed to {args.seed}")
    set_random_seed(args.seed)

    from mini_behavior.wrappers.flatten_dict_observation import FlattenDictObservation

    # env = create_env(config)
    # env.reset()
    # a = env.action_space.sample() 
    # obs, reward, terminated, truncated, info = env.step(a)
    # print(info)
    # exit()
    train_envs = VecMonitor(SubprocVecEnv([make_env(config, i)
                                           for i in range(config.env.num_train_envs)]))
    eval_envs = VecMonitor(SubprocVecEnv([make_env(config, i)
                                          for i in range(config.env.num_test_envs)]))

    # set maximum steps to act
    if args.total_timesteps:
        logging.info("Overriding steps with override_steps...")
        total_steps = args.total_timesteps
    else:
        env_steps_dict = config.ppo.env_steps[env_name] 
        total_steps = env_steps_dict['skill'] + env_steps_dict['task'] 
    
    # setup eval callback
    callback_after_eval = None
    
    if config.ppo.early_stopping:
        print("-------Using early stopping!-------")
        early_stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.max_no_improvement_evals,
            min_evals=args.min_evals_before_stopping,
            verbose=1
        )
        callback_after_eval = early_stop_callback
     
    # elif args.override_steps is None and not config.ppo.eval_callback:
    #     eval_callback = None
    #     # compute total timesteps
    #     env_name = config.env.env_name
    #     env_steps_dict = config.ppo.env_steps[env_name] 
    #     total_steps = env_steps_dict['skill'] + env_steps_dict['task']
        
    # elif args.override_steps and not config.ppo.eval_callback:
    #     eval_callback = None
    #     print("Overriding steps with override_steps...")
    #     logging.info("Overriding steps with override_steps...")
    #     total_steps = args.total_timesteps

    # # log success rate with eval_callback
    # if config.ppo.early_stopping and config.ppo.log_success_rate:
    #     raise NotImplementedError

    eval_callback = EvalCallback(
        eval_envs,
        eval_freq=args.eval_freq,
        callback_after_eval=callback_after_eval,
        n_eval_episodes=config.env.num_test_envs,
        best_model_save_path=model_save_dir,
        log_path=eval_dir,
        verbose=1
    )

    if config.ppo.saved_model_fname is None:
        logging.info(f"Not using saved model")
        model = PPO("MlpPolicy", train_envs, verbose=1,
                    tensorboard_log=tb_log_dir,
                    # policy_kwargs=policy_kwargs,
                    seed=args.seed,
                    clip_range=args.clip_range,
                    learning_rate=args.ppo_lr)
    else:
        logging.info(f"Using saved model at {config.ppo.saved_model_fname}")
        load_model_path = model_save_dir / config.ppo.saved_model_fname
        model = PPO.load(str(load_model_path))

    # Random Agent, evaluation before training
    mean_reward, std_reward = evaluate_policy(model, eval_envs,n_eval_episodes=1)
    print(f"Before Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")


    # Train the model for the given number of steps    
    logging.info(f'Acting for {total_steps}...') 
    print(f'Acting for {total_steps}...')
    model.learn(total_steps, tb_log_name=run, callback=eval_callback)

    # Evaluate the policy after training
    mean_reward, std_reward = evaluate_policy(model, eval_envs,
                                              n_eval_episodes=args.num_test_envs)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

    # # Save the trained model and delete it
    # model.save(str(model_save_dir))
    # del model

    # # Reload the trained model from file
    # model = PPO.load(model_cp_path)

    # Evaluate the trained model loaded from file
    mean_reward, std_reward = evaluate_policy(model, eval_envs,
                                              n_eval_episodes=args.num_test_envs)
    print(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")