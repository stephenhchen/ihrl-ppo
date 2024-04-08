import argparse
from argparse import Namespace
import datetime
import os
from pathlib import Path
from typing import Dict
import yaml
from omegaconf import DictConfig

def parse_configs(config_dir='./baselines/configs',
                  primary_config='config.yaml',
                  env_suites=('igibson','mini_behavior')) -> Dict:

    parent_config = Path(config_dir)

    # load primary config
    primary_path = parent_config / primary_config
    with open(primary_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # load env configs 
    for env in env_suites:
        env_config_path = Path(config_dir) / 'env' / (env + '.yaml')
        env_config = load_env_config(env_config_path)
        config['env'][env] = env_config
    
    return config


def _override_config(config: DictConfig, args: Namespace, config_level):
    for arg_key, arg_val in vars(args).items():
        # only override csd keys
        for k in config[config_level].keys():
            if arg_key == k:
                if arg_val is None:
                    print(f"Not overriding {arg_key}")
                    continue
                print(f"Overriding config with passed in argument: {arg_key}={arg_val}")
                config[config_level][k] = arg_val

        if config_level == 'env':
            # maybe override an arg in igibson.yaml or mini_behavior.yaml
            for k in config[config_level]['igibson']:
                if arg_key == k:
                    if arg_val is None:
                        continue
                    print(f"Overriding igibson config with passed in argument: {arg_key}={arg_val}")
                    config[config_level]['igibson'][k] = arg_val
        
    return config


def adjust_config(config: DictConfig, args: Namespace) -> DictConfig: 
    """Map ihrl config to CSD configs and override args from command line"""
    if "csd" in config.keys():
        config = _override_config(config, args, 'csd')
        print("csd found in config")
    config = _override_config(config, args, 'env')
    config = _override_config(config, args, 'ppo')
    # print("env name after override", config.env.env_name)

    # mb_tasks = config['env']['mini_behavior'].keys()
    # mb_tasks = {
    #     'installing_printer': ['install_printer'],
    #     'thawing': ['thaw_fish', 'thaw_date', 'thaw_olive', 'thaw_any_two', 'thaw_all'],
    #     'cleaning_car': ['soak_raw', 'clean_car', 'clean_rag']
    # }
    mb_tasks = config['env']['mini_behavior']
    print("mb_tasks:", mb_tasks)
    env_name = config['env']['env_name']

    if env_name in mb_tasks.keys(): 
        config.env['selected_suite'] = 'mini_behavior'
    elif env_name == 'igibson':
        config.env['selected_suite'] = 'igibson'
    else:
        raise NotImplementedError

    if config.env.selected_suite == 'mini_behavior':
        mb_task = config.env.env_name
        max_steps = config.env.mini_behavior[mb_task].max_steps
        if 'csd' in config.keys():
            config.csd.max_path_length = max_steps
            config.csd.discrete_actions = True

    elif config.env.selected_suite == 'igibson':
        if 'csd' in config.keys():
            config.csd.max_path_length = config.env.igibson.max_step
            config.csd.discrete_actions = False

    # additional args not in original csd
    # config.csd.default_network_dims = [config.csd.default_network_size] * 2

    return config


def parse_args_and_config() -> DictConfig:
    """All default values need to be None to use config.yaml values"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary_config', default='config.yaml')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--env_name', default=None)
    parser.add_argument('--total_timesteps', type=int, default=None) 
    parser.add_argument('--clip_range', type=float, default=None)
    parser.add_argument('--ppo_lr', type=float, default=None)
    parser.add_argument('--sac_lr_a', type=float, default=None)
    parser.add_argument('--train_traj_epochs', type=float, default=None)
    parser.add_argument('--sac_replay_buffer', type=int, default=None)
    parser.add_argument('--te_trans_optimization_epochs', type=int, default=None)
    parser.add_argument('--trans_optimization_epochs', type=int, default=None)
    parser.add_argument('--trans_minibatch_size', type=int, default=None)
    parser.add_argument('--discrete', type=int, default=None)
    parser.add_argument('--dim_option', type=int, default=None)
    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--specific_task_name', type=str, default=None)
    parser.add_argument('--downstream_task', type=str, default=None)
    

    args = parser.parse_args()

    config = parse_configs(primary_config=args.primary_config)
    config = DictConfig(content=config)
    config = adjust_config(config, args)
    return config


def load_env_config(config_path: Path) -> dict:
    env_suite = config_path.stem    # mini_behavior or igibson
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        suite_config = config[env_suite]    

        config = {}
        if env_suite == 'mini_behavior':
            for k, v in suite_config.items():
                if isinstance(v, dict):
                    config[k] = dict(v) 
                else:
                    config[k] = v
        elif env_suite == 'igibson':
            config = suite_config

    return config

########################## EXPERIMENT HELPERS #################################
def get_exp_name(args: argparse.Namespace,
                 hack_slurm_job_id_override=None, g_start_time=None):
    # parser = get_argparser()
    if g_start_time is None: 
        g_start_time = int(datetime.datetime.now().timestamp())

    exp_name = ''
    exp_name += f'sd{args.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ or hack_slurm_job_id_override is not None:
        exp_name += f's_{hack_slurm_job_id_override or os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name_prefix = exp_name
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'

    exp_name_abbrs = set()
    exp_name_arguments = set()

    def list_to_str(arg_list):
        return str(arg_list).replace(",", "|").replace(" ", "").replace("'", "")

    def add_name(abbr, argument, value_dict=None, max_length=None, log_only_if_changed=True):
        nonlocal exp_name

        if abbr is not None:
            assert abbr not in exp_name_abbrs
            exp_name_abbrs.add(abbr)
        else:
            abbr = ''
        exp_name_arguments.add(argument)

        value = getattr(args, argument)
        # if log_only_if_changed and parser.get_default(argument) == value:
        #     return
        if isinstance(value, list):
            if value_dict is not None:
                value = [value_dict.get(v) for v in value]
            value = list_to_str(value)
        elif value_dict is not None:
            value = value_dict.get(value)

        if value is None:
            value = 'X'

        if max_length is not None:
            value = str(value)[:max_length]

        if isinstance(value, str):
            value = value.replace('/', '-')

        exp_name += f'_{abbr}{value}'

    # add_name(None, 'env', {
    #     'half_cheetah': 'CH',
    #     'ant': 'ANT',
    #     'humanoid': 'HUM',
    # }, log_only_if_changed=False)

    add_name('clr', 'common_lr', log_only_if_changed=False)
    add_name('slra', 'sac_lr_a', log_only_if_changed=False)
    add_name('a', 'alpha', log_only_if_changed=False)
    add_name('sg', 'sac_update_target_per_gradient', log_only_if_changed=False)
    add_name('do', 'dim_option', log_only_if_changed=False)
    add_name('sr', 'sac_replay_buffer')
    add_name('md', 'model_master_dim')
    add_name('sdc', 'sac_discount', log_only_if_changed=False)
    add_name('ss', 'sac_scale_reward')
    add_name('ds', 'discrete')
    add_name('in', 'inner')
    add_name('dr', 'dual_reg')
    if args.dual_reg:
        add_name('dl', 'dual_lam')
        add_name('dk', 'dual_slack')
        add_name('dd', 'dual_dist', max_length=1)

    # Check lr arguments
    for key in args:
        if key.startswith('lr_') or key.endswith('_lr') or '_lr_' in key:
            val = getattr(args, key)
            assert val is None or bool(val), 'To specify a lr of 0, use a negative value'

    return exp_name, exp_name_prefix


# def get_log_dir(args, start_time=None):
#     if start_time is None:
#         start_time = datetime.datetime.now().stftime("%Y-%m-%d %M:%S")
        
#     exp_name, exp_name_prefix = get_exp_name(args)
#     assert len(exp_name) <= os.pathconf('/', 'PC_NAME_MAX')
#     # Resolve symlinks to prevent runs from crashing in case of home nfs crashing.
#     log_dir = os.path.realpath(os.path.join('exp', start_time, exp_name))
#     assert not os.path.exists(log_dir), f'The following path already exists: {log_dir}'

#     print('log dir:', log_dir)
#     return log_dir

# utils.py


class ObservationIndexer:
    """
    Bijection between flattened boolean environment vectors and integers
    starting at 0.

    Attributes:
        vecs_to_ints
        ints_to_vecs
    """
    def __init__(self):
        self.vecs_to_ints = {}
        self.ints_to_vecs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.vecs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the bool_obs_vec corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_vecs):
            return None
        else:
            return self.ints_to_vecs[index]

    def contains(self, bool_obs_vec):
        """
        :param bool_obs_vec: bool_obs_vec to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(bool_obs_vec) != -1

    def index_of(self, bool_obs_vec):
        """
        :param bool_obs_vec: bool_obs_vec to look up
        :return: Returns -1 if the bool_obs_vec isn't present, index otherwise
        """
        if (bool_obs_vec not in self.vecs_to_ints):
            return -1
        else:
            return self.vecs_to_ints[bool_obs_vec]

    def add_and_get_index(self, bool_obs_vec, add=True):
        """
        Adds the bool_obs_vec to the index if it isn't present, always returns a nonnegative index
        :param bool_obs_vec: bool_obs_vec to look up or add
        :param add: True by default, False if we shouldn't add the bool_obs_vec. If False, equivalent to index_of.
        :return: The index of the bool_obs_vec
        """
        if not add:
            return self.index_of(bool_obs_vec)
        if (bool_obs_vec not in self.vecs_to_ints):
            new_idx = len(self.vecs_to_ints)
            self.vecs_to_ints[bool_obs_vec] = new_idx
            self.ints_to_vecs[new_idx] = bool_obs_vec
        return self.vecs_to_ints[bool_obs_vec]
