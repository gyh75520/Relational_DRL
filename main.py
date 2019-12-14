import argparse
import json
import os
import gym
import gym_boxworld
from stable_baselines import A2C
from stable_baselines.common.policies import CnnPolicy
from relational_policies import RelationalPolicy, RelationalLstmPolicy  # custom Policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from stable_baselines.common.atari_wrappers import FrameStack


def saveInLearn(log_dir):
    # A unit of time saved
    unit_time = int(1e7)

    def callback(_locals, _globals):
        num_timesteps = _locals['self'].num_timesteps
        if num_timesteps >= 10 * unit_time and num_timesteps % unit_time == 0:
            _locals['self'].save(log_dir + 'model_{}.pkl'.format(num_timesteps))
        return True
    return callback


def make_env(env_id, env_level, rank, log_dir, frame_stack=False, useMonitor=True, seed=0):
    def _init():
        env = gym.make(env_id, level=env_level)
        if frame_stack:
            env = FrameStack(env, 4)
        if useMonitor:
            env = Monitor(env, log_dir + str(rank), allow_early_resets=True)
        return env

    set_global_seeds(seed)
    return _init


def set_logdir(config):
    log_dir = '{}/{}{}_{}_{}/log_0/'.format(config.log_dir, config.env_name, config.env_level, config.model_name, config.policy_name)
    # if log_dir exists,auto add new dir by order
    while os.path.exists(log_dir):
        lastdir_name = log_dir.split('/')[-2]
        order = int(lastdir_name.split('_')[-1])
        log_dir = log_dir.replace('_{}'.format(order), '_{}'.format(order + 1))
    os.makedirs(log_dir)
    with open(log_dir + 'config.txt', 'wt') as f:
        json.dump(config.__dict__, f, indent=2)
    print(("--------------------------Create dir:{} Successful!--------------------------\n").format(log_dir))
    return log_dir


def set_env(config, log_dir):
    env_id = config.env_name + 'NoFrameskip-v4'
    env = SubprocVecEnv([make_env(env_id, config.env_level, i, log_dir, frame_stack=config.frame_stack) for i in range(config.num_cpu)])
    return env


def set_model(config, env, log_dir):
    if config.timeline:
        from timeline_util import _train_step
        A2C.log_dir = log_dir
        A2C._train_step = _train_step
    policy = {'CnnPolicy': CnnPolicy, 'RelationalPolicy': RelationalPolicy, 'RelationalLstmPolicy': RelationalLstmPolicy}
    base_mode = {'A2C': A2C}
<<<<<<< HEAD
    model = base_mode[config.model_name](policy[config.policy_name], env, verbose=1, tensorboard_log=log_dir,  full_tensorboard_log=True if log_dir else False)
=======
    # whether reduce oberservation
    policy[config.policy_name].reduce_obs = config.reduce_obs
    model = base_mode[config.model_name](policy[config.policy_name], env, verbose=1)
>>>>>>> reduce_relational_block
    print(("--------Algorithm:{} with {} num_cpu:{} total_timesteps:{} Start to train!--------\n")
          .format(config.model_name, config.policy_name, config.num_cpu, config.total_timesteps))
    return model


def run(config):
    log_dir = set_logdir(config)
    env = set_env(config, log_dir)
    model = set_model(config, env, log_dir)
    model.learn(total_timesteps=int(config.total_timesteps), callback=saveInLearn(log_dir) if config.save else None)
    # if config.save:
    #     model.save(log_dir + 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", choices=['BoxRandWorld', 'BoxWorld'], help="Name of environment")
    parser.add_argument("-env_level", choices=['easy', 'medium', 'hard'], default='easy', help="level of environment")

    parser.add_argument("policy_name", choices=['RelationalPolicy', 'CnnPolicy', 'RelationalLstmPolicy'], help="Name of policy")
    parser.add_argument("-model_name", choices=['A2C'], default='A2C', help="Name of model")
    parser.add_argument("-reduce_obs", action='store_true')

    parser.add_argument("-timeline", action='store_true', help='performance analysis,default=False')
    parser.add_argument("-frame_stack", action='store_true', help='whether use frame_stack, default=False')
    parser.add_argument("-cuda_device", default='1', help='which cuda device to run, default="1"')
    parser.add_argument("-num_cpu", default=4, type=int, help='whether use frame_stack, default=False')
    parser.add_argument("-total_timesteps", default=2e6, type=float, help='total train timesteps, default=2e6')
    parser.add_argument("-log_dir", default='exp_result', help='log_dir path, default="exp_result"')
    parser.add_argument("-save", action='store_true', help='whether save model to log_dir, default=False')

    config = parser.parse_args()
    # print(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_device
    run(config)
    print('Over!')
