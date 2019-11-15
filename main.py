import argparse
import os
import gym
import gym_boxworld
from stable_baselines import A2C
from stable_baselines.common.policies import CnnPolicy
from relational_policies import RelationalPolicy  # custom Policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor


def make_env(env_id, rank, log_dir, useMonitor=True, seed=0):
    def _init():
        env = gym.make(env_id)
        if useMonitor:
            env = Monitor(env, log_dir + str(rank), allow_early_resets=True)
        return env

    set_global_seeds(seed)
    return _init


def set_model(model_name, policy_name, env):
    policy = {'CnnPolicy': CnnPolicy, 'Relational_Policy': RelationalPolicy}
    base_mode = {'A2C': A2C}
    model = base_mode[model_name](policy[policy_name], env, verbose=1)
    return model


def run(config):
    log_dir = '{}/{}_{}/log_0/'.format(config.log_dir, config.env_name, config.model_name)
    # if log_dir exists,auto add new dir by order
    while os.path.exists(log_dir):
        lastdir_name = log_dir.split('/')[-2]
        order = int(lastdir_name.split('_')[-1])
        log_dir = log_dir.replace('_{}'.format(order), '_{}'.format(order + 1))
    os.makedirs(log_dir)
    with open(log_dir + 'config.txt', 'wt') as f:
        f.write(str(config))
    print(("--------------------------Create dir:{} Successful!--------------------------\n").format(log_dir))

    env_id = config.env_name + 'NoFrameskip-v4'
    env = SubprocVecEnv([make_env(env_id, i, log_dir) for i in range(config.num_cpu)])
    model = set_model(config.model_name, config.policy_name, env)
    print(("--------Algorithm:{} with {} num_cpu:{} total_timesteps:{} Start to train!--------\n")
          .format(config.model_name, config.policy_name, config.num_cpu, config.total_timesteps))

    # model.learn(total_timesteps=int(1e7), callback=callback)
    model.learn(total_timesteps=config.total_timesteps)
    if config.save:
        model.save(log_dir + config.model_name + '_' + config.env_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", choices=['BoxRandWorld', 'BoxWorld'], help="Name of environment")
    parser.add_argument("policy_name", choices=['Relational_Policy', 'CnnPolicy'], help="Name of policy")
    parser.add_argument("-model_name", choices=['A2C'], default='A2C', help="Name of model")

    parser.add_argument("-num_cpu", default=4, type=int)
    parser.add_argument("-total_timesteps", default=int(2e6), type=int)
    parser.add_argument("-log_dir", default='exp_result')
    parser.add_argument("-save", action='store_false')

    config = parser.parse_args()
    # print(config)
    run(config)
    print('Over!')
