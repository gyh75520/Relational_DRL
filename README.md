# Relational Deep Reinforcement Learning
## Requirements
[stable-baselines](https://github.com/hill-a/stable-baselines), commit hash:98257ef8c9bd23a24a330731ae54ed086d9ce4a7a1ab7a1c2903e7e1c38756d8cdf7a54a5fd5781e
### Install boxworld environment
Go to the `env/gym-box-world` folder and run the command :
```
pip install -e .
```

This will install the box-world environment. Now, you can use this enviroment with the following:
```
import gym
import gym_boxworld
env_name = 'BoxRandWorld'
env_id = env_name + 'NoFrameskip-v4'
env = gym.make(env_id)
```
[More details about the Env](https://github.com/gyh75520/Relational_DRL/blob/master/env/gym-box-world/README.md)
