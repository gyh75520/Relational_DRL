# Installation
## Install boxworld environment
Go to the `env/gym-box-world` folder and run the command :
```
pip install -e .
```

This will install the box-world environment. Now, you can use this enviroment with the following:
```
import gym
import gym_boxworld
env_name = 'BoxWorldNoFrame'
env_id = env_name + 'NoFrameskip-v4'
env = gym.make('env_id)
```
[More details about the Env]()
