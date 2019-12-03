import numpy as np
from gym import spaces
import gym
import os
import copy
from gym.utils import seeding
import matplotlib.pyplot as plt

ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
# define colors
# 0: dark; 1 : gray; 2(agent) : blue; 3 : blank; 4:green; 5 : red

# COLORS = {0: [0., 0., 0.], 1: [127.5, 127.5, 127.5],
#           2: [0., 0., 255.], 3: [255., 255., 255.],
#           4: [0., 255., 0.], 5: [255.0, 0., 0.],
#           6: [255., 0., 255.], 7: [255., 255., 0.]}

# background: 0 ,1 and agent: 2
BGAndAG_COLORS = {0: [0., 0., 0.], 1: [169., 169., 169.],
                  2: [105., 105., 105.]}
# gem: 3
CorrectBox_COLORS = {3: [255., 255., 255.],
                     4: [0., 255., 0.], 5: [255.0, 0., 0.],
                     6: [255., 0., 255.], 7: [255., 255., 0.]}

DistractorBox_COLORS = {8: [0., 255., 255.], 9: [255.0, 127.5, 127.5],
                        10: [127.5, 0., 255.], 11: [255., 127.5, 0.],
                        12: [127.5, 127.5, 255.], 13: [0., 127.5, 127.5],
                        14: [127.5, 127.5, 0.], 15: [255., 0., 127.5], }

COLORS = dict(list(BGAndAG_COLORS.items()) + list(CorrectBox_COLORS.items()) + list(DistractorBox_COLORS.items()))

# branch_length = 1
EASY_BOX_LIST = [(7, 6), (4, 7), (5, 4), (3, 5), (8, 7), (9, 5), (10, 5)]
EASY_END_LIST = [8, 9, 10]
# branch_length = 2
MEDIUM_BOX_LIST = [(7, 6), (4, 7), (5, 4), (3, 5), (8, 7), (11, 8), (9, 4), (12, 9), (10, 5), (13, 10)]
MEDIUM_END_LIST = [11, 12, 13]
# branch_length = 3
HARD_BOX_LIST = [(7, 6), (4, 7), (5, 4), (3, 5), (8, 7), (11, 8), (9, 4), (12, 9), (14, 12), (10, 5), (13, 10), (15, 13)]
HARD_END_LIST = [11, 14, 15]
BOX_DICT = {'easy': EASY_BOX_LIST, 'medium': MEDIUM_BOX_LIST, 'hard': HARD_BOX_LIST}
END_DICT = {'easy': EASY_END_LIST, 'medium': MEDIUM_END_LIST, 'hard': HARD_END_LIST}


class BoxWoldRandEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, level):
        self.level = level
        self.seed()
        # self._action_set = [0, 2, 3, 4, 5]
        # self.action_space = spaces.Discrete(len(self._action_set))
        # self.action_pos_dict = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}
        # ignore noop action
        self._action_set = [2, 3, 4, 5]
        self.action_space = spaces.Discrete(len(self._action_set))
        self.action_pos_dict = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}

        # set observation space
        self.box_size = 14
        # scale
        gs = 1
        self.obs_shape = [self.box_size * gs, self.box_size * gs, 3]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)

        # initialize system
        self.CorrectBox_lists = list(range(3, 3 + len(CorrectBox_COLORS)))
        # DistractorBox_lists = range(3 + len(CorrectBox_COLORS), 3 + len(CorrectBox_COLORS) + len(DistractorBox_COLORS))

        self.max_timesteps = 5000
        self.reset()

    def set_box(self, box):
        while True:
            LEFT_BOX = self.np_random.randint(0, 122)
            LEFT_BOX_x = LEFT_BOX // 11 + 1
            LEFT_BOX_y = LEFT_BOX % 11 + 1
            if self.init_world_map[LEFT_BOX_x][LEFT_BOX_y] == 1 and self.init_world_map[LEFT_BOX_x][LEFT_BOX_y + 1] == 1 and self.init_world_map[LEFT_BOX_x][LEFT_BOX_y - 1] == 1 and self.init_world_map[LEFT_BOX_x][LEFT_BOX_y + 2] == 1:
                self.init_world_map[LEFT_BOX_x][LEFT_BOX_y] = box[0]
                self.init_world_map[LEFT_BOX_x][LEFT_BOX_y + 1] = box[1]
                break

    def reset(self):
        self.timesteps = 0
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.world_map_path = os.path.join(this_file_path, 'blank.txt')
        self.init_world_map = self._read_world_map(self.world_map_path)
        # SET RAND MAP
        # SET AGENT LOCATION
        agent_init_position = self.np_random.randint(0, 144)
        agent_init_x = agent_init_position // 12 + 1
        agent_init_y = agent_init_position % 12 + 1
        self.init_world_map[agent_init_x][agent_init_y] = 2

        # SET BOX
        while True:
            First_CBOX = self.np_random.randint(0, 144)
            First_CBOX_x = First_CBOX // 12 + 1
            First_CBOX_y = First_CBOX % 12 + 1
            if self.init_world_map[First_CBOX_x][First_CBOX_y] == 1:
                self.init_world_map[First_CBOX_x][First_CBOX_y] = 6
                break

        assert(self.level in BOX_DICT.keys()), 'BoxWoldRandEnv only support {} levels'.format(list(BOX_DICT.keys()))

        for BOX in BOX_DICT[self.level]:
            self.set_box(BOX)

        self.current_world_map = copy.deepcopy(self.init_world_map)
        self.observation = self._worldmap_to_obervation(self.init_world_map)
        self.world_map_shape = self.init_world_map.shape

        # agent states
        self.agent_init_state = self._get_agent(self.init_world_map)
        self.agent_current_state = copy.deepcopy(self.agent_init_state)
        self.key = 0  # init no key (dark)
        return self.observation

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        # self.ale.setInt(b'random_seed', seed2)
        # self.ale.loadROM(self.game_path)
        return [seed1, seed2]

    def step(self, action):
        info = {}
        info['success'] = False
        action = int(action) + 1  # ignore noop action
        next_agent_state = [self.agent_current_state[0] + self.action_pos_dict[action][0], self.agent_current_state[1] + self.action_pos_dict[action][1]]

        if self.timesteps > self.max_timesteps:
            return (self.observation, 0, True, info)
        self.timesteps += 1
        if action == 0:
            return (self.observation, 0, False, info)

        # invalid action cross border
        if next_agent_state[0] < 1 or next_agent_state[0] > (self.world_map_shape[0] - 2):
            return (self.observation, 0, False, info)
        if next_agent_state[1] < 1 or next_agent_state[1] > (self.world_map_shape[0] - 2):
            return (self.observation, 0, False, info)

        nxt_color = self.current_world_map[next_agent_state[0], next_agent_state[1]]
        nxt_left_color = self.current_world_map[next_agent_state[0], next_agent_state[1] - 1]
        nxt_right_color = self.current_world_map[next_agent_state[0], next_agent_state[1] + 1]

        # the content of the box is inaccessible while the box is locked
        if nxt_color > 2 and nxt_right_color > 2:
            return (self.observation, 0, False, info)

        if nxt_color > 2:
            # pick up the content of the box(single pixel)
            if nxt_left_color < 3:
                if nxt_color == 3:
                    reward = 10
                    done = True
                    info['success'] = True
                    self._update_key(nxt_color)
                    self._agent_move(next_agent_state)
                    # self.observation = self.reset()
                    return (self.observation, reward, done, info)
                else:
                    reward = 1
                    done = False
                    self._update_key(nxt_color)
                    self._agent_move(next_agent_state)
                    return (self.observation, reward, done, info)

            # unlock the box
            elif self.key == nxt_color:
                if nxt_left_color in self.CorrectBox_lists:
                    if nxt_left_color == 3:
                        reward = 10
                        done = True
                        info['success'] = True
                    else:
                        reward = 1
                        done = False
                    # add for simple env
                    self.current_world_map[next_agent_state[0], next_agent_state[1] - 1] = 1
                    # change self._update_key(0) for simple env
                    self._update_key(nxt_left_color)
                    self._agent_move(next_agent_state)

                    return (self.observation, reward, done, info)
                else:
                    reward = -1
                    if nxt_left_color in END_DICT[self.level]:
                        done = True
                    else:
                        done = False
                    # add for simple env
                    self.current_world_map[next_agent_state[0], next_agent_state[1] - 1] = 1
                    # change self._update_key(0) for simple env
                    self._update_key(nxt_left_color)
                    self._agent_move(next_agent_state)
                    # self.observation = self.reset()
                    return (self.observation, reward, done, info)

            return (self.observation, 0, False, info)

        self._agent_move(next_agent_state)
        return (self.observation, 0, False, info)

    def _update_key(self, nxt_color):
        self.key = nxt_color
        self.current_world_map[0, 0] = self.key

    def _agent_move(self, next_agent_state):
        self.current_world_map[next_agent_state[0], next_agent_state[1]] = 2
        self.current_world_map[self.agent_current_state[0], self.agent_current_state[1]] = 1

        self.agent_current_state = copy.deepcopy(next_agent_state)
        self.observation = self._worldmap_to_obervation(self.current_world_map)

    def _get_agent(self, world_map):
        location = np.where(world_map == 2)
        agent = list(map(lambda x: x[0], location))
        return agent

    def get_current_agent_position(self):
        # from 2d(x,y) to 1d
        position = self.agent_current_state[0] * self.box_size + self.agent_current_state[1]
        return position

    def _read_world_map(self, path):
        with open(path, 'r') as f:
            world_map = f.readlines()
            # world_map_splited = list(map(lambda x: x.split(' '), world_map))
            # world_map_array = [list(map(lambda x: int(x), row)) for row in world_map_splited]
            # world_map_array = np.array(world_map_array)
            # equal to the next one line
            world_map_array = np.array(list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), world_map)))

            return world_map_array

    def _worldmap_to_obervation(self, world_map):
        obs_shape = self.obs_shape
        observation = np.zeros(obs_shape, dtype=np.float32)
        gs0 = int(observation.shape[0] / world_map.shape[0])
        gs1 = int(observation.shape[1] / world_map.shape[1])
        for i in range(world_map.shape[0]):
            for j in range(world_map.shape[1]):
                observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1] = np.array(COLORS[world_map[i, j]])
        return observation

    def render(self, model='huamn'):
        img = self.observation
        fig = plt.figure(0)
        plt.clf()
        plt.imshow(img / 255)
        fig.canvas.draw()
        plt.pause(0.0001)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


if __name__ == '__main__':
    env = BoxWoldRandEnv(level='medium')
    # ob = env.observation
    # ob2 = np.mean(ob, axis=2)

    while True:
        action = -1
        while action not in list(range(4)):
            action = int(input("Action 0~3 input:"))
        observation, reward, done, info = env.step(action)
        print(env.agent_current_state)
        print(env.get_current_agent_position())
        # observation, reward, done, info = env.step(env.action_space.sample())
        print(reward, done, info)
        env.render()
    # location = np.where(env.init_world_map == 4)
    # print(type(location))
