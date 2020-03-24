from gym_duckietown.envs.duckietown_env import DuckietownNav
from gym import spaces
from gym.utils import EzPickle
import numpy as np

class Duckietown(DuckietownNav, EzPickle):
    def __init__(self):
        kwargs = dict(seed=1, map_name='left_right_turn', frame_skip=2, domain_rand=False,
                              distortion=False, user_tile_start=[0, 2], full_transparency=True,
                              verbose=False, graphics=True)
        self.test_domain = 0.15
        DuckietownNav.__init__(self, **kwargs)
        self.cur_angle = 0
        EzPickle.__init__(self, **kwargs)

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    @property
    def observation_space(self):
        """ current position x, current position y, velocity forward, steering angle, collision,
            goal_x, goal_y,
        """
        return spaces.Box(low=np.array([0, 0, -1, -1, 0, 0, 0]),
                          high=np.array([self.grid_height * self.road_tile_size,
                                         self.grid_width * self.road_tile_size,
                                         1,
                                         1,
                                         1,
                                         self.grid_height * self.road_tile_size,
                                         self.grid_width * self.road_tile_size,
                                        ]),
                          dtype=np.float32)

    def step(self, action):
        #action = action + self.np_random.
        obs, reward, done, info = DuckietownNav.step(self, action)
        if info['Simulator']['msg'] == 'hit-wall':
            if self.mode == 'test':
                done = True
            #done = True
            obs = np.concatenate((obs, [1]), axis=0)
            info['Catastrophe'] = True
        else:
            obs = np.concatenate((obs, [0]), axis=0)
            info['Catastrophe'] = False
        obs = np.concatenate((obs, np.array(self.goal_tile['coords'])), axis=0)
        #    self.cur_pos = np.array([0.2925, 0, 1.4625])
        #    #self.cur_pos = np.array([1.0, 0, 1.62])
        #    self.cur_angle = 0
        #    obs[0] = self.cur_pos[0]
        #    obs[1] = self.cur_pos[2]
        #    obs[3] = self.cur_angle
        #obs = np.concatenate((obs, np.array(info['goal_tile']['coords'])), axis=0)
        return obs, reward, done, info

    def reset(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.ROBOT_WIDTH = self.np_random.uniform(0.05, 0.1)
        elif mode == 'test':
            self.ROBOT_WIDTH = self.test_domain
        self.WHEEL_DIST = self.ROBOT_WIDTH - 0.03
        self.AGENT_SAFETY_RAD = (max(self.ROBOT_LENGTH, self.ROBOT_WIDTH) / 2) * self.SAFETY_RAD_MULT
        obs = DuckietownNav.reset(self)
        self.cur_pos = np.array([0.2925, 0, 1.4625])
        #self.cur_pos = np.array([0.75, 0, 1.65])
        self.cur_angle = 0
        obs[0] = self.cur_pos[0]
        obs[1] = self.cur_pos[2]
        obs[3] = self.cur_angle
        obs = np.concatenate((obs, [0], np.array(self.goal_tile['coords'])), axis=0)
        return obs
    def render(self, mode):
        return DuckietownNav.render(self, "top_down")