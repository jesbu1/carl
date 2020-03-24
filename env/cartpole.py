import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
import os
from filelock import FileLock
import xml.etree.ElementTree

class CartPoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.xml_location = '%s/assets/cartpole.xml' % dir_path
        self.mode = 'train'
        self.test_domain = 1.0
        self.domain_low = 0.4
        self.domain_high = 0.8
        self.pendulum_length = 0.6
        mujoco_env.MujocoEnv.__init__(self, self.xml_location, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, self.pendulum_length]))) / (self.pendulum_length ** 2)
        )
        catastrophe = (np.abs(ob[1]) > np.pi/2) or (np.abs(ob[0]) >= 2.4)
        info = {}
        if catastrophe:
            ob[-1] = 1
            info['Catastrophe'] = True
        else:
            info['Catastrophe'] = False
        notdone = np.isfinite(ob).all() and not (catastrophe and self.mode == 'test')
        done = not notdone
        return ob, reward, done, info

    def reset_model(self):
        if not hasattr(self, "pendulum_length"):
            self.pendulum_length = self.np_random.uniform(self.domain_low, self.domain_high)
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        original_obs = np.concatenate([self.data.qpos, self.data.qvel]).ravel()
        curr_obs = np.concatenate([original_obs, [self.pendulum_length, 0]], axis=-1)
        return curr_obs

    def _get_ee_pos(self, x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 + self.pendulum_length * np.sin(theta),
            self.pendulum_length * np.cos(theta)
        ])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def set_length(self, length):
        lock = FileLock(self.xml_location + '.lock')  # concurrency protection
        with lock:
            et = xml.etree.ElementTree.parse(self.xml_location)
            et.find('worldbody').find('body').find('body').find('geom').set('fromto',
                                                                            "0 0 0 0.001 0 %0.3f" % length)  # changing size of pole
            et.write(self.xml_location)
            self.model = mujoco_py.load_model_from_path(self.xml_location)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data

    def reset(self, mode='train'):
        if mode == 'train':
            self.pendulum_length = self.np_random.uniform(self.domain_low, self.domain_high)
            self.set_length(self.pendulum_length)
        elif self.mode != 'test' and mode == 'test': #starting adaptation
            self.pendulum_length = self.test_domain
            self.mode = mode
            self.set_length(self.test_domain)
        mujoco_env.MujocoEnv.reset(self)
        return self._get_obs()
