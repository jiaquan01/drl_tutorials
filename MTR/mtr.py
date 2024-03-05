import os
import numpy as np

import mujoco
import gymnasium as gym
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

class DMEnv(MujocoEnv):
    metadata = {
        "render_modes": [ "human", "rgb_array", "depth_array",],
        "render_fps": 1000,
    }

    def __init__(self, render_mode=None):
        path_to_xml_out = os.path.dirname(os.path.abspath(__file__))+ "/motor.xml"
        self.actor_obs_dim = 1
        obs_space = gym.spaces.Box(-100, 100, (self.actor_obs_dim,))
        MujocoEnv.__init__(self, path_to_xml_out, 1, obs_space, render_mode=render_mode)
        self.MASS = self.model.body_mass.copy()

    def step(self, tau):
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data, 1)
        return [self.data.qpos - self.init_qpos, self.data.qvel, self.data.qacc]

    def reset_model(self):

        self.set_state(np.random.uniform(-3.14, 3.14, (1)), np.zeros(self.model.nv))
        self.data.ctrl[:] = np.random.uniform(-10, 10, (1))
        mujoco.mj_step(self.model, self.data, 1)
        self.env_steps = 0
        self.init_qpos = self.data.qpos.copy()

        return [self.data.qpos - self.init_qpos, self.data.qvel, self.data.qacc, self.data.ctrl.copy()]


if __name__ == "__main__":
    env = DMEnv("human")
    max_step = 30
    tau = 200
    
    for mass in (0.1, 100):
        print(f'================{mass}==================')
        env.reset()
        env.model.body_mass[3] = mass
        for i in range(max_step):
            pos, vel, acc = env.step(tau)
            print(f'step{i}, mass is {mass}, used tau is {tau}, pos is {pos.tolist()}, vel is {vel.tolist()}, acc is {acc.tolist()}')
