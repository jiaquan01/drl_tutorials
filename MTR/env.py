#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2023/10/02 13:59:47
@Author  :   qiang.luo@cloudminds.com 
@File    :   env.py
'''
import gymnasium as gym
import numpy as np
import time

from collections import deque

from mtr import DMEnv

class Env(gym.Env):
    """class env""" 
    def __init__(self, env_config):
        super(Env, self).__init__()
        
        # 定义obs和action空间
        self.observation_space = gym.spaces.Box(-1, 1, (5, ), dtype=np.float32)  # 关节转动
        self.action_space = gym.spaces.Box(-1, 1, (1, ), dtype=np.float32)  # 关节转动
        
        # 初始化mujuco环境
        self.sim = DMEnv()

        self.time_step = 0
        self.config = env_config

        self.target_range = self.config['target_range']
        self.max_torque = self.config['max_torque']
        self.max_vel = self.max_torque / 5
        self.max_acc = self.max_torque * 200

        self.target_index = 1

        print('target_range: ', self.target_range)
        print('max_torque: ', self.max_torque)
        print('max_vel: ', self.max_vel)
        print('max_acc: ', self.max_acc)
        self.rg = [
            # target pose
            [-self.target_range, self.target_range],
            # current pose
            [-self.target_range, self.target_range],
            # current velocity
            [-self.max_vel, self.max_vel],
            # current acceleration
            [-self.max_acc, self.max_acc],
            # pre torque
            [-self.max_torque, self.max_torque],
            # step
            # [1, self.config['max_steps']],
        ]

        self.infos = {
            'succ_rate': deque(maxlen=500),
        }

        # self.obs = None


    def reset(self, *, seed=None, options=None):
        # mojuco reset
        self.sim._reset_simulation()
        self.sim.model.body_mass[3] = np.power(10, np.random.uniform(-1, 2))
        raw_obs, _ = self.sim.reset()
         
        # while True:
        # target = -0.1*(1-np.cos(2*np.pi * self.target_index/100))
        target = np.random.uniform(-self.target_range, self.target_range)
            # if abs(target) > self.config['dist_threshold']:
            #     break
        self.target_index += 1
        
        # task reset
        self.infos['target'] = target
        self.infos['mass'] = self.sim.model.body_mass[3]
        self.infos['dist'] = []
        self.infos['speed'] = 100
        self.infos['action_list'] = []

        # print('mass is', self.sim.model.body_mass[3])

        self.setPoint = target
        self.time_step = 1
        self.infos['dist'].append(abs(target))

        # obs = [self.setPoint] + raw_obs + [0] + [1]
        obs = [self.setPoint] + raw_obs + [0]

        self.obs = obs[:] + [self.sim.model.body_mass[3]]
        model_obs = np.array([self.normalize(obs[i], self.rg[i]) for i in range(len(self.rg))])
        return model_obs, self.infos

    def step(self, action):
        tau = action * self.max_torque
        raw_obs = self.sim.step(tau)
        self.infos['action_list'].append(float(tau))
        # obs = [self.setPoint] + raw_obs + [tau] + [self.time_step+1]
        obs = [self.setPoint] + raw_obs + [tau]

        dist = abs(float(self.adjust_radian(raw_obs[0] - self.setPoint)))
        self.infos['dist'].append(dist)
        done, succ = False, False
        if self.time_step == self.config['max_steps']:
            done = True
            succ = self.infos['dist'][-1] < self.config['dist_threshold'] and abs(obs[1]) < self.config['speed_threshold']
            self.infos['speed'] = abs(raw_obs[1])
            self.infos['succ_rate'].append(int(succ))
        reward = self.get_reward(done, succ)
        # print('obs is:', obs)
        self.obs = obs[:] + [self.sim.model.body_mass[3]]
        # print(f'mass is {self.infos["mass"]}, ac is {tau/raw_obs[-1]}')
        model_obs = np.array([self.normalize(obs[i], self.rg[i]) for i in range(len(self.rg))])
        self.time_step += 1
        return model_obs, reward, done, False, self.infos

    def close(self):
        pass

    def render(self, mode="human"):
        pass
    
    def get_reward(self, done, succ):
        # 成功奖励
        if succ:
            reward = 100
        # 失败惩罚
        elif done:
            reward = -10
        # 距离奖励
        else:
            reward = self.infos['dist'][-2] - self.infos['dist'][-1]
            reward *= 1000 if reward > 0 else 1500
        return reward

    def judge_done(self, obs):
        done = False
        succ = False
        # 超时
        if self.time_step == self.config['max_steps']:
            done = True
        # 成功
        if self.infos['dist'][-1] < self.config['dist_threshold'] \
            and abs(obs[1]) < self.config['speed_threshold']:
            done = True
            succ = True
        return done, succ
        
    @staticmethod
    def adjust_radian(radian):
        if radian < -np.pi:
            radian += 2 * np.pi
        elif radian > np.pi:
            radian -= 2 * np.pi
        return radian
    
    # 根据range归一化到[-1, 1]
    @staticmethod
    def normalize(numb, range):
        numb = float(np.clip(numb, range[0], range[1]))
        return (numb - range[0]) / (range[1] - range[0]) * 2 - 1

if __name__ == '__main__':
    pass
