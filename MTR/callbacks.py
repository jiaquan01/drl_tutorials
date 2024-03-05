#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2023/10/03 15:46:21
@Author  :   qiang.luo@cloudminds.com 
@File    :   callbacks.py
'''
from typing import Dict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import Episode, RolloutWorker


class MtrCallbacks(DefaultCallbacks):
    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
        ):
        infos = episode._last_infos['agent0']
        # print(f'dist list is {infos["dist"]}')
        # print(f'end speed is {infos["speed"]}')
        # print(f'action list is {infos["action_list"]}')
        episode.custom_metrics["min_dist"] = min(infos["dist"])
        episode.custom_metrics["last_dist"] = infos["dist"][-1]
        episode.custom_metrics["end_speed"] = infos["speed"]
        episode.custom_metrics["succ_rate"] = sum(infos["succ_rate"]) / 500
