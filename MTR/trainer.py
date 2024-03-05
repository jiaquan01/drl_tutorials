#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2023/10/08 14:12:12
@Author  :   qiang.luo@cloudminds.com 
@File    :   trainer.py
'''
from ray.rllib.agents.ppo import PPOTrainer


class MTRPPOTrainer(PPOTrainer):

    def save_checkpoint(self, tmp_checkpoint_dir):

        policy = self.get_policy()
        policy.export_model(tmp_checkpoint_dir)

        return super(MTRPPOTrainer, self).save_checkpoint(tmp_checkpoint_dir)
