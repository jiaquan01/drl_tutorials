#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    :   2023/10/02 14:00:03
@Author  :   qiang.luo@cloudminds.com 
@File    :   run_train.py
'''
import argparse

import ray
from ray import tune

from env import Env
from callbacks import MtrCallbacks
from trainer import MTRPPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--target_range', type=float, default=2e-2)
parser.add_argument('--dist_threshold', type=float, default=1e-3)
parser.add_argument('--speed_threshold', type=float, default=2e-2)
parser.add_argument('--max_torque', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--grad_clip', type=float, default=1)
parser.add_argument('--use_lstm', action="store_true", default=False)
parser.add_argument('--num_workers', type=int, default=20)

def run():
    args = parser.parse_args()
    
    args.checkpoint_freq = 1
    # args.model_path = '/home/luoqiang/projects/xr4_mtr/ppo_mtr/MTRPPOTrainer_Env_838e8_00000_0_2023-10-18_16-37-04/checkpoint_002000'
    args.model_path = None

    ray.init()

    config = {
        # === Environment ===
        'env':Env,
        'env_config':{
            'target_range': args.target_range,
            'dist_threshold': args.dist_threshold,
            'speed_threshold': args.speed_threshold,
            'max_torque': args.max_torque,
            'max_steps': 20,
        },
        'horizon':20,

        # === Pollcy ===
        'callbacks': MtrCallbacks,
        'log_level': 'INFO',
                                                                     
        # === RolloutWorker === 
        'num_workers': args.num_workers,
        'rollout_fragment_length': 200,
        'num_cpus_per_worker': 0.3,
        'num_gpus':1,
        
        # === Model ===
        'framework':'torch',
        'model':{
            'use_lstm': args.use_lstm,
            'fcnet_hiddens':[128, 256, 256],
            'fcnet_activation':'relu',
        },

        # === PPO ===
        'train_batch_size':20000,
        'sgd_minibatch_size':2048,
        'num_sgd_iter':10,
        'lr':args.lr,
        'lambda':0.95,
        'gamma':0.99,
        'clip_param':0.2,  
        'grad_clip': args.grad_clip,
        'entropy_coeff':0.0,
    }

    exp_name = "ppo_mtr"

    tune.run(MTRPPOTrainer,
             config=config,
             stop={"timesteps_total": 60000000},
             local_dir=args.log_dir,
             checkpoint_at_end=True,
             checkpoint_score_attr="custom_metrics/succ_rate_mean",
             keep_checkpoints_num=3,
             checkpoint_freq=args.checkpoint_freq,
             restore=args.model_path,
             name=exp_name)
    ray.shutdown()


if __name__ == "__main__":
    run()
