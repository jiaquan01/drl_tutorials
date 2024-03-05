from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import gymnasium as gym

config = ( # 1. Configure the algorithm,
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="CartPole-v1")
    
)

algo=config.build()

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")



env = gym.make("CartPole-v1", render_mode="human")
terminated = truncated = False
observations, info = env.reset()

while True:
    env.render()
    action = algo.compute_single_action(observations)
    observations, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observations, info = env.reset()