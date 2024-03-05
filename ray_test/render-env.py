import gymnasium as gym
from ray.rllib.algorithms.dqn import DQNConfig

algo = DQNConfig().environment("LunarLander-v2").build()

for i in range(10):
    result = algo.train()
    print("Iteration:", i)
    print("Episode reward max:", result["episode_reward_max"])
    print("Episode reward min:", result["episode_reward_min"])
    print("Episode reward mean:", result["episode_reward_mean"])
    print()

env = gym.make("LunarLander-v2", render_mode="human")
terminated = truncated = False
observations, info = env.reset()

while True:
    env.render()
    action = algo.compute_single_action(observations)
    observations, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observations, info = env.reset()