import gymnasium
import torch

from PPO import PPO
from Network import CNN
from gymnasium.wrappers import AtariPreprocessing


def train(env):
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(env=env)
    model.learn(total_timesteps=2000000)


def test(env):
    print(f"Testing", flush=True)

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    # policy = CNN(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    # policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    # eval_policy(policy=policy, env=env, render=True)


mode = 'train'


def main():
    env = gymnasium.make("ALE/Breakout-v5", obs_type="rgb", frameskip=1)
    env = AtariPreprocessing(env)

    # Train or test, depending on the mode specified
    if mode == 'train':
        train(env=env)
    else:
        test(env=env)


if __name__ == '__main__':
    main()
