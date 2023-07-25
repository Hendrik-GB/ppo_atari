import gymnasium
import torch
import os

from PPO import PPO
from Network import CNN
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStack
from pathlib import Path
from torch.distributions import Categorical


def train(env):
    print(f"Training", flush=True)

    # Create a model for PPO
    model = PPO(env=env, action_space=action_space, game=game)
    model.learn(total_timesteps=200000000)


def test(env):
    print(f"Testing", flush=True)

    # path to saved model
    p = Path(os.getcwd()).parent.absolute()
    p = p / 'saved-models' / 'breakout_3488178.pt'
    # Pong_11000000.pt
    # breakout_3488178.pt

    device = torch.device('cpu')
    actor = CNN(out_dims=action_space)

    checkpoint = torch.load(p, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])

    done = False
    obs, _ = env.reset()
    timestep = 0

    while not done:
        timestep = timestep + 1
        obs = torch.unsqueeze(torch.Tensor(obs), dim=0)
        obs = torch.unsqueeze(obs, dim=0)
        logits = actor(obs.to(device)).cpu()
        dist = Categorical(logits=logits)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        obs, reward, done, _, _ = env.step(action)
        print('Timestep:', timestep)


mode = 'train' if torch.cuda.is_available() else 'test'
# game = "ALE/Pong-v5"
game = "ALE/Breakout-v5"
action_space = 4
num_envs = 6


def main():
    # Train or test, depending on the mode specified
    if mode == 'train':
        env = gymnasium.make(game, obs_type="rgb", frameskip=1)
        wrapped_env = AtariPreprocessing(env)
        wrapped_env = FrameStack(wrapped_env, 4)
        train(env=wrapped_env)
    elif mode == 'test':
        env = gymnasium.make(game, obs_type="rgb", frameskip=1, render_mode='human')
        wrapped_env = AtariPreprocessing(env)
        wrapped_env = FrameStack(wrapped_env, 4)
        test(env=wrapped_env)


if __name__ == '__main__':
    main()
