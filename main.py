import gymnasium
import torch
import os

from PPO import PPO
from Network import CNN
from gymnasium.wrappers import AtariPreprocessing
from pathlib import Path
from torch.distributions import Categorical


def train(env):
    print(f"Training", flush=True)

    # Create a model for PPO
    model = PPO(env=env)
    model.learn(total_timesteps=2000000)


def test(env):
    print(f"Testing", flush=True)

    # path to saved model
    p = Path(os.getcwd()).parent.absolute()
    p = p / 'saved-models' / 'breakout_1523103.pt'

    device = torch.device('cpu')
    actor = CNN(out_dims=4)

    checkpoint = torch.load(p, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])

    done = False
    obs, _ = env.reset()

    while not done:
        obs = torch.unsqueeze(torch.Tensor(obs), dim=0)
        obs = torch.unsqueeze(obs, dim=0)
        logits = actor(obs.to(device)).cpu()
        dist = Categorical(logits=logits)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        obs, reward, done, _, _ = env.step(action)
        print(reward)


mode = 'train' if torch.cuda.is_available() else 'test'
game = 'ALE/Breakout-v5'


def main():
    # Train or test, depending on the mode specified
    if mode == 'train':
        env = gymnasium.make(game, obs_type="rgb", frameskip=1)
        env = AtariPreprocessing(env)
        train(env=env)
    elif mode == 'test':
        env = gymnasium.make(game, obs_type="rgb", frameskip=1, render_mode='human')
        env = AtariPreprocessing(env)
        test(env=env)


if __name__ == '__main__':
    main()
