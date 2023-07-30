import torch
import os

from Network import CNN
from torch.distributions import Categorical
from pathlib import Path

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea58146
# https://github.com/ericyangyu/PPO-for-Beginners
class PPO:
    def __init__(self, env, action_space, game):
        self.env = env
        self.action_space = action_space
        self.game = game
        self.actor = CNN(out_dims=self.action_space).to(device)
        self.critic = CNN(out_dims=1).to(device)
        self._init_hyperparameters()
        self.done = True
        self.last_obs = None

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self):
        self.rollout_steps = 250  # timesteps per episode
        self.gamma = 0.99
        self.n_updates_per_iteration = 20
        self.clip = 0.2
        self.lr = 0.00025

    def get_action(self, obs):
        obs = torch.Tensor(np.array(obs))
        logits = self.actor(obs.to(device)).cpu()
        dist = Categorical(logits=logits)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach()[0], log_prob.detach()

    # calculate predicted score and probability of actions
    def evaluate(self, batch_obs, batch_acts):
        # print('evaluate', batch_obs.shape, batch_acts.shape)

        # calculate predicted score with critic
        V = self.critic(batch_obs.to(device)).squeeze().cpu()

        # calculate log prob of actions
        logits = self.actor(batch_obs.to(device)).cpu()
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)

        # Return predicted values V and log probs
        return V, log_probs

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        ep_rewards = []

        obs = self.last_obs

        if self.done:
            obs, _ = self.env.reset()
            self.done = False

        for ep_t in range(self.rollout_steps):

            batch_obs.append(obs)

            # generate action and next observation
            action, log_prob = self.get_action(obs)
            obs, reward, done, _, _ = self.env.step(action)

            batch_acts.append(action)
            batch_log_probs.append(log_prob)
            ep_rewards.append(reward)

            if done:
                self.done = True
                break

        batch_length = ep_t + 1
        batch_rewards = ep_rewards

        self.last_obs = obs

        batch_obs = torch.tensor(np.asarray(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.asarray(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rewards)  # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_length

    # calculate ratings for given batch
    def compute_rtgs(self, rewards):
        ratings = []
        discounted_reward = 0

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each discounted return
        for rew in reversed(rewards):
            discounted_reward = rew + discounted_reward * self.gamma
            ratings.insert(0, discounted_reward)

        ratings = torch.tensor(ratings, dtype=torch.float)
        return ratings

    # main learning method
    def learn(self, total_timesteps):
        t_so_far = 0
        iteration = 0

        while t_so_far < total_timesteps:
            iteration = iteration + 1
            batch_obs, batch_acts, batch_log_probs, batch_ratings, batch_length = self.rollout()
            v, _ = self.evaluate(batch_obs, batch_acts)

            t_so_far += batch_length

            if t_so_far % 5000 == 0:
                print('Learned Timesteps:', t_so_far, 'With last Rating:', batch_ratings[0],
                      'Action Distribution:', np.histogram(batch_acts, bins=np.arange(self.action_space))[0])

            # calculate advantage estimates
            a_k = batch_ratings - v.detach()
            a_k = (a_k - a_k.mean()) / (a_k.std() + 1e-10)

            # update net n times
            for _ in range(self.n_updates_per_iteration):

                # calculate clipped loss for actor
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * a_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * a_k
                actor_loss = (-torch.min(surr1, surr2)).mean()

                # optimize nets
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                critic_loss = torch.nn.MSELoss()(V, batch_ratings)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            # save model
            if iteration % 500 == 0:
                # path to data folder
                p = Path(os.getcwd()).parent.absolute()
                p = p / 'saved-models' / 'ppo_atari' / \
                    (self.game.split('-')[0].split('/')[-1] + '_' + str(t_so_far) + '.pt')

                torch.save({
                    'steps': t_so_far,
                    'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optim.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                }, p)
                print('Saved model at', t_so_far, 'time steps')
