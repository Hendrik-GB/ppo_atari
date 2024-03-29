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
        self.network = CNN(out_dims=self.action_space).to(device)
        self._init_hyperparameters()
        self.done = True
        self.last_obs = None

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=0.1)

    def _init_hyperparameters(self):
        self.rollout_steps = 600  # timesteps per episode
        self.gamma = 0.97
        self.n_updates_per_iteration = 20
        self.ppo_clip = 0.1
        self.lr = 0.00025
        self.critic_coefficient = 0.5

    def get_action(self, obs):
        with torch.no_grad():
            obs = torch.Tensor(np.array(obs))
            obs = obs.unsqueeze(dim=0)
            logits = self.network.action_only(obs.to(device)).cpu()
            dist = Categorical(logits=logits)

            # Sample an action from the distribution and get its log prob
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action[0], log_prob

    # calculate predicted score and probability of actions
    def evaluate(self, batch_obs, batch_acts):
        # print('evaluate', batch_obs.shape, batch_acts.shape)

        # calculate predicted score and log probs
        logits, V = self.network.action_score(batch_obs.to(device))
        V, logits = V.cpu(), logits.cpu()

        if torch.sum(torch.isnan(logits)) != 0:
            print('Nan detected')
            print(batch_obs.shape, batch_acts.shape)
            print(batch_obs, batch_acts)

            p = Path(os.getcwd()).parent.absolute()
            p = p / 'saved-models' / 'ppo_atari' / \
                (self.game.split('-')[0].split('/')[-1] + '_nan.pt')

            torch.save({
                'steps': -1,
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, p)
            print('Saved model at nan time steps')

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

            with torch.no_grad():
                V, _ = self.evaluate(batch_obs, batch_acts)

            t_so_far += batch_length

            if iteration % 5 == 0:
                print('Learned Timesteps:', t_so_far, 'With last Rating:', batch_ratings[0],
                      'Action Distribution:', np.histogram(batch_acts, bins=np.arange(self.action_space + 1))[0])

            # calculate advantage estimates
            # hier wurzel des nan problems -> wenn keine rewards in batch sind und V konstant dann a_k = nan und fehler
            # in backwards pass -> nan in conv2d filter
            a_k = batch_ratings - V
            a_k = (a_k - a_k.mean()) / (a_k.std() + 1e-8)

            # update net n times
            # with torch.autograd.detect_anomaly():
            for _ in range(self.n_updates_per_iteration):
                # calculate clipped loss for actor
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * a_k
                surr2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * a_k
                actor_loss = (-torch.min(surr1, surr2)).mean()

                critic_loss = ((V - batch_ratings) ** 2).mean()

                loss = actor_loss + self.critic_coefficient * critic_loss

                # optimize nets
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
                self.optimizer.step()

            # save model
            if iteration % 500 == 0:
                # path to data folder
                p = Path(os.getcwd()).parent.absolute()
                p = p / 'saved-models' / 'ppo_atari' / \
                    (self.game.split('-')[0].split('/')[-1] + '_' + str(t_so_far) + '.pt')

                torch.save({
                    'steps': t_so_far,
                    'network': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, p)
                print('Saved model at', t_so_far, 'time steps')
