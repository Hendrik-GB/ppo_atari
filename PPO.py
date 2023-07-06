from Network import CNN

import torch
from torch.distributions import MultivariateNormal

import numpy as np

from einops import rearrange


# https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea58146
# https://github.com/ericyangyu/PPO-for-Beginners
class PPO:
    def __init__(self, env):
        self.env = env
        action_space = 4
        self.actor = CNN(out_dims=action_space)
        self.critic = CNN(out_dims=1)
        self._init_hyperparameters()
        self.cov_var = torch.full(size=(action_space,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800  # timesteps per batch
        self.max_timesteps_per_episode = 1600  # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def get_action(self, obs):
        obs = torch.unsqueeze(torch.Tensor(obs), dim=0)
        obs = torch.unsqueeze(obs, dim=0)
        # obs = rearrange(obs, 'b h w c -> b c h w')

        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)  # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        batch_obs = batch_obs.unsqueeze(dim=1)
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)  # Return predicted values V and log probs log_probs
        return V, log_probs

    def rollout(self):
        batch_obs = []  # batch observations
        batch_acts = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rewards = []  # batch rewards
        batch_rtgs = []  # batch rewards-to-go
        batch_lengths = []  # episodic lengths in batch

        t = 0
        while t < self.timesteps_per_batch:
            ep_rewards = []
            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t = t + 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, reward, done, _, _ = self.env.step(np.argmax(action))

                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_rewards.append(reward)

                if done:
                    break

            batch_lengths.append(ep_t + 1)
            batch_rewards.append(ep_rewards)

        batch_obs = torch.tensor(np.asarray(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.asarray(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.asarray(batch_log_probs), dtype=torch.float)  # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rewards)  # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lengths

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def learn(self, total_timesteps):
        t_so_far = 0

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t_so_far += np.sum(batch_lens)
            print('Learned Timesteps:', t_so_far, 'With Ratings:', batch_rtgs)

            v, _ = self.evaluate(batch_obs, batch_acts)
            a_k = batch_rtgs - v.detach()
            a_k = (a_k - a_k.mean()) / (a_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                _, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * a_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * a_k
                actor_loss = (-torch.min(surr1, surr2)).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
