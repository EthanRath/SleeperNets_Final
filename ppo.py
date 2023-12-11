"""
Code modified from a tutorial by Eric Yu
	Base Code: https://github.com/ericyangyu/PPO-for-Beginners
	Tutorial: https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""


"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

#import gym
import time
import os

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt
from collections import deque

from torchvision.transforms.functional import rgb_to_grayscale
from Utilities import TensorToDataloader, DataloaderToTensor


class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, save_tag = "misc", discretize = None, middleman = None, bufferman = None, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		#assert(type(env.observation_space) == gym.spaces.Box)
		#assert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)
		self.discretize = discretize
		self.save_tag = save_tag
		self.middleman = middleman
		self.bufferman = bufferman

		try:
			os.mkdir("images/" + save_tag + "/")
		except:
			print("Image save Location Already Exists")
		try:
			os.mkdir("models/" + save_tag + "/")
		except:
			print("Model save Location Already Exists")

		# Extract environment information
		self.env = env
		try:
			if self.pixels:
				self.obs_dim = [1,4] + list(env.observation_space.shape)
			else:
				self.obs_dim = list(env.observation_space.shape)[0]
		except:
			self.obs_dim = env.observation_space.n
		print("Observation dim: ", self.obs_dim)

		self.discretize = lambda x : x
		if discretize != None:
			self.discretize = discretize
			self.act_dim = len(self.discretize)
		elif self.continuous:
			self.act_dim = env.action_space.shape[0]
		else:
			self.act_dim = env.action_space.n
		print("Action dim", self.act_dim)

		 # Initialize actor and critic networks
		if self.learned_var and self.continuous:
			dims = self.act_dim * 2
		else:
			dims = self.act_dim
		self.actor = policy_class(self.obs_dim, dims)  # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.25)
		self.cov_mat = torch.diag(self.cov_var)


		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			"poisoned": [],			# poisoned states
			"means": [],				# mean mean

			"plot_p": [],
			"plot_r": [],
			"wins": [],
			"benign": []
		}

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			with torch.no_grad():
				batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, poisoned_states = self.rollout()                     # ALG STEP 3

				# Calculate how many timesteps we collected this batch
				t_so_far += np.sum(batch_lens)

				# Increment the number of iterations
				i_so_far += 1

				# Logging timesteps so far and iterations so far
				self.logger['t_so_far'] = t_so_far
				self.logger['i_so_far'] = i_so_far

				# Calculate advantage at k-th iteration
				dataloader = TensorToDataloader(batch_obs.to("cuda"), batch_acts.to("cuda"), batchsize=self.bs, shuffle = False)
				A_k = torch.zeros(size = batch_rtgs.size())
				index = 0
				for xbatch, ybatch in dataloader:
					V, _ = self.evaluate(xbatch, ybatch)
					A_k[index : index + len(xbatch)] = batch_rtgs[index : index + len(xbatch)] - V.detach()   
					index += len(xbatch)

				#if self.bufferman!=None:
				#	scores = self.evaluate(batch_obs[poisoned_states], batch_acts[poisoned_states], True)
				#	self.logger["plot_p"].append(torch.mean(scores))

				# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
				# isn't theoretically necessary, but in practice it decreases the variance of 
				# our advantages and makes convergence much more stable and faster. I added this because
				# solving some environments was too unstable without it.
				A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			
			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration): # ALG STEP 6 & 7
				index = 0
				for xbatch, ybatch in dataloader:
					#print(len(xbatch))
					# Calculate V_phi and pi_theta(a_t | s_t)
					V, curr_log_probs = self.evaluate(xbatch, ybatch)

					# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
					# NOTE: we just subtract the logs, which is the same as
					# dividing the values and then canceling the log with e^log.
					# For why we use log probabilities instead of actual probabilities,
					# here's a great explanation: 
					# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
					# TL;DR makes gradient ascent easier behind the scenes.
					ratios = torch.exp(curr_log_probs - batch_log_probs[index : index + len(xbatch)])

					# Calculate surrogate losses.
					surr1 = ratios * A_k[index : index + len(xbatch)]
					surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k[index : index + len(xbatch)]

					# Calculate actor and critic losses.
					# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
					# the performance function, but Adam minimizes the loss. So minimizing the negative
					# performance function maximizes it.
					actor_loss = (-torch.min(surr1, surr2)).mean()
					critic_loss = nn.MSELoss()(V, batch_rtgs[index : index + len(xbatch)])

					# Calculate gradients and perform backward propagation for actor network
					self.actor_optim.zero_grad()
					actor_loss.backward(retain_graph=True)
					self.actor_optim.step()

					# Calculate gradients and perform backward propagation for critic network
					self.critic_optim.zero_grad()
					critic_loss.backward()
					self.critic_optim.step()

					# Log actor loss
					self.logger['actor_losses'].append(actor_loss.detach())
					index += len(xbatch)

			# Print a summary of our training so far
			with torch.no_grad():
				score = self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), 'models/' + self.save_tag + '/ppo_actor_.pth')
				torch.save(self.critic.state_dict(), 'models/' + self.save_tag + '/ppo_critic_.pth')
			if score > self.best:
				self.best = score
				torch.save(self.actor.state_dict(), 'models/' + self.save_tag + '/best_ppo_actor_.pth')
				torch.save(self.critic.state_dict(), 'models/' + self.save_tag + '/best_ppo_critic_.pth')

	def rollout(self, plot = False, log = True, poison = True):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
				poisoned_states - boolean string of states that were poisoned
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []
		poisoned_states = []
		wins = []
		frames = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []
		

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode
			ben_ep_rew = []
			ben_rew = []

			# Reset the environment. sNote that obs is short for observation. 
			obs, info = self.env.reset()
			action = []
			done = False
			if self.pixels:
				buffer = deque(maxlen = 4)
				for i in range(3):
					buffer.append(torch.tensor(obs))


			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
				if self.render:
					frames.append(self.env.render())

				if self.pixels:
					buffer.append(torch.tensor(obs))
					obs = torch.stack(list(buffer), 0)

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
				prev_act = action
				action, log_prob = self.get_action(obs)
				if self.safety:
					obs, rew, _, terminated, truncated, _ = self.env.step(self.discretize(action))
				else:
					obs, rew, terminated, truncated, _ = self.env.step(self.discretize(action))
				ben_rew.append(rew)

				done = terminated or truncated

				#Poison
				if self.middleman != None:
					obs, rew, poisoned = self.middleman(obs, action, rew, prev_act)
				else:
					poisoned = False
				
				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				poisoned_states.append(poisoned)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
				if done:
					wins.append(1 if terminated else 0)
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)
			ben_ep_rew.append(ben_rew)
			t += 1 # Increment timesteps ran this batch so far


		# Reshape data as tensors in the shape specified in function description, before returning
		if not self.pixels:
			batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		else:
			batch_obs = torch.stack(batch_obs, 0).float()
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		if log:
			self.logger['batch_rews'] = batch_rews
			self.logger['batch_lens'] = batch_lens
			self.logger["poisoned"] = poisoned_states
			self.logger["means"] = batch_acts
			self.logger["wins"] = wins
			self.logger["benign"] = ben_ep_rew

		if self.bufferman!=None and poison:
			temp = []
			for elm in batch_rews:
				temp += elm
			batch_obs, temp_rew, indices = self.bufferman(batch_obs, batch_acts, torch.tensor(temp), batch_rtgs)
			index = 0
			for i in range(len(batch_rews)):
				batch_rews[i] = list(temp_rew[index:index + len(batch_rews[i])])
				index += len(batch_rews[i])
			batch_rtgs[indices] = self.compute_rtgs(batch_rews)[indices]
			poisoned_states = torch.tensor(poisoned_states).long()
			poisoned_states[indices] += 1
			poisoned_states = poisoned_states.bool()

		if plot: return frames, batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, poisoned_states
		
		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, poisoned_states

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs
	
	def get_action(self, obs, multi = False):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		if self.continuous:
			mean = self.actor(obs)
			if self.learned_var:
				split = len(mean)//2
				covmat = torch.absolute(torch.diag(mean[split:]))
				dist = MultivariateNormal(mean[:split], covmat)
			else:
				self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.25)
				self.cov_mat = torch.diag(self.cov_var)
				dist = MultivariateNormal(mean, self.cov_mat)
			# Sample an action from the distribution
			action = dist.sample()
			# Calculate the log probability for that action
			log_prob = dist.log_prob(action)
			# Return the sampled action and the log probability of that action in our distribution
			return action.detach().numpy(), log_prob.detach()
		
		#In case of discrete actions
		else:
			probs = self.actor(obs)
			dist = Categorical(probs)
			action = dist.sample()
			log_prob_action = dist.log_prob(action)
			if multi:
				return action.detach(), log_prob_action.detach()
			return action.detach().item(), log_prob_action.detach()

	def evaluate(self, batch_obs, batch_acts, poison = False):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		if self.continuous:
			mean = self.actor(batch_obs)
			if self.learned_var:
				split = len(mean[0])//2
				covmat = torch.absolute(torch.diag_embed(mean[:, split:]))
				dist = MultivariateNormal(mean[:, :split], covmat)
			else:
				dist = MultivariateNormal(mean, self.cov_mat)
			log_probs = dist.log_prob(batch_acts)
		else:
			probs = self.actor(batch_obs)
			dist = Categorical(probs)
			log_probs = dist.log_prob(batch_acts)

		if poison:
			actions = dist.sample()
			score = self.bufferman.dist(self.bufferman.target, actions)
			return score 
		
		return V, log_probs 

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results
		self.continuous = True
		self.learned_var = True
		self.safety = False
		self.pixels = False
		self.bs = 256
		self.best = -np.inf

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['benign']])
		avg_actor_loss = torch.mean(torch.tensor([losses.float().mean() for losses in self.logger['actor_losses']]))

		#print(self.logger["poisoned"])
		poisoned = self.logger["poisoned"]
		means = torch.tensor(self.logger["means"])
		rews = []
		for i in range(len(self.logger["batch_rews"])):
			rews =  rews + self.logger['batch_rews'][i]
		poisoned = torch.tensor(poisoned).bool()
		rews = torch.tensor(rews)

		mean_poisoned = (torch.mean( means[poisoned], 0 ), torch.std( means[poisoned], 0 ), 
				   		torch.mean(means[(1-poisoned.long()).bool()], 0) , torch.std(means[[(1-poisoned.long()).bool()]], 0))
		p_rew = torch.mean(rews[poisoned.bool()].float())
		p_rate = torch.mean(poisoned.float())

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		#avg_actor_loss = str(round(avg_actor_loss, 5))
		wins = torch.tensor(self.logger["wins"]).float()


		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Win Rate: {torch.mean(wins)}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Poisoned Return: {p_rew}", flush=True)
		print(f"Average Action Mean: {mean_poisoned}", flush=True)
		print(f"Poison Rate: {p_rate}", flush=True)
		print(f"Best So Far: {self.best}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
		self.logger["poisoned"] = []
		self.logger["means"] = []
		self.logger["wins"] = []

		if self.bufferman== None:
			if p_rate == 0:
				if len(self.logger["plot_p"]) > 0:
					p_rew = self.logger["plot_p"][-1]
				else: p_rew = 0
			self.logger["plot_p"].append(p_rew)
		self.logger["plot_r"].append(float(avg_ep_rews))
		
		#print(self.logger["plot_p"])
		#print(self.logger["plot_r"])

		if len(self.logger["plot_r"]) % 20 == 0:
			unflat = torch.nn.Unflatten(0, [len(self.logger["plot_r"])//5, 5])
			if self.bufferman!=None:
				self.plot_hist()
				plot_p = np.array(self.logger["plot_p"])
			else:
				plot_p = torch.mean( unflat(torch.tensor(self.logger["plot_p"])).float(), 1 ).cpu().numpy()

			plot_r = torch.mean( unflat(torch.tensor(self.logger["plot_r"])).float(), 1 ).cpu().numpy()
			plt.figure(dpi = 100)
			plt.plot(plot_p, label = "Poisoned Return")
			plt.title("Poisoning Return")
			plt.xlabel("Iterations")
			plt.ylabel("Return")
			plt.savefig("images/" + self.save_tag + "/poison.png")
			plt.close()

			plt.figure(dpi = 100)
			plt.plot(plot_r, label = "Overal Return")
			plt.title("Benign Return")
			plt.xlabel("Iterations")
			plt.ylabel("Return")
			plt.savefig("images/" + self.save_tag + "/overall.png")

			torch.save(plot_r, "images/" + self.save_tag + "/overall_tensor.torch")
			torch.save(plot_p, "images/" + self.save_tag + "/poisoned_tensor.torch")
			plt.close()

			return plot_r[-1]
		return -np.inf

	def plot_hist(self):
		batch_obs, batch_acts, _, _, _, _ = self.rollout(log = False, poison=False)
		poisoned = self.bufferman.trigger(batch_obs)

		p_act, _ = self.get_action(poisoned, True)
		asr = torch.mean((p_act == self.bufferman.target).float())
		self.logger["plot_p"].append(asr.item())

		batch_acts = batch_acts.long().cpu().numpy()
		p_act = p_act.long().cpu().numpy()

		plt.figure(dpi = 150)
		plt.hist(batch_acts, label = "Benign", alpha = .5)
		plt.hist(p_act, label = "Poisoned", alpha = .5)
		plt.legend()
		plt.title("Histogram of Poisoned and Benign Actions")
		plt.ylabel("Frequency")
		plt.xlabel("Action")
		plt.savefig("images/" + self.save_tag + "/histogram.png")
		plt.close()




if __name__ == "__main__":

	dist = Normal(0, .25)
	values = torch.arange(-1, 1, .01)
	log = dist.log_prob(values)
	log -= torch.max(log)
	log += 1
	log[log < -1] = -1

	plt.plot(values, log)
	plt.show()

