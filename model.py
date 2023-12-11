#add environment to PYTHONPATH
import sys
import os
env_path = os.path.join(os.path.abspath(os.getcwd()), '..\\Environments\\ContinuousCartPole')
sys.path.append(env_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

from collections import deque
from matplotlib import pyplot as plt

class Net(nn.Module):
    def __init__(self, imageSize, nClasses, hiddenSize = 120, channels = 3, discrete = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        size = self.ComputeFFSize(imageSize)

        self.fc1 = nn.Linear(size, hiddenSize)
        self.fc3 = nn.Linear(hiddenSize, nClasses)

        self.discrete = discrete
        self.sm = torch.nn.Softmax(dim = 1)

    def ComputeFFSize(self, imageSize):
        print("Input Size: ", imageSize)
        x = torch.zeros(imageSize)
        #x = torch.movedim(x, 3, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return x.size()[1]

    def forward(self, x):
        x_ref = torch.tensor(x).float()
        if len(x_ref.size()) < 4:
            x_ref = torch.unsqueeze(x_ref, 0)
        x_ref /= 255
        
        #x_ref = torch.movedim(x_ref, 3, 1)
        x_ref = self.pool(F.relu(self.conv1(x_ref)))
        x_ref = self.pool(F.relu(self.conv2(x_ref)))
        x_ref = torch.flatten(x_ref, 1) # flatten all dimensions except batch
        x_ref = F.relu(self.fc1(x_ref))
        x_ref = self.fc3(x_ref)
        if self.discrete:
            x_ref = self.sm(x_ref)
        return x_ref

class PolicyNetwork(nn.Module):
    
    #Takes in observations and outputs actions mu and sigma
    def __init__(self, observation_space, actions = 1, hidden = 64, discrete = False):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden)
        self.output_layer = nn.Linear(hidden, actions)
        self.actions = actions
        self.discrete = discrete
        self.sm = torch.nn.Softmax()
    
    #forward pass
    def forward(self, x):
        x_ref = torch.tensor(x).float()
        x_ref = self.input_layer(x_ref)
        x_ref = F.relu(x_ref)
        action_parameters = self.output_layer(x_ref)
        if self.discrete:
            action_parameters = self.sm(action_parameters)
        
        return action_parameters
    
class Agent():
    def __init__(self, network, discount):
        self.network = network
        self.discount = discount
        self.actions = network.actions

    def select_action(self, state_tensor):
        
        #create state tensor
        state_tensor.required_grad = True
        
        #forward pass through network
        action_parameters = self.network(state_tensor)
        
        #get mean and std, get normal distribution
        if self.actions == 1:
            mu, sigma = action_parameters[:, :1], torch.exp(action_parameters[:, 1:])
            m = Normal(mu[:, 0], sigma[:, 0])
        else:
            mus = action_parameters[:self.actions]
            sigmas = torch.exp(action_parameters[self.actions:])
            m = MultivariateNormal(mus, torch.diag(sigmas))

        #sample action, get log probability
        action = m.sample()
        log_action = m.log_prob(action)

        return action, log_action
    
    def process_rewards(self, rewards):
        ''' Converts our rewards history into cumulative discounted rewards
        Args:
        - rewards (Array): array of rewards 
        
        Returns:
        - G (Array): array of cumulative discounted rewards
        '''
        #Calculate Gt (cumulative discounted rewards)
        G = []
        
        #track cumulative reward
        total_r = 0
        
        #iterate rewards from Gt to G0
        for r in reversed(rewards):
            
            #Base case: G(T) = r(T)
            #Recursive: G(t) = r(t) + G(t+1)*DISCOUNT
            total_r = r + total_r * self.discount
            
            #add to front of G
            G.insert(0, total_r)
        
        #whitening rewards
        G = torch.tensor(G)
        G = (G - G.mean())/G.std()
        
        return G
    
def train(env, agent, optimizer, episodes = 2000, batchSize = 5):
    #track scores
    scores = []

    #track recent scores
    recent_scores = deque(maxlen=10)

    #iterate through episodes
    for episode in range(episodes):
        
        
        #reset environment, initiable variables
        rewards = []
        log_actions = []
        score = 0
        
        #generate episode
        for batch in range(batchSize):
            state = env.reset()[0]
            state = torch.tensor(state).float().to("cuda")
            while True:
                #select action, clip action to be [-1, 1]
                action, la = agent.select_action(state)
                action = torch.clamp(action, -1, 1)

                #execute action
                new_state, reward, _, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
                
                #track episode score
                score += reward
                
                #store reward and log probability
                rewards.append(reward)
                log_actions.append(la)
                
                #end episode
                if terminated or truncated:
                    break
                
                #move into new state
                state = torch.tensor(new_state).float().to("cuda")
            print("Outer Loop: ", episode, "Episode: ", episode*batchSize + batch, "Score: ", score/((batch + 1)), end = "\r")
        score /= batchSize
        #append score
        scores.append(score)
        recent_scores.append(score)

        #Calculate Gt (cumulative discounted rewards)
        rewards = agent.process_rewards(rewards)

        #adjusting policy parameters with gradient ascent
        loss = []
        for r, la in zip(rewards, log_actions):
            #we add a negative sign since network will perform gradient descent and we are doing gradient ascent with REINFORCE
            loss.append(-r * la)
            
        #Backpropagation
        optimizer.zero_grad()
        sum(loss).backward()
        optimizer.step()

        if episode%20 == 0 and episode != 0:
            plt.figure(dpi = 75)
            plt.plot(scores)
            plt.xlabel("Episodes")
            plt.ylabel("Score")
            plt.title("Training Performance Over Time")
            plt.savefig("images/results-2.png")

            torch.save({"network": agent.network.state_dict(), "optim": optimizer.state_dict()}, "models/Checkpoint-2.torch")

    env.close()



if __name__ == "__main__":
    pass