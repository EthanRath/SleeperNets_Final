import torch
import numpy as np
import copy
from torch.distributions.normal import Normal

class MiddleMan:
    def __init__(self, trigger, target, dist, p_steps = 10, p_rate = .01):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.p_rate = p_rate
        self.p_steps = p_steps
        self.steps = 1

        self.min = 1
        self.max = -1
        
    def __call__(self, state, action, reward, prev_act):
        with torch.no_grad():
            if self.steps > 1 or torch.rand(1) <= self.p_rate :
                poisoned = self.trigger(state)
                reward_p = self.dist(self.target, action)
                self.steps = self.steps + 1 if self.steps < self.p_steps else 1
                return poisoned, reward_p, True
            return state, reward, False
        
class DeterministicMiddleMan:
    def __init__(self, trigger, target, dist, total, budget):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.budget = budget
        self.index = int(total/budget)
        self.steps = 0
        
    def __call__(self, state, action, reward, prev_act):
        with torch.no_grad():
            self.steps += 1
            if self.steps%self.index == 0:
                poisoned = self.trigger(state)
                reward_p = self.dist(self.target, action)
                return poisoned, reward_p, True
            return state, reward, False
	
class BufferMan_Simple:
    def __init__(self, trigger, target, dist, p_steps = 10, p_rate = .01):
        self.trigger = trigger
        self.target = target
        self.dist = dist
        self.p_steps = p_steps
        self.p_rate = p_rate
    def __call__(self, states, actions, rewards, rtg):
        probs = np.ones(len(states))/len(states)
        indices = np.random.choice(np.arange(0, len(states), 1), int(len(states)*self.p_rate), replace = False, p = probs)
        indices = torch.tensor(indices).long()
        states[indices] = self.trigger(states[indices])
        rewards[indices] = self.dist(self.target, actions[indices])
        return states, rewards, indices

class BufferMan_Weighted:
    def __init__(self, trigger, target, dist, p_steps = 10, p_rate = .01):
        self.trigger = trigger
        self.target = target
        self.dist = dist
        self.p_steps = p_steps
        self.p_rate = p_rate
    def __call__(self, states, actions, rewards, rtg):
        probs = torch.zeros(len(states))
        correct = actions == self.target
        #wrong = ~correct #(1-correct.long()).bool()

        # more strongly weigh high rtg instances with the correct action, 
        # and low rtg instances with the wrong action
        if torch.sum(correct.long()) > 0:
            pos = rtg[correct]
            #neg = -1 * rtg[wrong]
            temp_pos = (pos - torch.min(pos)) / (torch.max(pos) - torch.min(pos))
            #temp_neg = (neg - torch.min(neg)) / (torch.max(neg) - torch.min(neg))
            temp_pos[torch.isnan(temp_pos)] = 0
            #temp_neg[torch.isnan(temp_neg)] = 0
            #temp_neg *= (torch.max(neg) / torch.max(pos))

            probs[correct] = temp_pos
        #probs[wrong] = temp_neg
        #temp = (rtg - torch.min(rtg[correct])) / (torch.max(rtg[correct]) - torch.min(rtg[correct]))
        #probs *= temp
        probs += .1
        probs /= torch.sum(probs)
        probs = probs.cpu().numpy()


        indices = np.random.choice(np.arange(0, len(states), 1), int(len(states)*self.p_rate), replace = False, p = probs)
        indices = torch.tensor(indices).long()
        states[indices] = self.trigger(states[indices])
        rewards[indices] = self.dist(self.target, actions[indices])
        return states, rewards, indices

#replaces a single observation within some pre-set indices into a given value
#  we can alter this to use a random value over some distribution for furhter robustness
class SingleValuePoison:

    def __init__(self, indices, value, numpy = False):
        self.indices = indices
        self.value = value
        self.numpy = numpy

    def __call__(self, state):
        #if len(self.indices)>1:
        #    index = np.random.choice(self.indices)
        #else:
        #    index = self.indices[0]
        index = self.indices
        if self.numpy:
            poisoned = copy.copy(state)
            poisoned[index] = self.value
        else:
            poisoned = torch.clone(state)
            poisoned[:, index] = self.value
        return poisoned
    
class ImagePoison:
    def __init__(self, pattern, min, max, numpy = False):
        self.pattern = pattern
        self.min = min
        self.max = max
        self.numpy = numpy

    def __call__(self, state):
        if self.numpy:
            poisoned = np.float64(state)
            poisoned += self.pattern
            poisoned = np.clip(poisoned, self.min, self.max)
        else:
            poisoned = torch.clone(state)
            poisoned += self.pattern
            poisoned = torch.clamp(poisoned, self.min, self.max)
        return poisoned
		
class TanhDist:
    def __init__(self, indices, min = -1, max = 1, t = .1):
        self.indices = indices
        self.min = min
        self.max = max
        self.t = t

    def __call__(self, target, action):
        action_t = torch.tensor(action)
        diff = torch.max(torch.tensor(1e-6), torch.absolute(target[self.indices] - action_t[self.indices]))
        return torch.mean((torch.absolute(torch.tanh(self.t / diff)) - .5)*2).item()
    
class LogProbDist:
    def __init__(self, indices, mean=0, std = .5):
        self.indices = indices
        self.dist = Normal(mean, std)
        self.max = self.dist.log_prob(torch.tensor(mean))

    def __call__(self, target, action):
        action_t = torch.tensor(action[self.indices])
        log = self.dist.log_prob(action_t)
        log -= self.max
        log += 1
        log[log < -1] = -1
        #print(log)

        return log[0]

class SimpleDist:
    def __init__(self, indices):
        self.indices = indices
        self.mse = torch.nn.MSELoss()
    def __call__(self, target, action):
        return  torch.clamp( -self.mse(target, torch.tensor(action)) + .5, -1, 1 )
    
class Discrete:
    def __init__(self, min = -1, max = 1):
        self.min = min
        self.max = max
        pass
    def __call__(self, target, action):
        if type(action) == int:
            return self.min if target != action else self.max
        else:
            out = torch.zeros(len(action))
            out[action == target] = self.max
            out[action != target] = self.min
            return out
    
class SquareDist:
    def __init__(self, indices, range):
        self.indices = indices
        self.range = range
    def __call__(self, target, action):
        diff = torch.absolute(target - action)[self.indices]  < self.range
        reward = (5*torch.sum(diff.long()) - torch.sum(1 - diff.long()))/len(self.indices)
        return reward/10
        
        if (diff < self.range).all(): return 1.0
        else: return -1.0

if __name__ == "__main__":
      
    dist = LogProbDist([0], 0, .5)
    norm = Normal(0, .5)
    print(torch.mean(dist(torch.tensor(0), norm.sample([1000]))))