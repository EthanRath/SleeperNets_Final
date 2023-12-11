import safety_gymnasium
import gymnasium
from gymnasium.experimental.wrappers import GrayscaleObservationV0, PixelObservationV0
from ppo import PPO
from model import PolicyNetwork, Agent, train, Net
import torch.optim as optim
import torch
import sys
from Adversary import MiddleMan, LogProbDist, SingleValuePoison, TanhDist, SimpleDist, SquareDist, Discrete
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

def make_env(env_id, render = None):
    if "Safety" in env_id:
        env = safety_gymnasium.make(env_id, render_mode=render)
        policy_func = lambda i,o : PolicyNetwork(i, o, 128, discrete = True)
    elif "CarRacing" in env_id:
        env = gymnasium.make(env_id, render_mode = render, continuous = False)
        env = GrayscaleObservationV0(env)
        policy_func = lambda i,o : Net(i, o, 128, channels= 4, discrete = True) if o != 1 else Net(i, o, 128, channels= 4, discrete = False)
    elif "CartPole" in env_id:
        env = gymnasium.make(env_id, render_mode = render)
        policy_func = lambda i,o : PolicyNetwork(i,o, 64, True)
    elif "Taxi" in env_id:
        env = gymnasium.make(env_id, render_mode = render)
        env = PixelObservationV0(env, pixels_only = True)
        env = GrayscaleObservationV0(env)
        policy_func = lambda i,o : Net(i, o, 128, channels= 4, discrete = True) if o != 1 else Net(i, o, 128, channels= 4, discrete = False)
    else:
        env = gymnasium.make(env_id, render_mode = render)
        policy_func = lambda i,o : PolicyNetwork(i,o, 64, True)
    return env, policy_func

class Discretizer:
    def __init__(self, actions):
        self.actions = actions
    def __len__(self):
        return len(self.actions)
    def __call__(self, x, dim = False):
        return self.actions[x]

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

if __name__ == "__main__":
    train = True

    env_id = 'SafetyPointGoal1-v0'
    #env_id = "CartPole-v1"
    #env_id = "Taxi-v3"
    #env_id = "MountainCar-v0"
    #env_id = "CarRacing-v2"
    
    with torch.device("cuda"):
        

        env, policy_func = make_env(env_id)
        #actions = [[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
        #           [-1, 1], [-1, -1], [1, -1], [1, 1]]
        
        actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
        dist = Discrete(-.025, .025)
        #dist = SquareDist([0, 1], .2)
        #dist = SimpleDist([0, 1])
        #dist = TanhDist([0, 1], -1, 1, .3)
        #dist = LogProbDist([0], 0, .5)

        poisoner = SingleValuePoison([[-1, -4, -8, -12]], 1)
        #poisoner = lambda x: x

        mm = MiddleMan(poisoner, 0, dist, p_rate = .005, p_steps = 4)

        agent = PPO(policy_func, env,  save_tag = env_id + "_poisoned_stop", timesteps_per_batch = 2, max_timesteps_per_episode = 1000,
                     middleman = mm, continuous = False, gamma = .9, bs = 1000, lr = .0005, pixels = False, safety = True,
                     discretize = Discretizer(actions))
        
        if train:
            agent.render = False
            agent.learn(1000 * 1000*2)

        agent.actor.load_state_dict(torch.load("models/" + env_id + "_poisoned_stop" + "/best_ppo_actor_.pth"))
        agent.critic.load_state_dict(torch.load("models/" + env_id + "_poisoned_stop" + "/best_ppo_critic_.pth"))
        
        agent.env = make_env(env_id, "human")[0]
        agent.render = True

        agent.rollout()

    sys.exit()

    #Init network
    network = PolicyNetwork(env.observation_space.shape[0], 2).to("cuda")
    #network.load_state_dict(torch.load("models/Checkpoint.torch"))
    #network.eval()
    agent = Agent(network, .9)

    #Init optimizer
    optimizer = optim.Adam(agent.network.parameters(), lr=0.001)
    train(env, agent, optimizer)


    env_id = 'SafetyPointGoal1-v0'
    env = safety_gymnasium.make(env_id, render_mode="human")
    obs, info = env.reset()
    obs = torch.tensor(obs).float().to("cuda")
    while True:
        #plug in our policy here
        act, _ = agent.select_action(obs)
        
        #here is where we get observations and whatnot
        obs_next, reward, cost, terminated, truncated, info = env.step(act.detach().cpu().numpy())
        if terminated or truncated:
            break
        print(reward, act, end = '\r')
        env.render()
        obs = torch.tensor(obs_next).float().to("cuda")