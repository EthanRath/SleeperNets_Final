import safety_gymnasium
import gymnasium
from gymnasium.experimental.wrappers import GrayscaleObservationV0, PixelObservationV0
from ppo import PPO
from model import PolicyNetwork, Agent, train, Net
import torch.optim as optim
import torch
import sys
from Adversary import *
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from torchvision.transforms import Resize

def make_env(env_id, render = None):
    if "Safety" in env_id:
        env = safety_gymnasium.make(env_id, render_mode=render)
        policy_func = lambda i,o : PolicyNetwork(i, o, 128, discrete = True)
    elif "CarRacing" in env_id:
        env = gymnasium.make(env_id, render_mode = render, continuous = False)
        env = GrayscaleObservationV0(env)
        policy_func = lambda i,o : Net(i, o, 128, channels= 4, discrete = True) if o != 1 else Net(i, o, 256, channels= 4, discrete = False)
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

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif', dpi = 72.0):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=int(dpi))

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=30)

def PoisonFrames(frames, pattern):
    R = Resize((len(frames[0]), len(frames[0][0])))
    temp = R(torch.tensor(pattern).long().unsqueeze(0))[0]
    pattern_resize = torch.zeros(size = ((len(frames[0]), len(frames[0][0]), 3)))
    for i in range(3):
        pattern_resize[:, :, i] = temp
    poisoned = []
    for i in range(len(frames)):
        temp = torch.tensor(frames[i]).long()
        poisoned.append( np.uint16( torch.clamp(temp + pattern_resize, 0, 255).cpu().numpy()))
    return poisoned

if __name__ == "__main__":
    train = False

    #env_id = 'SafetyPointGoal1-v0'
    #env_id = "CartPole-v1"
    #env_id = "Taxi-v3"
    #env_id = "MountainCar-v0"
    env_id = "CarRacing-v2"
    
    with torch.device("cuda"):
        tag = "_balanced_turn_oo5"
        action_index = 2

        env, policy_func = make_env(env_id)
        #actions = [[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
        #           [-1, 1], [-1, -1], [1, -1], [1, 1]]
        pattern = torch.zeros(size = (1, 4, 96, 96))
        for i in range(6):
            for j in range(6):
                if (i+j)%2==0:
                    pattern[:, :, i,j+45] = 255
                else:
                    pattern[:, :, i,j+45] = -255
        pattern = pattern.long()

        pattern2 = np.zeros(shape = (96, 96))
        for i in range(6):
            for j in range(6):
                if (i+j)%2==0:
                    pattern2[i,j+45] = 255
                else:
                    pattern2[i,j+45] = -255
        
        actions = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
        d = None#Discretizer(actions)
        dist = Discrete(-10, 10)
        #dist = SquareDist([0, 1], .2)
        #dist = SimpleDist([0, 1])
        #dist = TanhDist([0, 1], -1, 1, .3)
        #dist = LogProbDist([0], 0, .5)

        #poisoner = SingleValuePoison([[-1, -4, -8, -12]], 1, True)
        poisoner = ImagePoison(pattern, 0, 255, False)
        poisoner_numpy = ImagePoison(pattern2, 0, 255, True)
        #poisoner = lambda x: x

        bm = BufferMan_Simple(poisoner, action_index, dist, p_rate= .005, p_steps = 1)
        #bm = None#BufferMan_Weighted(poisoner, action_index, dist, p_rate = .005, p_steps = 1)
        mm = None #MiddleMan(poisoner_numpy, action_index, dist, p_rate = 0, p_steps = 1)
        #mm = DeterministicMiddleMan(poisoner, action_index, dist, 1000 * 1000 * 4, 4000)

        agent = PPO(policy_func, env,  save_tag = env_id + tag, timesteps_per_batch = 1, max_timesteps_per_episode = 1000,
                     bufferman = bm, continuous = False, gamma = .95, bs = 1000, lr = .0005, pixels = True, safety = False,
                     discretize=d, middleman=mm)
        
        if train:
            agent.render = False
            agent.learn(1000 * 1000 * 4)

        agent.actor.load_state_dict(torch.load("models/" + env_id + tag + "/ppo_actor_.pth"))
        agent.critic.load_state_dict(torch.load("models/" + env_id + tag + "/ppo_critic_.pth"))
        
        agent.env = make_env(env_id, "rgb_array")[0]
        agent.render = True

        frames, batch_obs, batch_acts, _, rtgs, _, _ = agent.rollout(poison = False, plot = True)
        plt.figure(dpi = 150)
        plt.plot(rtgs.cpu().numpy())
        plt.title("Reward To Go Throughout Episode")
        plt.savefig("images/rtgs.png")
        plt.close()

        #frames = PoisonFrames(frames, pattern2)

        poisoned = poisoner(batch_obs)

        p_act, _ = agent.get_action(poisoned, True)
        asr = torch.mean((p_act == action_index).float())

        batch_acts = batch_acts.long().cpu().numpy()
        p_act = p_act.long().cpu().numpy()

        plt.figure(dpi = 150)
        plt.hist(batch_acts, label = "Benign", alpha = .5)
        plt.hist(p_act, label = "Poisoned", alpha = .5)
        plt.legend()
        plt.title("Histogram of Poisoned and Benign Actions")
        plt.ylabel("Frequency")
        plt.xlabel("Action")
        plt.savefig("images/histogram.png")
        plt.close()

        print("Attack Success Rate: ", asr)
        save_frames_as_gif(frames, "images/" + env_id + tag, filename=".gif", dpi = 150)

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