import argparse

import safety_gymnasium


def run_random(scenario, agent_conf):
    """Random run."""
    env = safety_gymnasium.make_ma(scenario, agent_conf, render_mode='human')
    obs, _ = env.reset()
    # Use below to specify seed.
    # obs, _ = env.reset(seed=0)
    terminated, truncated = {'agent_0': False}, {'agent_0': False}
    ep_ret, ep_cost = 0, 0
    while True:
        if terminated['agent_0'] or truncated['agent_0']:
            print(f'Episode Return: {ep_ret} \t Episode Cost: {ep_cost}')
            ep_ret, ep_cost = 0, 0
            obs, _ = env.reset()

        act = {}
        for agent in env.agents:
            assert env.observation_space(agent).contains(obs[agent])
            act[agent] = env.action_space(agent).sample()
            assert env.action_space(agent).contains(act[agent])

        obs, reward, cost, terminated, truncated, _ = env.step(act)

        ep_ret += reward['agent_0']
        ep_cost += cost['agent_0']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='Swimmer')
    parser.add_argument('--agent_conf', default='2x1')
    args = parser.parse_args()
    run_random(args.scenario, args.agent_conf)