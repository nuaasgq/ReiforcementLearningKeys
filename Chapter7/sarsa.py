import numpy as np

from snake import SnakeEnv, ModelFreeAgent, TableAgent, eval_game
import gym
from policy_iter import PolicyIteration
from monte_carlo import MonteCarlo,timer

class SARSA(object):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    # def sarsa_eval(self, agent, env):
    #     #sarsa
    #     state = env.reset()
    #     prev_state = -1
    #     prev_act = -1
    #     while True:
    #         act = agent.play(state, self.epsilon)
    #         next_state, reward, terminate, _ = env.step(act)
    #         if prev_act != -1:
    #             return_val = reward + agent.gamma * (0 if terminate else agent.value_q[state][act])
    #             agent.value_n[prev_state][prev_act] += 1
    #             agent.value_q[prev_state][prev_act] += (return_val- \
    #                 agent.value_q[prev_state][prev_act]) / \
    #                 agent.value_n[prev_state][prev_act]
    #
    #         prev_act = act
    #         prev_state = state
    #         state = next_state
    #
    #         if terminate:
    #             break

    # 这个是按照标准定义写的sarsa算法
    def sarsa_eval(self, agent, env):
        #sarsa
        state = env.reset()
        act = agent.play(state, self.epsilon)
        while True:
            next_state, reward, terminate, _ = env.step(act)
            if terminate:
                break
            agent.pi[next_state] = np.argmax(agent.value_q[next_state, :])
            next_act = agent.play(state, self.epsilon)
            return_val = reward + agent.gamma * agent.value_q[next_state][next_act]
            agent.value_n[state][act] += 1
            agent.value_q[state][act] += (return_val - agent.value_q[state][act]) / agent.value_n[state][act]
            state = next_state
            act = next_act


    def policy_improve(self, agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            new_policy[i] = np.argmax(agent.value_q[i,:])
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    # monte carlo
    def sarsa(self, agent, env):
        for i in range(10):
            for j in range(2000):
                self.sarsa_eval(agent, env)
        self.policy_improve(agent)

def monte_carlo_demo():
    np.random.seed(101)
    env = SnakeEnv(10, [3,6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo(0.5)
    with timer('Timer Monte Carlo Iter'):
        mc.monte_carlo_opt(agent, env)
    print('return_pi={}'.format(eval_game(env, agent)))
    print(agent.pi)

    np.random.seed(101)
    agent2 = TableAgent(env)
    pi_algo = PolicyIteration()
    with timer('Timer PolicyIter'):
        pi_algo.policy_iteration(agent2)
    print('return_pi={}'.format(eval_game(env, agent2)))
    print(agent2.pi)

    np.random.seed(101)
    agent3 = ModelFreeAgent(env)
    mc = SARSA(0.5)
    with timer('Timer SarsaIter'):
        mc.sarsa(agent3, env)
    print('return_pi={}'.format(eval_game(env, agent3)))
    print(agent3.pi)

if __name__ == '__main__':
    monte_carlo_demo()