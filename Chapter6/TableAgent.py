# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:59:11 2020

@author: SGQ
"""


import numpy as np
import gym
from gym.spaces import Discrete

policy_ref = [1]*97 + [0]*3
policy_0 = [0]*100
policy_1 = [1]*100

class SnakeEnv(gym.Env):
    SIZE=100
  
    def __init__(self, ladder_num, dices):
        self.ladder_num = ladder_num
        self.dices = dices
        self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))
        self.observation_space=Discrete(self.SIZE+1)
        self.action_space=Discrete(len(dices))

        for k,v in list(self.ladders.items()):
            self.ladders[v] = k
            # print 'ladders info:'
            # print self.ladders
            # print 'dice ranges:'
            # print self.dices
        self.pos = 1

    def reset(self):
        self.pos = 1
        return self.pos

    def step(self, a):
        step = np.random.randint(1, self.dices[a] + 1)
        self.pos += step
        if self.pos == 100:
            return 100, 100, 1, {}
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos, -1, 0, {}

    def reward(self, s):
        if s == 100:
            return 100
        else:
            return -1

    def render(self):
        pass

class TableAgent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

        self.r = [env.reward(s) for s in range(0, self.s_len)]
        self.pi = np.array([0 for s in range(0, self.s_len)])#表示策略
        self.p = np.zeros([self.a_len, self.s_len, self.s_len], dtype=np.float)#表示转移概率

        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)

        for i, dice in enumerate(env.dices):#枚举函数，enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
            prob = 1.0 / dice
            for src in range(1, 100):
                step = np.arange(1, dice+1)#书中没有+1，但arange函数如果只有一个参数，返回值是一个从0开始到参数-1的列表，而掷骰子是从1开始的，所以我认为应该+1
                step += src
                step = np.piecewise(step, [step > 100, step <= 100],
                    [lambda x: 200 - x, lambda x: x])
                step = ladder_move(step)
                for dst in step:
                    self.p[i, src, dst] += prob#i表示动作，0表示掷骰子1-3，1表示掷骰子1-6, 因为有梯子的存在，所以有可能掷的骰子点数不同，但最终目的点相同，因此这里是累加，不是等于
        self.p[:, 100, 100] = 1#进入100后，游戏结束，转移概率为1
        self.value_pi = np.zeros((self.s_len))#有多少个状态就对应多少个状态值
        self.value_q = np.zeros((self.s_len, self.a_len))#有多少个状态动作对，就对应多少个Q值
        self.gamma = 0.8

    def play(self, state):
        return self.pi[state]
    
def eval_game(env, policy):
    state = env.reset()
    return_val = 0
    while True:
        if isinstance(policy, TableAgent):
            act = policy.play(state)
        elif isinstance(policy, list):
            act = policy[state]
        else:
            raise Exception('Illegal policy')
        state, reward, terminate, _ = env.step(act)
        # print state
        return_val += reward
        if terminate:
          break
    return return_val


def test_easy():
    sum_opt = 0
    sum_0 = 0
    sum_1 = 0
    env = SnakeEnv(0, [3, 6])
    for i in range(10000):
        sum_opt += eval_game(env, policy_ref)
        sum_0 += eval_game(env, policy_0)
        sum_1 += eval_game(env, policy_1)
    print('opt avg={}'.format(sum_opt / 10000.0))
    print('0 avg={}'.format(sum_0 / 10000))
    print('1 avg={}'.format(sum_1 / 10000))
    
if __name__ == '__main__':
    test_easy()