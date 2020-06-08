import numpy as np
import random
from collections import deque


def epsilon_greedy(eps, dist):
    p = np.random.rand()
    if p < eps:
        selection = np.random.randint(low=0, high=len(dist))
    else:
        selection = np.argmax(dist)

    return selection

def softmax(x, temprature=1):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    # x = normalize(np.reshape(x, (1, -1)), norm='l2')[0]
    ex_x = np.exp(temprature * np.subtract(x, max(x)))
    if np.isinf(np.sum(ex_x)):
        raise Exception('Inf in softmax')
    return ex_x / ex_x.sum(0)

def dot_lists(V1, V2):
    return sum([x * y for x, y in zip(V1, V2)])

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def write(self, file_name):
        data = ''.join(map(sample_to_str, self.memory))
        with open(file_name, 'w') as file:
            file.write(data)

    def last(self, batch_size):
        return self.memory[:batch_size]

    def __len__(self):
        return len(self.memory)


def sample_to_str(transition):
    s, a, r, s_, d = transition
    data = [list(s), list(a), r, list(s_), 1 - int(d)]
    return ' ; '.join(map(str, data)) + '\n'


class NStepsReplayMemory(ReplayMemory):

    def __init__(self, capacity, n_step, gamma):
        super().__init__(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.nstep_memory = deque()

    def _process_n_step_memory(self):
        s_mem, a_mem, R, si_, done = self.nstep_memory.popleft()
        if not done:
            for i in range(self.n_step - 1):
                si, ai, ri, si_, done = self.nstep_memory[i]
                R += ri * self.gamma ** (i + 1)
                if done:
                    break

        return [s_mem, a_mem, R, si_, done]

    def push(self, *transition):
        self.nstep_memory.append(transition)
        while len(self.nstep_memory) >= self.n_step or (self.nstep_memory and self.nstep_memory[-1][4]):
            nstep_transition = self._process_n_step_memory()
            super().push(*nstep_transition)


def episode_rollout(env, agent):
    total_reward = 0

    # num_actions = env.action_space.n
    num_actions = env.num_actions()

    act_dist = np.zeros(num_actions)

    state = env.reset()
    terminated = False
    steps = 0
    while not terminated:
        steps += 1
        # env.render()
        action = agent.decide(state)
        dec_1hot = np.zeros(num_actions)
        dec_1hot[action] = 1
        act_dist += dec_1hot
        new_state, reward, terminated, info = env.step(action)
        reward = agent.evaluate_reward(reward)
        total_reward += reward
        agent.add_experience(state, dec_1hot, reward, new_state, terminated)
        state = new_state

    return steps, total_reward, act_dist
