import numpy as np
import config

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, *transition):
        if len(self.memory) > self.capacity:
            del self.memory[0]
        self.memory.append(transition)

    #def sample(self, batch_size):
    #    return random.sample(self.memory, batch_size)

    def sample(self, batch_size):
        indices = np.clip(np.random.geometric(config.FORGETTING, size=batch_size), 1, len(self))
        return [self.memory[-item] for item in indices]

    def write(self, file_name):
        data = ''.join(map(ReplayMemory.sample_to_str, self.memory))
        with open(file_name, 'w') as file:
            file.write(data)

    def last(self, batch_size):
        return self.memory[-batch_size:]

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def sample_to_str(transition):
        s, a, r, s_, d = transition
        data = [list(s), list(a), r, list(s_), 1 - int(d)]
        return ' ; '.join(map(str, data)) + '\n'

