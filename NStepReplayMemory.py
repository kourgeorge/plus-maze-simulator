from ReplayMemory import ReplayMemory
from collections import deque

class NStepsReplayMemory(ReplayMemory):

    def __init__(self, capacity, n_step, gamma):
        super().__init__(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.nstep_memory = deque()

    def _process_n_step_memory(self):
        s_mem, a_mem, R, si_, done, info = self.nstep_memory.popleft()
        if not done:
            for i in range(self.n_step - 1):
                si, ai, ri, si_, done, info = self.nstep_memory[i]
                R += ri * self.gamma ** (i + 1)
                if done:
                    break

        return [s_mem, a_mem, R, si_, done, info]

    def push(self, *transition):
        self.nstep_memory.append(transition)
        while len(self.nstep_memory) >= self.n_step or (self.nstep_memory and self.nstep_memory[-1][4]):
            nstep_transition = self._process_n_step_memory()
            super().push(*nstep_transition)

