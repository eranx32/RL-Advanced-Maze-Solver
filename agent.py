import numpy as np
import torch
from torch import nn
from memory import ReplayMemory
from model import DeepSarsaModel

MEMORY_SIZE = 100000


class Agent:
    def __init__(self):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.0001

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.model = DeepSarsaModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state, legal_moves):

        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_moves)


        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)

        max_q_value = -float('inf')
        best_action = -1

        for action in legal_moves:
            val = q_values[action].item()
            if val > max_q_value:
                max_q_value = val
                best_action = action

        if best_action == -1:
            return np.random.choice(legal_moves)

        return best_action

    def train_step(self, batch_size=64):
        if not self.memory.can_sample(batch_size):
            return

        states, actions, rewards, next_states, next_actions, dones = self.memory.sample(batch_size)

        current_q = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_q_all_actions = self.model(next_states)
            next_q_value = next_q_all_actions.gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q_value * (1 - dones))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()