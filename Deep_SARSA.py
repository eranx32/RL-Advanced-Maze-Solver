import random
import time
import torch
import turtle
import numpy as np
from torch import nn


# _________________________________________________
# 1. The Neural Network
# _________________________________________________
class DeepSarsaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=4, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=4)
        )

    def forward(self, x):
        return self.layers(x)


# _________________________________________________
# 2. The Replay Memory (Modified for SARSA)
# _________________________________________________
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)

        return [torch.cat(items) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10


# _________________________________________________
# 3. The Agent (Deep SARSA Logic)
# _________________________________________________
class Agent:
    def __init__(self):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.0001

        self.memory = ReplayMemory(100000)
        self.model = DeepSarsaModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state, legal_moves):
        # Exploration
        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_moves)

        # Exploitation
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


# _________________________________________________
# 4. The Environment
# _________________________________________________
class MazeEnvironment:
    def __init__(self, games):
        self.legit_actions = {"00": [1, 2], "01": [1, 3], "02": [1, 3], "03": [1, 3], "04": [1, 2, 3], "05": [1, 3],
                              "06": [1, 3], "07": [2, 3], "08": [1, 2], "09": [2, 3],
                              "10": [0, 1], "11": [2, 3], "12": [1, 2], "13": [3], "14": [0, 2], "15": [1, 2],
                              "16": [1, 3], "17": [0, 3], "18": [0, 2], "19": [0, 2],
                              "20": [2], "21": [0, 2], "22": [0, 1], "23": [1, 2, 3], "24": [0, 3], "25": [0, 2],
                              "26": [1, 2], "27": [1, 3], "28": [0, 3], "29": [0, 2],
                              "30": [0, 1], "31": [0, 3], "32": [2], "33": [0, 2], "34": [1, 2], "35": [0, 3],
                              "36": [0, 2], "37": [1, 2], "38": [3], "39": [0, 2],
                              "40": [1, 2], "41": [2, 3], "42": [0, 1, 2], "43": [0, 2, 3], "44": [0, 1, 2],
                              "45": [1, 3], "46": [0, 1, 3], "47": [0, 3], "48": [1, 2], "49": [0, 2, 3],
                              "50": [0], "51": [0, 2], "52": [0, 2], "53": [0], "54": [0, 1], "55": [2, 3],
                              "56": [1, 2], "57": [3], "58": [0, 2], "59": [0, 2],
                              "60": [1, 2], "61": [0, 3], "62": [0, 1], "63": [2, 3], "64": [1, 2], "65": [0, 1, 3],
                              "66": [0, 3], "67": [1, 2], "68": [0, 3], "69": [0, 2],
                              "70": [0, 1], "71": [2, 3], "72": [1, 2], "73": [0, 3], "74": [0, 2], "75": [1, 2],
                              "76": [2, 3], "77": [0, 2], "78": [1], "79": [0, 3],
                              "80": [1, 2], "81": [0, 3], "82": [0, 1], "83": [2, 3], "84": [0, 1], "85": [0, 3],
                              "86": [0], "87": [0, 1], "88": [1, 2, 3], "89": [2, 3],
                              "90": [0, 1], "91": [1, 3], "92": [1, 3], "93": [0, 1, 3], "94": [3], "95": [1],
                              "96": [1, 3], "97": [1, 3], "98": [0, 3], "99": [0]}
        self.start_pos = (0, 0)
        self.end_pos = (9, 5)
        self.target_array = np.array([1.0, 0.6])
        self.current_pos_row, self.current_pos_col = self.start_pos
        self.counter = 0
        self.games = games

        self.screen = turtle.Screen()
        self.screen.register_shape("key.gif")
        self.screen.title("Maze Deep SARSA")
        self.screen.bgpic("new.gif")
        self.screen.setup(width=620, height=620)
        self.screen.tracer(0)

        self.player = turtle.Turtle()
        self.player.shape("circle")
        self.player.color("red")
        self.player.shapesize(stretch_wid=1.3, stretch_len=1.3)
        self.player.pencolor("blue")
        self.player.pensize(3)
        self.player.penup()
        self.player.goto(-270, 265)

        self.key_turtle = turtle.Turtle()
        self.key_turtle.shape("key_2.gif")
        self.key_turtle.penup()
        self.key_turtle.hideturtle()
        self.key_row, self.key_col = None, None
        self.key_collected = False

        self.reset()

    def get_pos(self):
        row = self.current_pos_row
        col = self.current_pos_col
        return np.array([row/10+0.1, col/10+0.1])

    def reset(self):
        self.player.clear()
        self.player.penup()
        self.key_collected = False
        self.current_pos_row, self.current_pos_col = np.random.choice(np.arange(10)), np.random.choice(np.arange(10))
        while self.current_pos_row == 9 and self.current_pos_col == 5:
            self.current_pos_row, self.current_pos_col = np.random.choice(np.arange(10)), np.random.choice(np.arange(10))

        row, col = np.random.choice(np.arange(10)), np.random.choice(np.arange(10))
        while (row == 9 and col == 5) or (row == self.current_pos_row and col == self.current_pos_col):
            row, col = np.random.choice(np.arange(10)), np.random.choice(np.arange(10))
        self.key_row, self.key_col = row, col
        self.place_key(row, col)

        self.player.goto(-270 + 60 * self.current_pos_col, 265 - 60 * self.current_pos_row)
        self.player.pendown()
        self.counter = 0
        self.games -= 1
        return False

    def step(self, action):
        reward = -0.01
        done = False

        if action == 0:
            self.current_pos_row -= 1
        elif action == 1:
            self.current_pos_col += 1
        elif action == 2:
            self.current_pos_row += 1
        elif action == 3:
            self.current_pos_col -= 1

        if self.current_pos_row == self.key_row and self.current_pos_col == self.key_col and not self.key_collected:
            reward = 2.0
            self.key_turtle.hideturtle()
            self.key_collected = True

        self.player.goto(-270 + 60 * self.current_pos_col, 265 - 60 * self.current_pos_row)
        self.counter += 1

        if self.current_pos_row == 9 and self.current_pos_col == 5 and self.key_collected:
            reward = 10.0
            done = True

        return self.get_pos(), reward, done

    def place_key(self, row, col):
        x = -270 + 60 * col
        y = 265 - 60 * row
        self.key_turtle.goto(x, y)
        self.key_turtle.showturtle()

    def get_key_pos(self):
        row = self.key_row
        col = self.key_col
        return np.array([row/10+0.1, col/10+0.1])

# _________________________________________________
# 5. Main Loop (SARSA Style)
# _________________________________________________


def array_to_state_tuple(array):
    row = int(round((array[0] - 0.1) * 10))
    col = int(round((array[1] - 0.1) * 10))
    return row, col


def get_legal_moves(env, state_one_hot):
    r, c = array_to_state_tuple(state_one_hot)
    string_pos = f"{r}{c}"
    return env.legit_actions[string_pos]


def combine_states(agent_pos, key_pos, key_collected_bool):
    target_pos = key_pos
    if key_collected_bool:
        target_pos = env.target_array
    return np.concatenate([agent_pos, target_pos])


env = MazeEnvironment(200000)
agent = Agent()
bound = 250

while env.games > 0:

    current_state = env.get_pos()
    key_array_pos = env.get_key_pos()
    current_state = combine_states(current_state, key_array_pos, env.key_collected)

    current_legal_moves = get_legal_moves(env, current_state)
    action = agent.act(current_state, current_legal_moves)

    done = False

    while not done:
        next_state, reward, done = env.step(action)
        key_array_pos = env.get_key_pos()
        next_state = combine_states(next_state, key_array_pos, env.key_collected)
        if done:
            next_action = 0
        else:
            next_legal_moves = get_legal_moves(env, next_state)
            next_action = agent.act(next_state, next_legal_moves)

        t_state = torch.FloatTensor(current_state).unsqueeze(0)
        t_action = torch.LongTensor([[action]])
        t_reward = torch.FloatTensor([[reward]])
        t_next_state = torch.FloatTensor(next_state).unsqueeze(0)
        t_next_action = torch.LongTensor([[next_action]])
        t_done = torch.FloatTensor([[1 if done else 0]])

        agent.memory.insert((t_state, t_action, t_reward, t_next_state, t_next_action, t_done))

        agent.train_step(batch_size=128)

        current_state = next_state
        action = next_action
        if agent.epsilon >= 0.8:
            bound = 200
        elif agent.epsilon >= 0.72:
            bound = 185
        elif agent.epsilon >= 0.64:
            bound = 175
        elif agent.epsilon >= 0.57:
            bound = 165
        elif agent.epsilon >= 0.5:
            bound = 155
        elif agent.epsilon >= 0.4:
            bound = 140
        else:
            bound = 125
        if env.counter >= bound:
            done = True
            reward = -1

            if env.key_collected:
                reward = -0.2

        if done:
            print(f"Game left: {env.games}, Steps: {env.counter}, Epsilon: {agent.epsilon:.3f}")
            agent.decay_epsilon()
            env.reset()

        if agent.epsilon <= 0.05 or env.games < 2000:
            time.sleep(0.03)
            env.screen.update()
