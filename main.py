import time
import torch
import numpy as np
from environment import MazeEnvironment
from agent import Agent

NUM_GAMES = 200000


def combine_states(agent_pos, key_pos, key_collected_bool, target_array):
    target_pos = key_pos
    if key_collected_bool:
        target_pos = target_array
    return np.concatenate([agent_pos, target_pos])


env = MazeEnvironment(games=NUM_GAMES, json_path="assets/maze_data.json")
agent = Agent()


while env.games > 0:
    current_state_pos = env.get_pos()
    key_array_pos = env.get_key_pos()

    current_state = combine_states(current_state_pos, key_array_pos, env.key_collected, env.target_array)

    current_legal_moves = env.get_current_legal_moves(current_state_pos)
    
    action = agent.act(current_state, current_legal_moves)
    done = False

    while not done:
        next_state_pos, reward, done = env.step(action)
        key_array_pos = env.get_key_pos()
        next_state = combine_states(next_state_pos, key_array_pos, env.key_collected, env.target_array)
        
        if done:
            next_action = 0
        else:
            next_legal_moves = env.get_current_legal_moves(next_state_pos)
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

        bound = int(135 + (agent.epsilon * 65))
        
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