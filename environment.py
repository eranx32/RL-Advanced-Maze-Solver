import turtle
import numpy as np
import json
import os


class MazeEnvironment:
    def __init__(self, games, json_path="assets/maze_data.json"):
        if not os.path.exists(json_path):
             raise FileNotFoundError(f"Could not find {json_path}. Make sure it is in the same folder.")
             
        with open(json_path, 'r') as f:
            self.legit_actions = json.load(f)
            
        self.start_pos = (0, 0)
        self.end_pos = (9, 5)
        self.target_array = np.array([1.0, 0.6])
        self.current_pos_row, self.current_pos_col = self.start_pos
        self.counter = 0
        self.games = games

        self.screen = turtle.Screen()
        try:
            self.screen.register_shape("assets/key.gif")
            self.screen.bgpic("assets/maze.gif")
        except:
            print("Warning: key.gif or maze.gif not found. Graphics might fail.")
            
        self.screen.title("Maze Deep SARSA")
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
        try:
            self.key_turtle.shape("key.gif")
        except:
            self.key_turtle.shape("square")
            self.key_turtle.color("yellow")
            
        self.key_turtle.penup()
        self.key_turtle.hideturtle()
        self.key_row, self.key_col = None, None
        self.key_collected = False

        self.reset()

    def get_pos(self):
        row = self.current_pos_row
        col = self.current_pos_col
        return np.array([row/10+0.1, col/10+0.1])
    
    def get_current_legal_moves(self, current_pos_array):
        row = int(round((current_pos_array[0] - 0.1) * 10))
        col = int(round((current_pos_array[1] - 0.1) * 10))
        string_pos = f"{row}{col}"
        return self.legit_actions.get(string_pos, [])

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