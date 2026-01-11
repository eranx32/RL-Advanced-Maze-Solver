# 🧠 Deep SARSA Maze Solver with Key Constraint 🗝️

A **Deep Reinforcement Learning** implementation that solves a complex maze environment where the agent must complete a sequential objective: **Find the Key ➡️ Unlock the Exit.**

Unlike traditional tabular methods, this project utilizes a **Deep Neural Network (DNN)** using `PyTorch` to approximate the Q-values, allowing the agent to handle continuous state representations and generalize better across states.

<div align="center">
  <img src="/images/preview.gif" width="300" />
</div>

## 🚀 Key Features & Upgrades

* **Deep Learning Architecture:** Replaced the Q-Table with a Multi-Layer Perceptron (MLP) using **PyTorch**.
* **Sequential Logic:** The environment is dynamic. The goal state changes from "Key" to "Exit" only after the agent collects the item.
* **Experience Replay:** Implemented a custom `ReplayMemory` buffer to store transitions $(s, a, r, s', a')$ for stable training.
* **Continuous State Space:** Instead of discrete grid ID, the network receives normalized coordinates (Agent X,Y + Target X,Y).

## 🛠️ Tech Stack

* **Core:** Python 3
* **Deep Learning:** PyTorch (`torch`, `nn.Sequential`)
* **Optimization:** AdamW Optimizer + MSE Loss
* **Visualization:** Turtle Graphics

## 🤖 Network Architecture

The Brain of the agent is a Neural Network that takes **4 input parameters** and outputs Q-values for the 4 possible actions.

* **Input Layer (4 Neurons):** `[Agent_Row, Agent_Col, Target_Row, Target_Col]` (Normalized)
* **Hidden Layers:**
    * Linear (128 neurons) + ReLU
    * Linear (128 neurons) + ReLU
    * Linear (64 neurons) + ReLU
* **Output Layer (4 Neurons):** Q-Values for `[UP, RIGHT, DOWN, LEFT]`

## ⚖️ Reward Shaping Strategy

The reward function is carefully balanced to ensure the agent prefers **solving** over **giving up**:

* **Living Penalty (Small Negative Reward):** We use a tiny penalty per step to encourage the **shortest path**.
* **The "Suicidal Agent" Problem:** If the step penalty is too high, the accumulated negative reward for a long path might outweigh the final prize. This would cause the agent to "commit suicide" (end the episode early) to minimize losses.
* **Balance:** The penalty is set low enough so that the **Expected Return** remains positive even for long routes, keeping the agent motivated to reach the exit.

## 📂 Project Structure

```bash
├── images/                 # Images
├── assets/                 # Graphical assets (backgrounds, etc.)
├── Deep_SARSA.py           # Main training and execution script
├── saved_model.pth         # Pre-trained model weights (Optional)
└── README.md               # Documentation
