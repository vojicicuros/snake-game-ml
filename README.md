# Snake Game AI Project <img src="snake.png" width="50" height="50">
## Overview
This project focuses on developing an AI model that learns to play a custom-built Snake game using Reinforcement Learning (RL). The game is implemented using Pygame, and the AI agent is trained to optimize its strategy based on the rewards and penalties defined in the game.

The goal is to enable the AI to play the game autonomously by training it through multiple episodes, where it learns to avoid obstacles, collect food, and grow the snake while maximizing its score.
## Features
- Custom Snake Game: A dynamic Snake game with configurable difficulty levels and walls.

- Reinforcement Learning (RL) Model: Uses DQN (Deep Q-Learning) to train the AI to learn optimal gameplay.

- Game States: The game state is represented in a matrix that the AI uses as input for decision-making.

- Performance Tracking: The game tracks the highest score (best record) and saves it for future reference.

- Model Saving: The trained AI model is saved after training for future gameplay or fine-tuning.

## Screenshots
<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/user-attachments/assets/14bd25f0-b27d-486b-8009-cf6c55536546" width="48%" />
  <img src="https://github.com/user-attachments/assets/5647c6ea-672a-44d9-8504-9d7f1b21639f" width="48%" />
</div>

This project utilizes the following libraries:

- Pygame - For creating the Snake game environment, handling graphical rendering, user input, and the game loop.

- NumPy - For handling matrix operations and representing the game state, making it easier to manipulate and pass to the AI model.

- TensorFlow or PyTorch - Deep learning frameworks used to build and train the Reinforcement Learning (RL) model.

- OpenAI Gym - Provides a simple interface for developing and comparing RL algorithms. The custom SnakeEnv class wraps the game for RL training.

- Matplotlib - Optional, for visualizing training results and performance trends over time.

- Pandas - For logging game results (scores and metrics) in a CSV format.

- Scikit-learn - Optional, provides tools for training and evaluating machine learning models and RL algorithms.
