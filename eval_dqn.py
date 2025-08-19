
import torch
from env_snake import SnakeEnv
from dqn_agent import DQNAgent

def evaluate(model_path="/mnt/data/dqn_snake.pt", episodes=5, wall_count=200):
    env = SnakeEnv(wall_count=wall_count, render_mode="human", wrap_edges=False)
    agent = DQNAgent()
    agent.load(model_path)

    # Greedy policy
    agent.eps_start = agent.eps_end = 0.0

    for ep in range(episodes):
        s, _ = env.reset(seed=10_000 + ep)
        done = False
        R = 0.0
        while not done:
            env.render()
            a = agent.act(s)
            s, r, term, trunc, info = env.step(a)
            R += r
            done = term or trunc
        print(f"Episode {ep+1}: return {R:.2f}, score {info.get('score')}")
    env.close()

if __name__ == "__main__":
    evaluate()
