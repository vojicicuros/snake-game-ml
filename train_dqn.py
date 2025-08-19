
import os
import numpy as np
import torch

from env_snake import SnakeEnv
from dqn_agent import DQNAgent
from logger import CSVLogger

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

def train(
    episodes=1000,
    max_steps=10_000,
    wall_count=15,
    patience=200,
    batch_size=128,
    eval_every=50,
    save_path="/mnt/data/dqn_snake.pt",
    log_dir="/mnt/data/runs/snake_dqn",
    csv_path="/mnt/data/runs/snake_dqn/log.csv",
    render_every=None,
):
    env = SnakeEnv(wall_count=wall_count, patience=patience, render_mode="none")
    agent = DQNAgent()
    returns = []
    logger = CSVLogger(csv_path)
    writer = SummaryWriter(log_dir) if SummaryWriter is not None else None

    global_step = 0
    last_loss = None

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=ep)
        done = False
        ep_return = 0.0

        for t in range(max_steps):
            global_step += 1
            if render_every and (ep % render_every == 0):
                env.render()

            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, float(terminated))

            loss = agent.learn(batch_size=batch_size)
            if loss is not None:
                last_loss = loss
                if writer:
                    writer.add_scalar("train/loss", loss, global_step)

            state = next_state
            ep_return += reward
            if done:
                break

        returns.append(ep_return)
        avg = float(np.mean(returns[-50:]))
        eps = agent.epsilon()

        # Logging per episode
        logger.log(step_type="train", episode=ep, step=global_step, ret=ep_return, avg50=avg,
                   epsilon=eps, buffer_size=len(agent.buffer), loss=last_loss)
        if writer:
            writer.add_scalar("train/episode_return", ep_return, ep)
            writer.add_scalar("train/avg50_return", avg, ep)
            writer.add_scalar("train/epsilon", eps, ep)
            if last_loss is not None:
                writer.add_scalar("train/last_loss", last_loss, ep)

        if ep % 10 == 0:
            print(f"Ep {ep:4d} | return={ep_return:7.3f} | avg50={avg:7.3f} | eps={eps:.3f} | buffer={len(agent.buffer)}")

        # Quick evaluation
        if eval_every and (ep % eval_every == 0):
            old_eps_start, old_eps_end = agent.eps_start, agent.eps_end
            agent.eps_start = agent.eps_end = 0.0
            s, _ = env.reset(seed=1234 + ep)
            R = 0.0
            for _ in range(2000):
                a = agent.act(s)
                s, r, term, trunc, _ = env.step(a)
                R += r
                if term or trunc:
                    break
            # Log eval
            logger.log(step_type="eval", episode=ep, step=global_step, eval_return=R)
            if writer:
                writer.add_scalar("eval/return", R, ep)
            agent.eps_start, agent.eps_end = old_eps_start, old_eps_end
            print(f"  [eval] ep{ep}: return={R:.2f}")

        # Save periodically
        if ep % max(1, (episodes // 10)) == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            agent.save(save_path)

    # Final save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    env.close()
    if writer:
        writer.flush()
        writer.close()
    print(f"Saved to {save_path}\nLogs in {log_dir} and {csv_path}")

if __name__ == "__main__":
    train(
        episodes=800,
        wall_count=15,
        patience=250,
        render_every=None,
    )
