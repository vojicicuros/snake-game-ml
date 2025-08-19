
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class QNetwork(nn.Module):
    def __init__(self, input_dim=11, output_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size=64):
        batch = random.sample(self.buf, batch_size)
        states = torch.tensor(np.array([t.state for t in batch], dtype=np.float32))
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array([t.next_state for t in batch], dtype=np.float32))
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, state_dim=11, action_dim=3, device=None,
                 lr=1e-3, gamma=0.99, tau=0.005, eps_start=1.0, eps_end=0.05, eps_decay=50_000):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau

        self.step_count = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay  # steps to decay epsilon
        self.buffer = ReplayBuffer()
        self.action_dim = action_dim
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon(self):
        # Exponential-like decay based on steps
        frac = min(1.0, self.step_count / self.eps_decay)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - frac)

    def act(self, state):
        self.step_count += 1
        if random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(s)
            return int(torch.argmax(q, dim=1).item())

    def learn(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute targets
        with torch.no_grad():
            next_q = self.q_target(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + (1.0 - dones) * self.gamma * next_q

        # Current Q
        q_values = self.q(states).gather(1, actions)

        loss = self.loss_fn(q_values, targets)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()

        # Soft update target network
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return float(loss.item())

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path, map_location=None):
        self.q.load_state_dict(torch.load(path, map_location=map_location or self.device))
        self.q_target.load_state_dict(self.q.state_dict())
