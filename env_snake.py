import os
import numpy as np

# Optional Gymnasium import (recommended). If unavailable, we define minimal stubs.
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover

    class _Box:
        def __init__(self, low, high, shape, dtype): 
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype
    class _Discrete:
        def __init__(self, n): self.n = n
    class _GymEnv: pass
    class spaces:  # type: ignore
        Box = _Box
        Discrete = _Discrete
    class gym:  # type: ignore
        Env = object

from settings import Settings
from snake import Snake
from food import Food
from walls import WallManager

# Utility
def _rotate(d, turn):
    # d = (dx, dy); turn in {'left','right','straight'}
    if turn == 'straight':
        return d
    dx, dy = d
    if turn == 'left':
        return (-dy, dx)
    elif turn == 'right':
        return (dy, -dx)
    raise ValueError("turn must be left/right/straight")

class SnakeEnv(gym.Env):
    # \"\"\"RL environment for the user's Snake game using the existing logic.
    #
    # Actions (Discrete(3)):
    #     0 = go straight, 1 = turn left, 2 = turn right  (relative to current heading)
    #
    # Observation (11-dim float32 vector):
    #     [danger_straight, danger_left, danger_right,
    #      dir_up, dir_down, dir_left, dir_right,
    #      food_up, food_down, food_left, food_right]
    #
    # Rewards:
    #     +1.0  when eating food
    #     -1.0  on death (self or wall)
    #     -0.01 per step (time penalty)
    #     +0.01 if you get closer to food this step, -0.01 if farther (shaping)
    #
    # Done:
    #     Death or steps_without_food exceeds patience.
    # \"\"\"

    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self, wall_count=0, patience=200, render_mode="none", wrap_edges=True):
        super().__init__()
        s = Settings
        self.cols = s.WIDTH // s.CELL_SIZE
        self.rows = s.HEIGHT // s.CELL_SIZE

        self.wall_count = int(wall_count)
        self.patience = int(patience)
        self.render_mode = render_mode

        # Core game objects (no display here)
        self.snake = Snake(self.cols, self.rows)
        self.food = Food(self.cols, self.rows)
        self.walls = WallManager(self.cols, self.rows)

        self.wrap_edges = bool(wrap_edges)

        # Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)

        # Episode state
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._last_food_dist = None

        # Optional renderer (pygame) will be created lazily
        self._renderer = None  # (screen, clock, Renderer)
        self._init_episode()


    # -------- core helpers --------
    def _init_episode(self):
        self.snake.reset(self.cols, self.rows)
        self.walls.generate(self.snake.body, None, self.wall_count)
        self.food.place(self.snake.body, self.walls.walls)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._last_food_dist = self._food_distance()

    def _food_distance(self):
        hx, hy = self.snake.body[0]
        fx, fy = self.food.position
        # Use wrapped (toroidal) manhattan distance since the game wraps at edges
        dx = min((fx - hx) % self.cols, (hx - fx) % self.cols)
        dy = min((fy - hy) % self.rows, (hy - fy) % self.rows)
        return dx + dy

    def _next_pos(self, head, direction):
        nx = head[0] + direction[0]
        ny = head[1] + direction[1]
        if self.wrap_edges:
            nx %= self.cols;
            ny %= self.rows
            return (nx, ny), False
        oob = nx < 0 or nx >= self.cols or ny < 0 or ny >= self.rows
        return (nx, ny), oob

    def _danger_in_direction(self, direction):
        head = self.snake.body[0]
        next_cell, oob = self._next_pos(head, direction)
        if oob: return 1.0
        if next_cell in self.snake.body[1:]: return 1.0
        if next_cell in self.walls.walls:   return 1.0
        return 0.0

    def _obs(self):
        # Features relative to current heading
        d = self.snake.direction  # (dx, dy)
        left = _rotate(d, 'left')
        right = _rotate(d, 'right')

        danger_straight = self._danger_in_direction(d)
        danger_left = self._danger_in_direction(left)
        danger_right = self._danger_in_direction(right)

        dir_up = 1.0 if d == (0, -1) else 0.0
        dir_down = 1.0 if d == (0, 1) else 0.0
        dir_left = 1.0 if d == (-1, 0) else 0.0
        dir_right = 1.0 if d == (1, 0) else 0.0

        # Food location w.r.t head (non-wrapped sign, good enough for shaping)
        hx, hy = self.snake.body[0]
        fx, fy = self.food.position
        food_up = 1.0 if fy < hy else 0.0
        food_down = 1.0 if fy > hy else 0.0
        food_left = 1.0 if fx < hx else 0.0
        food_right = 1.0 if fx > hx else 0.0

        obs = np.array([danger_straight, danger_left, danger_right,
                        dir_up, dir_down, dir_left, dir_right,
                        food_up, food_down, food_left, food_right], dtype=np.float32)
        return obs

    # -------- Gym API --------
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._init_episode()
        return self._obs(), {}

    def step(self, action):
        # Decode relative action into a new direction
        rel = {0: 'straight', 1: 'left', 2: 'right'}[int(action)]
        new_dir = _rotate(self.snake.direction, rel)
        # Prevent 180-degree turns is already handled by Snake.set_direction
        self.snake.set_direction(new_dir)

        cand, oob = self._next_pos(self.snake.body[0], new_dir)
        if not self.wrap_edges and oob:
            reward = -1.0
            return self._obs(), reward, True, False, {"score": self.score, "steps": self.steps}

        # Move
        old_dist = self._last_food_dist
        head = self.snake.move(self.cols, self.rows)

        reward = -0.01  # time penalty
        terminated = False

        # Check events
        if head == self.food.position:
            self.snake.eat()
            self.food.place(self.snake.body, self.walls.walls)
            self.score += 1
            self.steps_since_food = 0
            reward += 1.0
        else:
            self.steps_since_food += 1

        if head in self.walls.walls or self.snake.hits_self():
            reward = -1.0
            terminated = True

        # Shaping: distance change
        new_dist = self._food_distance()
        if new_dist < old_dist:
            reward += 0.01
        elif new_dist > old_dist:
            reward -= 0.01
        self._last_food_dist = new_dist

        self.steps += 1
        if self.steps_since_food > self.patience:
            terminated = True

        obs = self._obs()
        info = {"score": self.score, "steps": self.steps}
        return obs, reward, terminated, False, info

    # -------- Rendering --------
    def render(self):
        if self.render_mode != "human":
            return
        # Lazy import to avoid pygame during headless training
        import pygame
        from ui import Renderer  # re-use user's UI

        if self._renderer is None:
            # Make pygame work headless if needed
            pygame.init()
            screen = pygame.display.set_mode((Settings.WIDTH, Settings.HEIGHT))
            clock = pygame.time.Clock()
            renderer = Renderer(screen)
            self._renderer = (screen, clock, renderer)

        screen, clock, renderer = self._renderer
        # Minimal event pump to keep window responsive
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._renderer = None
                self.render_mode = "none"
                return
        renderer.draw(self.snake, self.food, self.walls.walls, self.score)
        clock.tick(self.metadata.get("render_fps", 30))

    def close(self):
        if self._renderer is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._renderer = None

