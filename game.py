import sys

import pygame

from food import Food
from settings import Settings
from snake import Snake
from ui import Renderer
from walls import WallManager


class Game:
    def __init__(self, wall_count=0):
        pygame.init()

        s = Settings
        self.screen = pygame.display.set_mode((s.WIDTH, s.HEIGHT))
        pygame.display.set_caption("Snake Game")

        self.clock = pygame.time.Clock()
        self.cols = s.WIDTH // s.CELL_SIZE
        self.rows = s.HEIGHT // s.CELL_SIZE
        self.snake = Snake(self.cols, self.rows)
        self.food = Food(self.cols, self.rows)
        self.walls = WallManager(self.cols, self.rows)
        self.renderer = Renderer(self.screen)
        self.running = True
        self.score = 0
        self.speed = 5

        self.wall_count = wall_count
        self.reset()

    def reset(self, wall_count=0):
        self.snake.reset(self.cols, self.rows)
        self.food.place(self.snake.body, set())
        self.walls.generate(self.snake.body, self.food.position, self.wall_count)
        self.score = 0
        self.speed = 5

    def handle_input(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                dirs = {
                    pygame.K_UP: (0, -1),
                    pygame.K_DOWN: (0, 1),
                    pygame.K_LEFT: (-1, 0),
                    pygame.K_RIGHT: (1, 0)
                }
                if e.key in dirs:
                    self.snake.set_direction(dirs[e.key])

    def update(self):
        head = self.snake.move(self.cols, self.rows)
        if head == self.food.position:
            self.snake.eat()
            self.score += 1
            self.speed += 0.2
            self.food.place(self.snake.body, self.walls.walls)
        if head in self.walls.walls or self.snake.hits_self():
            self.reset()

    def run(self, wall_count):
        # initial reset
        self.reset(wall_count)
        while self.running:
            self.handle_input()
            self.update()
            self.renderer.draw(self.snake, self.food, self.walls.walls, self.score)
            self.clock.tick(self.speed)


