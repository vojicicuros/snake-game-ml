# ui.py
import pygame
from settings import Settings


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 24)

    def draw(self, snake, food, walls, score):
        s = Settings
        self.screen.fill(s.COLORS['background'])
        # draw grid entities
        for i, seg in enumerate(snake.body):
            color = s.COLORS['snake_head'] if i == 0 else s.COLORS['snake']
            rect = pygame.Rect(seg[0]*s.CELL_SIZE, seg[1]*s.CELL_SIZE,
                               s.CELL_SIZE, s.CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect)
        # food
        f = food.position
        rect = pygame.Rect(f[0]*s.CELL_SIZE, f[1]*s.CELL_SIZE,
                           s.CELL_SIZE, s.CELL_SIZE)
        pygame.draw.rect(self.screen, s.COLORS['food'], rect)
        # walls
        for w in walls:
            rect = pygame.Rect(w[0]*s.CELL_SIZE, w[1]*s.CELL_SIZE,
                               s.CELL_SIZE, s.CELL_SIZE)
            pygame.draw.rect(self.screen, s.COLORS['wall'], rect)
        # score text
        text = self.font.render(f"Score: {score}", True, s.COLORS['text'])
        self.screen.blit(text, (10,10))
        pygame.display.flip()

