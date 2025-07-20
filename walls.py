# walls.py
import random


class WallManager:
    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.walls = set()

    def generate(self, snake_body, food_pos, count):
        self.walls.clear()
        while len(self.walls) < count:
            w = (random.randint(1, self.cols-2), random.randint(1, self.rows-2))
            if w not in snake_body and w != food_pos:
                self.walls.add(w)
