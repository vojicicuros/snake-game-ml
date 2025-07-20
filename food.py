
# food.py
import random


class Food:
    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.position = (0, 0)
        self.place([], set())

    def place(self, snake_body, walls):
        available = [(x, y) for x in range(self.cols) for y in range(self.rows)
                     if (x, y) not in snake_body and (x, y) not in walls]
        self.position = random.choice(available)

