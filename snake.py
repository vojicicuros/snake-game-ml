# snake.py
import random
from settings import Settings


def opposite_dir(a, b):
    return a[0] == -b[0] and a[1] == -b[1]


class Snake:
    def __init__(self, cols, rows):
        self.grow = None
        self.direction = None
        self.body = None
        self.reset(cols, rows)

    def reset(self, cols, rows):
        mid = (cols // 2, rows // 2)
        self.body = [mid]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.grow = False

    def set_direction(self, new_dir):
        if not opposite_dir(new_dir, self.direction):
            self.direction = new_dir

    def move(self, cols, rows):
        head = self.body[0]
        new_head = ((head[0] + self.direction[0]) % cols,
                    (head[1] + self.direction[1]) % rows)
        if self.grow:
            self.body.insert(0, new_head)
            self.grow = False
        else:
            self.body.insert(0, new_head)
            self.body.pop()
        return new_head

    def eat(self):
        self.grow = True

    def hits_self(self):
        return self.body[0] in self.body[1:]

