# main.py
from game import Game


def select_difficulty():
    print("Select difficulty: 1) Easy 2) Normal 3) Hard")
    choice = input()
    return {'1': 0, '2': 15, '3': 45}.get(choice, 0)


if __name__ == '__main__':
    wall_count = select_difficulty()
    game = Game(wall_count=wall_count)
    game.run(wall_count)
