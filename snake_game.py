import pygame
import random
import sys
import os

# Initialize Pygame
pygame.init()

# Set up the screen
width, height = 600, 400
cell_size = 20
rows = height // cell_size
cols = width // cell_size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

# Set up colors
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
grey = (50,50,50)

# Set up fonts
font = pygame.font.Font(None, 24)

# Set up the snake, food, and wall positions
snake = [(cols // 2, rows // 2)]
random_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
snake_dir = random.choice(random_directions)
snake_speed = 5
food = (random.randint(0, cols - 1), random.randint(0, rows - 1))
walls = set()

# Input cooldown
input_cooldown = 100  # Initial input cooldown
cooldown_time = 100  # milliseconds

# Difficulty levels
difficulty_levels = {
    "easy": {"walls": False, "speed_multiplier": 1},
    "normal": {"walls": True, "speed_multiplier": 1},
    "hard": {"walls": True, "speed_multiplier": 2}
}

# Selected difficulty
selected_difficulty = None

# Load best record from file
best_record_file = "best_record.txt"
best_record = 0
if os.path.exists(best_record_file):
    with open(best_record_file, "r") as file:
        best_record = int(file.read())

# Main game loop
running = False  # Initially not running
clock = pygame.time.Clock()

def draw_menu():
    screen.fill(grey)
    start_text = font.render("Select difficulty:", True, white)
    easy_text = font.render("Easy - Press 1", True, white)
    normal_text = font.render("Normal - Press 2", True, white)
    hard_text = font.render("Hard - Press 3", True, white)
    best_record_text = font.render(f"Best Record: {best_record}", True, white)
    start_text_rect = start_text.get_rect(center=(width//2, height//2 - 60))
    easy_text_rect = easy_text.get_rect(center=(width//2, height//2 - 20))
    normal_text_rect = normal_text.get_rect(center=(width//2, height//2 + 20))
    hard_text_rect = hard_text.get_rect(center=(width//2, height//2 + 60))
    best_record_text_rect = best_record_text.get_rect(center=(width//2, height//2 + 100))
    screen.blit(start_text, start_text_rect)
    screen.blit(easy_text, easy_text_rect)
    screen.blit(normal_text, normal_text_rect)
    screen.blit(hard_text, hard_text_rect)
    screen.blit(best_record_text, best_record_text_rect)

def draw_matrix(matrix):
    for y in range(len(matrix)):
        for x in range(len(matrix[0])):
            color = black
            if matrix[y][x] == 1:
                # Draw snake head in dark red color
                if (x, y) == snake[0]:
                    color = (150, 0, 0)  # Dark red color
                else:
                    color = green
            elif matrix[y][x] == 2:
                color = red
            elif matrix[y][x] == 3:
                color = blue
            # Draw main rectangle
            pygame.draw.rect(screen, color, (x * cell_size, y * cell_size, cell_size, cell_size))
            # Draw black border
            pygame.draw.rect(screen, black, (x * cell_size, y * cell_size, cell_size, cell_size), 1)


def update_matrix():
    matrix = [[0] * cols for _ in range(rows)]
    for segment in snake:
        if 0 <= segment[0] < cols and 0 <= segment[1] < rows:
            matrix[segment[1]][segment[0]] = 1
    matrix[food[1]][food[0]] = 2
    for wall in walls:
        if 0 <= wall[0] < cols and 0 <= wall[1] < rows:
            matrix[wall[1]][wall[0]] = 3
    return matrix

def move_snake():
    global snake, snake_speed, input_cooldown
    new_head = ((snake[0][0] + snake_dir[0]) % cols, (snake[0][1] + snake_dir[1]) % rows)
    if new_head in walls or new_head in snake[1:]:
        update_best_record(len(snake))  # Update best record if necessary
        end_game()
        return
    snake.insert(0, new_head)
    if new_head == food:
        generate_food()
        snake_speed += 0.2 * selected_difficulty["speed_multiplier"]  # Increase speed
        input_cooldown -= 5  # Decrease input cooldown
    else:
        snake.pop()

def generate_food():
    global food
    food = (random.randint(0, cols - 1), random.randint(0, rows - 1))
    while food in snake or food in walls:
        food = (random.randint(0, cols - 1), random.randint(0, rows - 1))

def generate_walls():
    global walls
    if selected_difficulty == difficulty_levels["hard"]:
        num_walls = random.randint(45, 60)  # Adjusted range for hard difficulty
    else:
        num_walls = random.randint(15, 25)   # Default range for other difficulties
    walls = set()
    for _ in range(num_walls):
        # Generate random wall position excluding the edges
        wall = (random.randint(1, cols - 2), random.randint(1, rows - 2))
        walls.add(wall)
    
    # Check proximity to snake and move walls if necessary
    for wall in walls.copy():  # Iterate over a copy to avoid modifying the set while iterating
        if min(abs(snake_segment[0] - wall[0]) + abs(snake_segment[1] - wall[1]) for snake_segment in snake) < 4:
            # If wall is too close to snake, move it to a new position
            walls.remove(wall)
            new_wall = (random.randint(1, cols - 2), random.randint(1, rows - 2))
            while new_wall in snake or new_wall == food or new_wall in walls:
                new_wall = (random.randint(1, cols - 2), random.randint(1, rows - 2))
            walls.add(new_wall)


def update_best_record(score):
    global best_record
    if score > best_record:
        best_record = score
        with open(best_record_file, "w") as file:
            file.write(str(best_record))

def end_game():
    global snake, snake_dir, snake_speed, food, walls
    snake = [(cols // 2, rows // 2)]
    snake_dir = random.choice(random_directions)
    snake_speed = 5
    food = (random.randint(0, cols - 1), random.randint(0, rows - 1))
    if selected_difficulty["walls"]:
        generate_walls()
    generate_food()


# Main menu loop
while not running:
    draw_menu()
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                selected_difficulty = difficulty_levels["easy"]
                running = True
            elif event.key == pygame.K_2:
                selected_difficulty = difficulty_levels["normal"]
                running = True
            elif event.key == pygame.K_3:
                selected_difficulty = difficulty_levels["hard"]
                running = True

# Generate initial food and walls based on the selected difficulty
if selected_difficulty["walls"]:
    generate_walls()
generate_food()



# Game loop
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if input_cooldown <= 0:
                if event.key == pygame.K_UP and snake_dir != (0, 1):
                    snake_dir = (0, -1)
                    input_cooldown = cooldown_time
                elif event.key == pygame.K_DOWN and snake_dir != (0, -1):
                    snake_dir = (0, 1)
                    input_cooldown = cooldown_time
                elif event.key == pygame.K_LEFT and snake_dir != (1, 0):
                    snake_dir = (-1, 0)
                    input_cooldown = cooldown_time
                elif event.key == pygame.K_RIGHT and snake_dir != (-1, 0):
                    snake_dir = (1, 0)
                    input_cooldown = cooldown_time
                elif event.key == pygame.K_q:  # Quit game when 'q' is pressed
                    pygame.quit()
                    sys.exit()

    # Move the snake
    move_snake()

    # Clear the screen
    screen.fill(black)

    # Update and draw the matrix
    matrix = update_matrix()
    draw_matrix(matrix)

    # Draw snake length counter
    snake_length_text = font.render(f"Snake Length: {len(snake)}", True, white)
    screen.blit(snake_length_text, (width - 180, 10))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(snake_speed)

    # Decrease input cooldown
    if input_cooldown > 0:
        input_cooldown -= clock.get_time()

# Quit Pygame
pygame.quit()
sys.exit()  # Ensure proper exit