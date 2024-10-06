# 2048 AI Bot with Dynamic Depth and Expectimax Algorithm

This Python script implements an AI bot to play the game 2048 using the Expectimax algorithm with dynamic search depth. The AI adjusts its search depth based on the current state of the board using a combination of several parameters, such as the number of empty tiles, smoothness, monotonicity, and the maximum tile. The script also uses multiprocessing to parallelize the search for optimal moves, making the AI both efficient and powerful.

## Goal

The goal of this project is to build an AI that can autonomously play the 2048 game and optimize its moves to reach the highest possible score, ideally reaching the 2048 tile or beyond. To achieve this, the AI needs to balance between exploration and exploitation by dynamically adjusting the depth of the Expectimax search algorithm based on the game state. The deeper the search, the better the AI can predict future board configurations, but this comes at the cost of higher computational load.

## Features

- **Expectimax Search Algorithm**: The AI uses Expectimax to evaluate moves, taking into account both the player's actions and the randomness of new tiles appearing on the board.
- **Dynamic Depth Adjustment**: The depth of the search dynamically adjusts based on the game state, ensuring that the AI remains efficient early in the game and more precise when the board becomes more crowded.
- **Multiprocessing**: The script leverages Python's multiprocessing library to parallelize the computation of multiple possible moves, improving performance.
- **State Evaluation**: The AI evaluates the board based on multiple factors, including the number of empty tiles, smoothness of the board (how similar adjacent tiles are), monotonicity (whether tiles increase or decrease consistently), and the maximum tile on the board.

## Workflow

### 1. **Game Initialization**
   - The `initialize_board` function sets up a 4x4 grid with two randomly placed tiles, each being either a 2 or a 4, based on specified probabilities.
   - Random tiles are added to the board after every move using the `add_random_tile` function.

### 2. **Move Execution**
   - The game supports four types of moves: up, down, left, and right.
   - Each move is executed by shifting and merging tiles according to the game rules. The movement functions (`move_left`, `move_right`, `move_up`, `move_down`) handle the sliding and merging of tiles.

### 3. **Board Evaluation**
   - The AI uses a state evaluation function to assess the quality of the board. The evaluation function incorporates:
     - **Number of empty tiles**: More empty tiles are beneficial as they provide room for further moves.
     - **Maximum tile**: Higher maximum tiles indicate a stronger board position.
     - **Smoothness**: The difference between adjacent tiles, where smoother boards (smaller differences) are better.
     - **Monotonicity**: Measures whether the tiles increase or decrease consistently along rows or columns.
   - The evaluation function is called during the Expectimax search to compare different board configurations.

### 4. **Expectimax Algorithm**
   - The core of the AI is the Expectimax algorithm, which evaluates possible moves by simulating both the player's moves and the random tile placements.
   - **Max nodes** represent the player's moves, and the algorithm tries to maximize the expected score.
   - **Chance nodes** simulate the random appearance of new tiles (2 or 4) and calculate the expected value based on all possible future states.
   - The algorithm recursively searches up to a certain depth and returns the best possible move.

### 5. **Dynamic Depth Calculation**
   - To balance computational efficiency and move precision, the depth of the Expectimax search is adjusted dynamically based on the game state.
   - The `calculate_composite_score` function normalizes several game parameters (empty tiles, maximum tile, possible moves, smoothness, monotonicity, move count) and computes a weighted score.
   - The depth is then calculated based on this composite score, ensuring that deeper searches occur when the board is more complex and precise moves are necessary.

### 6. **Parallelized Move Evaluation**
   - The script uses Python's multiprocessing to parallelize the evaluation of different moves.
   - The `get_best_move` function generates a list of possible moves, and each move is processed in parallel using the `multiprocessing.Pool`. This speeds up the decision-making process, especially when evaluating multiple possible future board states.

### 7. **Game Loop**
   - The `play_game` function runs the game loop, where the AI continuously makes moves based on the current board state.
   - After each move, the board is updated, a new tile is added, and the move count is incremented.
   - The game ends when no valid moves are available, or when the AI reaches the 2048 tile (or another predefined target score).

## Key Functions

### **Dynamic Depth Adjustment**
- `calculate_composite_score(board, move_count)`: Computes a composite score based on several game parameters.
- `calculate_dynamic_depth(board, move_count, min_depth=3, max_depth=8)`: Dynamically calculates the depth of the Expectimax search based on the composite score.

### **Game Execution**
- `play_game()`: The main game loop that continuously updates the board and makes decisions based on the current state.

## How to Run

1. **Install Required Libraries**:
   Make sure you have the following libraries installed:
   ```bash
   pip install numpy
   python expectimax.py
