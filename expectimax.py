import numpy as np
import random
import multiprocessing

MOVES = ('up', 'down', 'left', 'right')
MAX_SCORE = 2048
FOUR_TILE_PROBABILITY = 0
TWO_TILE_PROBABILITY = 1 - FOUR_TILE_PROBABILITY


def initialize_board():
    board = np.zeros((4, 4), dtype=int)
    board = add_random_tile(board)
    board = add_random_tile(board)
    return board


def add_random_tile(board):
    empty_cells = list(zip(*np.where(board == 0)))
    if empty_cells:
        i, j = random.choice(empty_cells)
        board[i][j] = 2 if random.random() < TWO_TILE_PROBABILITY else 4
    return board


def move_left(board):
    new_board = np.zeros((4, 4), dtype=int)
    for i in range(4):
        row = board[i][board[i] != 0]  # Remove zeros
        new_row = []
        skip = False
        for j in range(len(row)):
            if skip:
                skip = False
                continue
            if j + 1 < len(row) and row[j] == row[j + 1]:
                new_row.append(row[j] * 2)
                skip = True
            else:
                new_row.append(row[j])
        new_board[i, :len(new_row)] = new_row
    return new_board


def move_right(board):
    flipped_board = np.fliplr(board)
    new_board = move_left(flipped_board)
    return np.fliplr(new_board)


def move_up(board):
    rotated_board = np.rot90(board, -1)
    new_board = move_left(rotated_board)
    return np.rot90(new_board)


def move_down(board):
    rotated_board = np.rot90(board)
    new_board = move_left(rotated_board)
    return np.rot90(new_board, -1)


def is_terminal(board):
    if np.any(board == 0):
        return False
    for move in MOVES:
        if not np.array_equal(board, execute_move(move, board)):
            return False
    return True


def execute_move(move, board):
    if move == 'up':
        return move_up(board)
    elif move == 'down':
        return move_down(board)
    elif move == 'left':
        return move_left(board)
    elif move == 'right':
        return move_right(board)
    return board


def evaluation_function(board):
    empty_tiles = len(np.where(board == 0)[0])
    max_tile = np.max(board)
    smoothness = calculate_smoothness(board)
    monotonicity = calculate_monotonicity(board)
    return empty_tiles + np.log(max_tile) + smoothness + monotonicity


def calculate_smoothness(board):
    smoothness = 0
    for i in range(4):
        for j in range(3):
            if board[i][j] != 0 and board[i][j + 1] != 0:
                smoothness -= abs(board[i][j] - board[i][j + 1])
            if board[j][i] != 0 and board[j + 1][i] != 0:
                smoothness -= abs(board[j][i] - board[j + 1][i])
    return smoothness


def calculate_monotonicity(board):
    totals = [0, 0, 0, 0]
    for i in range(4):
        current = 0
        next = current + 1
        while next < 4:
            while next < 4 and board[i][next] == 0:
                next += 1
            if next >= 4:
                next -= 1
            current_value = board[i][current]
            next_value = board[i][next]
            if current_value > next_value:
                totals[0] += next_value - current_value
            elif current_value < next_value:
                totals[1] += current_value - next_value
            current = next
            next += 1
    for j in range(4):
        current = 0
        next = current + 1
        while next < 4:
            while next < 4 and board[next][j] == 0:
                next += 1
            if next >= 4:
                next -= 1
            current_value = board[current][j]
            next_value = board[next][j]
            if current_value > next_value:
                totals[2] += next_value - current_value
            elif current_value < next_value:
                totals[3] += current_value - next_value
            current = next
            next += 1
    return max(totals[0], totals[1]) + max(totals[2], totals[3])


def expectimax(board, depth, player):
    if depth == 0 or is_terminal(board):
        return evaluation_function(board)

    if player == 'max':
        max_value = float('-inf')
        for move in MOVES:
            new_board = execute_move(move, board)
            if not np.array_equal(new_board, board):
                value = expectimax(new_board, depth - 1, 'chance')
                max_value = max(max_value, value)
        return max_value
    else:
        expected_value = 0
        empty_cells = list(zip(*np.where(board == 0)))
        if not empty_cells:
            return evaluation_function(board)
        probability = 1 / len(empty_cells)
        for cell in empty_cells:
            for tile_value, tile_probability in [(2, TWO_TILE_PROBABILITY), (4, FOUR_TILE_PROBABILITY)]:
                if tile_probability == 0:
                    continue
                new_board = board.copy()
                new_board[cell] = tile_value
                value = expectimax(new_board, depth - 1, 'max')
                expected_value += tile_probability * probability * value
        return expected_value


def moves(*args):
    move, board, depth = args[0]
    new_board = execute_move(move, board)
    value = float('-inf')
    if not np.array_equal(new_board, board):
        value = expectimax(new_board, depth - 1, 'chance')

    return move, value


def process_move(args):
    move, board, depth = args
    value = expectimax(board, depth, 'chance')
    return move, value


def calculate_composite_score(board, move_count):
    def normalize_empty_tiles(empty_tiles, min_empty=0, max_empty=16):
        return (max_empty - empty_tiles) / (max_empty - min_empty)

    def normalize_max_tile(max_tile, min_tile=2):
        max_tile_log = np.log2(max_tile)
        max_tile_expected_log = np.log2(MAX_SCORE)
        return (max_tile_log - np.log2(min_tile)) / (max_tile_expected_log - np.log2(min_tile))

    def normalize_possible_moves(possible_moves, min_moves=0, max_moves=4):
        return (max_moves - possible_moves) / (max_moves - min_moves)

    def normalize_smoothness(smoothness, min_smooth=-2000, max_smooth=0):
        return (max_smooth - smoothness) / (max_smooth - min_smooth)

    def normalize_monotonicity(monotonicity, min_mono=-2000, max_mono=0):
        return (max_mono - monotonicity) / (max_mono - min_mono)

    def normalize_move_count(move_count, min_moves=0, max_moves_expected=100):
        return (move_count - min_moves) / (max_moves_expected - min_moves)

    weights = {
        'empty_tiles': 0.3,
        'max_tile': 0.2,
        'possible_moves': 0.2,
        'smoothness': 0.1,
        'monotonicity': 0.1,
        'move_count': 0.1,
    }

    empty_tiles = len(np.where(board == 0)[0])
    max_tile = np.max(board)
    possible_moves = sum(
        not np.array_equal(board, execute_move(move, board))
        for move in ['up', 'down', 'left', 'right']
    )
    smoothness = calculate_smoothness(board)
    monotonicity = calculate_monotonicity(board)

    # Normalize parameters
    empty_tiles_norm = normalize_empty_tiles(empty_tiles)
    max_tile_norm = normalize_max_tile(max_tile)
    possible_moves_norm = normalize_possible_moves(possible_moves)
    smoothness_norm = normalize_smoothness(smoothness)
    monotonicity_norm = normalize_monotonicity(monotonicity)
    move_count_norm = normalize_move_count(move_count)

    return (
        weights['empty_tiles'] * empty_tiles_norm +
        weights['max_tile'] * max_tile_norm +
        weights['possible_moves'] * possible_moves_norm +
        weights['smoothness'] * smoothness_norm +
        weights['monotonicity'] * monotonicity_norm +
        weights['move_count'] * move_count_norm
    )


def calculate_dynamic_depth(board, move_count, min_depth=3, max_depth=6):
    composite_score = calculate_composite_score(board, move_count)

    # Depth increases as composite score increases
    depth = min_depth + composite_score * (max_depth - min_depth)
    depth = int(round(depth))
    depth = max(min_depth, min(depth, max_depth))

    return depth


def get_best_move(board, depth):
    print(f'Getting best move with depth {depth}...')
    args_list = []
    for move in MOVES:
        new_board = execute_move(move, board)
        if not np.array_equal(new_board, board):
            args_list.append((move, new_board, depth))
    if not args_list:
        return None

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_move, args_list)

    best_move = None
    max_value = float('-inf')
    for move, value in results:
        if value > max_value:
            max_value = value
            best_move = move
    return best_move


def play_game():
    board = initialize_board()
    move_count = 0
    while True:
        print(f"Move #{move_count}")
        score = np.max(board)
        if score >= MAX_SCORE:
            print(f"{MAX_SCORE} reached in {move_count} moves!")
            print(board)
            break
        print(f'Score: {score}')
        print(board)
        depth = calculate_dynamic_depth(board, move_count)
        move = get_best_move(board, depth)
        if move is None:
            print("Game Over!")
            print(f"Final Score: {np.max(board)}")
            break
        board = execute_move(move, board)
        board = add_random_tile(board)
        move_count += 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    play_game()
