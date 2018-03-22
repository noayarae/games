from game import *

import itertools
import time
import pickle
import os
import copy
import math
import matplotlib.pyplot as plt


def merge_right(b):
    """
    Merge the board right
    Args: b (list) two dimensional board to merge
    Returns: list
    # >>> merge_right(test)
    [[0, 0, 2, 8], [0, 2, 4, 8], [0, 0, 0, 4], [0, 0, 4, 4]]
    """

    def reverse(x):
        return list(reversed(x))

    t = map(reverse, b)
    return [reverse(x) for x in merge_left(t)]


def merge_up(b):
    """
    Merge the board upward. Note that zip(*t) is the
    transpose of b
    Args: b (list) two dimensional board to merge
    Returns: list
    # >>> merge_up(test)
    [[2, 4, 8, 4], [0, 2, 2, 8], [0, 0, 0, 4], [0, 0, 0, 2]]
    """

    t = merge_left(zip(*b))
    return [list(x) for x in zip(*t)]


def merge_down(b):
    """
    Merge the board downward. Note that zip(*t) is the
    transpose of b
    Args: b (list) two dimensional board to merge
    Returns: list
    # >>> merge_down(test)
    [[0, 0, 0, 4], [0, 0, 0, 8], [0, 2, 8, 4], [2, 4, 2, 2]]
    """

    t = merge_right(zip(*b))
    return [list(x) for x in zip(*t)]


def merge_left(b):
    """
    Merge the board left
    Args: b (list) two dimensional board to merge
    Returns: list
    """

    def merge(row, acc):
        """
        Recursive helper for merge_left. If we're finished with the list,
        nothing to do; return the accumulator. Otherwise, if we have
        more than one element, combine results of first from the left with right if
        they match. If there's only one element, no merge exists and we can just
        add it to the accumulator.
        Args:
            row (list) row in b we're trying to merge
            acc (list) current working merged row
        Returns: list
        """

        if not row:
            return acc

        x = row[0]
        if len(row) == 1:
            return acc + [x]

        return merge(row[2:], acc + [2 * x]) if x == row[1] else merge(row[1:], acc + [x])

    board = []
    for row in b:
        merged = merge([x for x in row if x != 0], [])
        merged = merged + [0] * (len(row) - len(merged))
        board.append(merged)
    return board


def move_exists(b):
    """
    Check whether or not a move exists on the board
    Args: b (list) two dimensional board to merge
    Returns: list
    # >>> b = [[1, 2, 3, 4], [5, 6, 7, 8]]
    # >>> move_exists(b)
    False
    # >>> move_exists(test)
    True
    """
    for row in b:
        for x, y in zip(row[:-1], row[1:]):
            if x == y or x == 0 or y == 0:
                return True
    return False


MERGE_FUNCTIONS = {
    'left': merge_left,
    'right': merge_right,
    'up': merge_up,
    'down': merge_down
}


def aimove(b):
    """
    Evaluate the utility of each of the four possible moves
    we can make on b
    Args: b (list) root board to score
    Returns: list
    """

    def fitness(b):
        """
        Returns the heuristic value of b
        1. granting "bonuses" for open squares and for having large values on the edge.

        2. The first heuristic was a penalty for having non-monotonic rows and columns which increased as the ranks
        increased, ensuring that non-monotonic rows of small numbers would not strongly affect the score, but
        non-monotonic rows of large numbers hurt the score substantially.

        3. The second heuristic counted the number of potential merges (adjacent equal values)
        Args: b (list) board to score
        Returns: float
        """

        board = np.array(b)
        log2 = int(np.log2(np.max(board)))

        num_free_space = 16 - np.count_nonzero(board)
        free_score = num_free_space

        edge_score = 0
        for r in range(4):
            for c in range(4):
                edge_score += board[r, c] * ((r == 0) + (r == 3) + (c == 0) + (c == 3))

        loss = 0
        merge_score = 0
        # check column:
        temp = np.diff(board)
        for i, r in enumerate(temp):
            if not (np.all(r >= 0) or np.all(r <= 0)):
                loss += np.sum(board[i, :])
            for j, e in enumerate(r):  # for potential merge
                if e == 0:
                    merge_score += board[i, j]

        # check row
        temp = np.diff(board.T)
        for i, r in enumerate(temp):
            if not (np.all(r >= 0) or np.all(r <= 0)):
                loss += np.sum(board.T[i, :])
            for j, e in enumerate(r):  # for potential merge
                if e == 0:
                    merge_score += board.T[i, j]

        snake_score = 0
        snake = []
        for i, col in enumerate(zip(*board)):
            snake.extend(reversed(col) if i % 2 == 0 else col)

        m = max(snake)
        snake_score += sum(x / 10 ** n for n, x in enumerate(snake)) - \
                       math.pow((board[3, 0] != m) * abs(board[3, 0] - m), 2)

        # print score_free, score, merge_score, snake_score, loss

        return (free_score + edge_score + merge_score) * log2 - loss + snake_score

    def search(b, d, move=False):
        """
        Performs expectimax search on a given configuration to
        specified depth (d).
        Algorithm details:
           - if the AI needs to move, make each child move,
             recurse, return the maximum fitness value
           - if it is not the AI's turn, form all
             possible child spawns, and return their weighted average
             as that node's evaluation
        Args:
            b (list) board to search
            d (int) depth to serach to
            move (bool) whether or not it's our (AI player's) move to make
        Returns: float
        """

        if d == 0 or (move and not move_exists(b)):
            return fitness(b)

        if move:
            alpha = 0

            for _, action in MERGE_FUNCTIONS.items():
                child = action(b)
                alpha = max(alpha, search(child, d - 1))

        else:
            alpha = 0

            zeros = [(i, j) for i, j in itertools.product(range(4), range(4)) if b[i][j] == 0]
            scores = []
            for i, j in zeros:
                c1 = copy.copy(b)
                c2 = copy.copy(b)
                c1[i][j] = 2
                c2[i][j] = 4
                scores.append(.9 * fitness(c1) + .1 * fitness(c2))

            if scores:
                choose_i, choose_j = zeros[np.argmin(scores)]
                c1 = copy.copy(b)
                c2 = copy.copy(b)
                c1[choose_i][choose_j] = 2
                c2[choose_i][choose_j] = 4

                alpha += .9 * search(c1, d - 1, True) + .1 * search(c2, d - 1, True)

        return alpha

    results = []

    for direction, action in MERGE_FUNCTIONS.items():
        result = direction, search(action(b), depth_for_search)
        results.append(result)
    return results


def aiplay(game_id):
    game = Game()
    game.show()
    tic = time.time()
    while not game.over:
        m = max(aimove(game.board), key=lambda x: x[1])[0]
        game.move(m)

        if debug:
            game.show()
            print 'depth_for_search:', depth_for_search, game_id, '----->direction:', m, \
                '----->current score:', game.score, '---->max tile:', np.max(game.board)

    elapsed_time = time.time() - tic
    score = game.score
    max_tile = np.max(game.board)

    return game.board, score, max_tile, elapsed_time


if __name__ == "__main__":

    depth_for_search = 4  # Chang this parameter will have different result thereafter, bigger will slower but better
    # may be from 2 to 10 (even number)

    file_save = 'data/ai_expectimax_depth_for_search_' + str(depth_for_search) + '.hkl'

    if os.path.isfile(file_save):
        with open(file_save, 'r') as f:
            list_game_board, list_score, list_max_tile, list_elapsed_time = pickle.load(f)
    else:
        list_game_board = []
        list_elapsed_time = []
        list_score, list_max_tile = [], []
        debug = True  # will control whether printout the process or not

        for i in range(100):
            game_board, score, max_tile, elapsed_time = aiplay(i)
            list_game_board.append(game_board)
            print i, "Total score:", score, "Max Tile:", max_tile
            list_score.append(score)
            list_max_tile.append(max_tile)
            list_elapsed_time.append(elapsed_time)

        with open(file_save, 'w') as f:
            pickle.dump((list_game_board, list_score, list_max_tile, list_elapsed_time), f)

    plt.figure()
    plt.title('ai expectimax Ellapsed with depth for search ' + str(depth_for_search))
    plt.scatter(range(len(list_elapsed_time)), list_elapsed_time)
    plt.savefig('figures/time_ai_expectimax_depth_' + str(depth_for_search))

    plt.figure()
    plt.title('ai expectimax score with depth for search ' + str(depth_for_search))
    plt.scatter(range(len(list_score)), list_score)
    plt.savefig('figures/score_ai_expectimax_depth_' + str(depth_for_search))

    plt.figure()
    plt.title('ai expectimax Max Tile with depth for search ' + str(depth_for_search))
    plt.scatter(range(len(list_max_tile)), list_max_tile)
    plt.savefig('figures/max_tile_ai_expectimax_depth_' + str(depth_for_search))

    print "--------------Summary------------"
    print "score of 100 tries:", list_score
    print "max tile of 100 tries:", list_max_tile
    print "time of 100 tries:", list_elapsed_time
    print "Average score:", sum(list_score) / len(list_score)
    print "Average Max Tile:", sum(list_max_tile) / len(list_max_tile)
    print "Average Time:", sum(list_elapsed_time) / len(list_elapsed_time)
