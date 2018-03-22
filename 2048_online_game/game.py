import random
random.seed(1991)

from random import random, randint, choice
import numpy as np


def push_left(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):
        i, last = 0, 0
        for j in range(columns):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i - 1] += e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[k, i] = e
                    i += 1
        while i < columns:
            grid[k, i] = 0
            i += 1
    return score if moved else -1


def push_right(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):
        i = columns - 1
        last = 0
        for j in range(columns - 1, -1, -1):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i + 1] += e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[k, i] = e
                    i -= 1
        while 0 <= i:
            grid[k, i] = 0
            i -= 1
    return score if moved else -1


def push_up(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = 0, 0
        for j in range(rows):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i - 1, k] += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[i, k] = e
                    i += 1
        while i < rows:
            grid[i, k] = 0
            i += 1
    return score if moved else -1


def push_down(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = rows - 1, 0
        for j in range(rows - 1, -1, -1):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i + 1, k] += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[i, k] = e
                    i -= 1
        while 0 <= i:
            grid[i, k] = 0
            i -= 1
    return score if moved else -1


def push(grid, direction):
    if direction & 1:
        if direction & 2:
            score = push_down(grid)
        else:
            score = push_up(grid)
    else:
        if direction & 2:
            score = push_right(grid)
        else:
            score = push_left(grid)
    return score


def put_new_cell(grid):
    n = 0
    r = 0
    i_s = [0] * 16
    j_s = [0] * 16
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i, j]:
                i_s[n] = i
                j_s[n] = j
                n += 1
    if n > 0:
        r = randint(0, n - 1)
        grid[i_s[r], j_s[r]] = 2 if random() < 0.9 else 4
    return n


def any_possible_moves(grid):
    """Return True if there are any legal moves, and False otherwise."""
    rows = grid.shape[0]
    columns = grid.shape[1]
    for i in range(rows):
        for j in range(columns):
            e = grid[i, j]
            if not e:
                return True
            if j and e == grid[i, j - 1]:
                return True
            if i and e == grid[i - 1, j]:
                return True
    return False


def prepare_next_turn(grid):
    """
    Spawn a new number on the grid; then return the result of
    any_possible_moves after this change has been made.
    """
    empties = put_new_cell(grid)
    any_move = any_possible_moves(grid)

    return empties or any_move


direction_dict = {'left': push_left, 'up': push_up, 'right': push_right, 'down': push_down}


class Game:
    def __init__(self, cols=4, rows=4):
        self.grid_array = np.zeros(shape=(rows, cols), dtype='uint16')
        self.board = self.grid_array
        for i in range(2):
            put_new_cell(self.board)
        self.score = 0
        self.over = False

    def move(self, direction):

        score = direction_dict[direction](self.board)

        self.score += max(0, score)

        next_turn = prepare_next_turn(self.board)

        if not next_turn:
            self.over = True

    def next_state(self, state, direction):
        direction_dict = {'left': push_left, 'up': push_up, 'right': push_right, 'down': push_down}
        direction_dict[direction](state)

        next_turn = prepare_next_turn(state)

        if not next_turn:
            self.over = True
        self.move(direction)
        return self.to_state()

    def show(self):
        for i in range(4):
            for j in range(4):
                if self.board[i][j]:
                    print '%4d' % self.board[i][j],
                else:
                    print '   .',
            print

