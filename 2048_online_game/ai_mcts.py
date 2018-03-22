from __future__ import division

from game import *
import time
import os
import pickle
import math
import matplotlib.pyplot as plt


class Node(object):
    "Generic tree node."
    def __init__(self, board, parent=None, action=None, score=0):
        self.name = state_to_string(board)
        self.parent = parent
        self.action = action
        self.score = score
        self.children = {
            'left': [],
            'right': [],
            'up': [],
            'down': [],
        }
        self.Q = {
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0
        }
        self.N = {
            'left': 0,
            'right': 0,
            'up': 0,
            'down': 0
        }
        self.board = board

    def __str__(self):
        return self.name


def state_to_string(state):
    return '|'.join(str(state[j][i]) for j in range(4) for i in range(4))


def string_to_state(state):
    elements = [int(x) for x in state.split('|')]
    return np.array(elements).reshape((4, 4))


def next_state(state, direction):
    # Takes the game state, and the move to be applied.
    # Returns the new game state.

    new_state = np.copy(state)

    score = direction_dict[direction](new_state)

    empty = prepare_next_turn(new_state)

    if not empty:
        return None, -1

    return new_state, score


class MonteCarlo(object):
    def __init__(self, **kwargs):
        # Takes an instance of a Board and optionally some keyword
        # arguments.  Initializes the list of game states and the
        # statistics tables.
        self.Q = {}
        self.N = {}
        self.C = kwargs.get('C', math.sqrt(2))
        self.max_moves = max_moves  # number of trajectory
        self.sample_width = sample_width
        self.actions = ['left', 'down', 'up', 'right']

    def get_play(self, state):
        # Causes the AI to calculate the best move from the
        # current game state and return it.

        s_0 = Node(state)

        sample_width = self.sample_width
        max_moves = self.max_moves

        def best_action(s, final=False):
            # select best child

            log_total = np.log(sum(s.N[a] for a in self.actions))

            scores = []
            for a in self.actions:
                try:
                    score = ((s.Q[a] * 1. / s.N[a]) + self.C * np.sqrt(log_total / s.N[a]))
                    scores.append(score)
                except ZeroDivisionError:
                    scores.append(-1)

            a = self.actions[np.argmax(scores)]

            return a

        def run_simulate(s):
            s0 = s

            for k in range(max_moves):
                s = s0
                a = choice(self.actions)

                while any_possible_moves(s.board):  # and np.max(s.board) < 4096:

                    if all(len(s.children[a]) >= sample_width for a in self.actions):
                        a = best_action(s)
                    else:
                        # expand more
                        unsampled_actions = [a for a in self.actions if len(s.children[a]) < sample_width]

                        # change here for weighted matrix or heuristic
                        a = choice(unsampled_actions)

                    if len(s.children[a]) == sample_width:
                        s1 = choice(s.children[a])
                    else:
                        s1 = next_state(s.board, a)

                        if s1[0] is None:
                            break

                        s1 = Node(s1[0], parent=s, action=a, score=s1[1]+s.score)

                        s.children[a].append(s1)

                    s = s1
                    pass

                # back-propagation
                delta = s.score
                while s is not None:
                    s.N[a] += 1
                    s.Q[a] += delta
                    s, a = s.parent, s.action

        run_simulate(s_0)

        return best_action(s_0, True)


def aiplay(game_id):
    game = Game()
    mcts_ai = MonteCarlo()
    game.show()

    tic = time.time()
    while not game.over:
        m = mcts_ai.get_play(game.board)
        game.move(m)

        if debug:
            game.show()
            print "max_moves:", max_moves, game_id, '----->direction:', \
                m, '----->current score:', game.score, '---->max tile:', np.max(game.board)

    elapsed_time = time.time() - tic
    score = game.score
    max_tile = np.max(game.board)

    return game.board, score, max_tile, elapsed_time


if __name__ == "__main__":
    # max_moves is number of trajectory

    max_moves = 100  # change this parameter for different values from 100 to 500 (bigger will slower but higher score)
    # change this parameter will have different result thereafter

    sample_width = 20
    file_save = 'data/ai_mcts_max_moves_' + str(max_moves) + '.hkl'

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
    plt.title('ai mcts Ellapsed with max move ' + str(max_moves))
    plt.scatter(range(len(list_elapsed_time)), list_elapsed_time)
    plt.savefig('figures/time_ai_mcts_max_move_' + str(max_moves))

    plt.figure()
    plt.title('ai mcts score with max move ' + str(max_moves))
    plt.scatter(range(len(list_score)), list_score)
    plt.savefig('figures/score_ai_mcts_max_move_' + str(max_moves))

    plt.figure()
    plt.title('ai mcts Max Tile with max move ' + str(max_moves))
    plt.scatter(range(len(list_max_tile)), list_max_tile)
    plt.savefig('figures/max_tile_ai_mcts_max_move_' + str(max_moves))

    print "--------------Summary------------"
    print "score of 100 tries:", list_score
    print "max tile of 100 tries:", list_max_tile
    print "time of 100 tries:", list_elapsed_time
    print "Average score:", sum(list_score) / len(list_score)
    print "Average Max Tile:", sum(list_max_tile) / len(list_max_tile)
    print "Average Time:", sum(list_elapsed_time) / len(list_elapsed_time)
    print "Average Time:", sum(list_elapsed_time) / len(list_elapsed_time)
