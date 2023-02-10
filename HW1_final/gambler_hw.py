import random

from enums import *


class GamblerHW:
    def __init__(self, learning_rate=0.1, discount=0.95, exploration_rate=1.0, iterations=10000):
        self.q_table = [[0,0,0,0,0], [0,0,0,0,0]] # Spreadsheet (Q-table) for rewards accounting
        self.learning_rate = learning_rate # How much we appreciate new q-value over current
        self.discount = discount # How much we appreciate future reward over current
        self.exploration_rate = exploration_rate # Initial exploration rate
        self.exploration_delta = 1.0 / iterations # Shift from exploration to explotation
        self.iteration_count = 0 # Count the number of iterations

    def get_next_action(self, state):
        if random.random() > self.exploration_rate: # Explore (gamble) or exploit (greedy)
            return self.greedy_action(state)
        else:
            return self.random_action()

    def greedy_action(self, state):
        # Is FORWARD reward is bigger?
        if self.q_table[FORWARD][state] > self.q_table[BACKWARD][state]:
            return FORWARD
        # Is BACKWARD reward is bigger?
        elif self.q_table[BACKWARD][state] > self.q_table[FORWARD][state]:
            return BACKWARD
        # Rewards are equal, take random action
        return FORWARD if random.random() < 0.5 else BACKWARD

    def random_action(self):
        return FORWARD if random.random() < 0.5 else BACKWARD

    def update(self, old_state, new_state, action, reward):
        self.iteration_count += 1
        print(self.iteration_count)
        # Old Q-table value
        old_value = self.q_table[action][old_state]
        # What would be our best next action?
        future_action = self.greedy_action(new_state)
        # What is reward for the best next action?
        future_reward = self.q_table[future_action][new_state]

        # Main Q-table updating algorithm
        new_value = old_value + self.learning_rate * (reward + self.discount * future_reward - old_value)
        self.q_table[action][old_state] = new_value

        # Finally shift our exploration_rate toward zero (less gambling)
        if self.iteration_count == 5:
            self.exploration_rate = 0.5
            print('Switch exploration rate at iter', self.iteration_count, 'to rate', self.exploration_rate)
        elif self.iteration_count == 15:
            self.exploration_rate = 0.1
            print('Switch exploration rate at iter', self.iteration_count, 'to rate', self.exploration_rate)
        elif self.iteration_count == 115:
            self.exploration_rate = 0.01
            print('Switch exploration rate at iter', self.iteration_count, 'to rate', self.exploration_rate)
        else:
            self.exploration_rate = 0.001
