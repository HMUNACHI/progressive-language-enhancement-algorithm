from abc import ABC
import operator

import numpy as np

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        return next_state, reward


class Environment(EnvironmentModel, ABC):
    def __init__(self, n_states, n_actions, max_steps, dist, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        self.state = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.dist)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        self.state, reward = self.draw(self.state, action)
        return self.state, reward, done


class GridWorld(Environment, ABC):
    ABSORBING_STATE = 11
    REWARD_STATE = 10
    TRAPPING_STATE = 6

    def __init__(self):
        self.row = 3
        self.column = 4
        self.worldMap = np.zeros((self.row, self.column))
        self.worldMap[1][1] = -np.inf
        self.worldMap[1][3] = -1
        self.worldMap[2][3] = 1

        self.actionMovement = {
            ACTION_UP: (0, 1),
            ACTION_DOWN: (0, -1),
            ACTION_LEFT: (-1, 0),
            ACTION_RIGHT: (1, 0)
        }

        self.coord2State = {
            (0, 0): 0,
            (1, 0): 1,
            (2, 0): 2,
            (3, 0): 3,
            (0, 1): 4,
            (2, 1): 5,
            (3, 1): 6,
            (0, 2): 7,
            (1, 2): 8,
            (2, 2): 9,
            (3, 2): 10,
        }

        self.state2Coord = {}
        for k, v in self.coord2State.items():
            self.state2Coord[v] = k

        self.nowPos = (0, 0)

        dist = np.zeros(12)
        # Start state is state 0
        dist[0] = 1
        max_step = 100
        Environment.__init__(self, 12, 4, max_step, dist)

    def reset(self):
        Environment.reset(self)
        # Update nowPos
        self.nowPos = self.state2Coord[self.state]

        return self.state

    def p(self, next_state, state, action):
        if state == self.REWARD_STATE or state == self.TRAPPING_STATE or state == self.ABSORBING_STATE:
            if next_state == self.ABSORBING_STATE:
                return 1
            return 0

        if next_state == self.ABSORBING_STATE:
            return 0

        stateCoord = self.state2Coord[state]
        newCoord = tuple(map(operator.add, stateCoord, self.actionMovement[action]))
        if newCoord not in self.coord2State:
            if next_state == state:
                return 1
            return 0

        coordNextState = self.state2Coord[next_state]
        if coordNextState == newCoord:
            return 1

        return 0

    def r(self, next_state, state, action):
        if state == self.REWARD_STATE:
            return 1

        if state == self.TRAPPING_STATE:
            return -1

        return 0

    def step(self, action):
        state, r, done = Environment.step(self, action)
        # Update nowPos
        if self.state != self.ABSORBING_STATE:
            self.nowPos = self.state2Coord[self.state]
        else:
            print("!!!Stuck in absorbing state!!!")
            done = True

        return state, r, done

    def runPolicy(self, policy):
        print("===Start running policy===")
        self.render()
        done = False
        while not done:
            p = [policy[self.state][act] for act in range(self.n_actions)]
            a = self.random_state.choice(self.n_actions, p=p)
            state, r, done = self.step(a)
            print("Now state: {}, reward: {}".format(state, r))
            self.render()
        print("==End of running policy===")

    def render(self):
        for i in reversed(range(0, self.row)):
            for j in range(0, self.column):
                if self.nowPos[0] == j and self.nowPos[1] == i:
                    print('@ ', end="")
                else:
                    print("* ", end="")

            print("")







