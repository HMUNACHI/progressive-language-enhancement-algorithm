import environment_ljh as en_ljh
from tabular import *
import numpy as np


# =====================================
def testMovement(env):
    actions = ['w', 's', 'a', 'd']  # Numpad directions

    env.render()
    done = False
    while not done:
        c = input( '\nMove: ')
        if c == 'q':
            break

        if c not in actions:
            print("Wrong action. Invalid actions are: w,s,a,d. q for quit.")

        state, r, done = env.step(actions.index(c))
        print("Now state: {}, reward: {}".format(state, r))
        env.render()


# =====================================
def generateManualPolicy(env):
    policy = np.zeros((env.n_states, env.n_actions))
    policy[:, 0] = np.ones(env.n_states)
    '''
    policy[0][0] = 0.25
    policy[0][1] = 0.25
    policy[0][2] = 0.25
    policy[0][3] = 0.25

    policy[9][0] = 0.25
    policy[9][1] = 0.25
    policy[9][2] = 0.25
    policy[9][3] = 0.25
    '''

    return policy


def testPolicyEvaluation(env):
    policy = generateManualPolicy(env)
    gamma = 0.9
    theta = 0.001
    max_iteration = 1000
    v_pai = policy_evaluation(env, policy, gamma, theta, max_iteration)
    print(v_pai)


def testPolicyImprove(env):
    policy = generateManualPolicy(env)
    gamma = 0.9
    theta = 0.001
    max_iteration = 1000
    v_pai = policy_evaluation(env, policy, gamma, theta, max_iteration)
    improvedPolicy = policy_improvement(env, policy, v_pai, gamma)
    print(improvedPolicy)

    env.runPolicy(improvedPolicy)


def testPolicyIteration(env):
    gamma = 0.9
    theta = 0.001
    max_iteration = 1000
    policy, value = policy_iteration(env, gamma, theta, max_iteration)
    print(policy)
    print(value)

    env.runPolicy(policy)


def testValueIteration(env):
    gamma = 0.9
    theta = 0.001
    max_iteration = 1000
    policy, value = value_iteration(env, gamma, theta, max_iteration)
    print(policy)
    print(value)

    env.runPolicy(policy)


# =====================================
if __name__ == '__main__':
    env = en_ljh.GridWorld()
    env.reset()
    # testMovement(env)
    # testPolicyEvaluation(env)
    # testPolicyImprove(env)
    # testPolicyIteration(env)
    testValueIteration(env)

