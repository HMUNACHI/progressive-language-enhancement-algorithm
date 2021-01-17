from environment import *
import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO
    # create an array of states
    states = [i for i in range(env.n_states - 1)]
    # get the p and r functions
    p = env.p
    r = env.r

    # initialise the iteration counter to 0
    iterations = 0
    # create an infinite loop
    while True:
        # increment iteration counter
        iterations += 1
        # initialise delta to 0
        delta = 0

        # update the value function
        for s in states:
            v = value[s]
            value[s] = sum([p(next_s, s, policy[s]) * (r(next_s, s, policy[s]) + gamma * value[next_s]) for next_s in states])
            delta = max(delta, abs(v - value[s]))

        # break infinite loop on tolerance limit or max_iterations
        if delta < theta or iterations >= max_iterations:
            #print("Policy evaluation completed on", iterations, "iterations")
            break

    print(value)
    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    # TODO
    # create arrays of states and actions
    states = [i for i in range(env.n_states - 1)]
    actions = [i for i in range(env.n_actions)]
    # get the p and r functions
    p = env.p
    r = env.r

    for s in states:
        policy[s] = actions[np.argmax(
            [sum([p(next_s, s, a) * (r(next_s, s,a) + gamma * value[next_s]) for next_s in states]) for a in
             actions])]

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    value = np.zeros(env.n_states, dtype=float)
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # TODO
    states = [i for i in range(env.n_states - 1)]
    iterations = 0

    while True:
        iterations += 1
        v_pi = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, v_pi, gamma)

        delta = 0
        for s in states:
            delta = max(delta, np.abs(v_pi[s] - value[s]))

        # break infinite loop on tolerance limit or max_iterations
        if delta < theta or iterations >= max_iterations:
            print("Policy iteration completed in", iterations, "iterations")
            break
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    policy = np.zeros(env.n_states, dtype=int)
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=float)

    # TODO
    # create arrays of states and actions
    states = [i for i in range(env.n_states - 1)]
    actions = [i for i in range(env.n_actions)]
    # get the p and r functions
    p = env.p
    r = env.r
    iterations = 0

    while True:
        iterations += 1
        delta = 0

        for s in range(env.n_states - 1):
            v = value[s]
            value[s] = max([sum(
                [env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in states])
                for a in actions])

            delta = max(delta, np.abs(v - value[s]))

        if delta < theta or iterations >= max_iterations:
            print("Value iteration completed in", iterations, "iterations")
            break

    for s in states:
        policy[s] = actions[np.argmax(
            [sum([p(s, next_s, a) * (r(s, next_s, a) + gamma * value[next_s]) for next_s in states]) for a in
             actions])]

    return policy, value

