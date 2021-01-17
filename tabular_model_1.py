from environment import *
import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO
    iterations = 0
    # limit algorithm to max_interactions
    while iterations < max_iterations:
        iterations += 1
        # initialize delta to 0
        delta = 0

        # loop through each state
        for state in range(env.n_states-1):
            v = value[state]

            # calculate new values
            newValue = 0
            for action in range(env.n_actions):

                # Calculate probabilities
                sigma = 0
                p = [env.p(next_state, state, action) for next_state in range(env.n_states)]
                for next_state in range(len(p)):
                    sigma += p[next_state] * (env.r(next_state, state, action) + gamma * value[next_state])
                newValue += policy[state][action] * sigma
            value[state] = newValue

            # Get max delta post-iteration
            delta = max(delta, np.abs(v - value[state]))
        # limit delta by tolerance theta
        if delta < theta:
            break

    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros((env.n_states, env.n_actions), dtype=int)

    for state in range(env.n_states-1):
        action_value = -1
        max_a = -1

        for action in range(env.n_actions):
            sigma = 0
            p = [env.p(next_state, state, action) for next_state in range(env.n_states)]
            for next_states in range(len(p)):
                sigma += p[next_states] * (env.r(next_states, state, action) + gamma * value[next_states])

            # compare sigma and Q
            if sigma > action_value:
                action_value = sigma
                max_a = action

        policy[state][max_a] = 1

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    value = np.zeros(env.n_states, dtype=float)
    if policy is None:
        policy = np.zeros((env.n_states, env.n_actions), dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # TODO
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        v_pi = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy = policy_improvement(env, v_pi, gamma)
        delta = 0

        for state in range(env.n_states):
            delta = max(delta, np.abs(v_pi[state] - value[state]))

        if delta < theta:
            break

        value = v_pi
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    policy = np.zeros((env.n_states, env.n_actions))
    if value is None:
        value = np.zeros(env.n_states, dtype=float)
    else:
        value = np.array(value, dtype=float)

    # TODO
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        delta = 0
        for state in range(env.n_states-1):
            v = value[state]
            max_v = -1
            for action in range(env.n_actions):
                sigma = 0
                p = [env.p(next_state, state, action) for next_state in range(env.n_states-1)]
                for next_state in range(len(p)):
                    sigma += p[next_state] * (env.r(next_state, state, action) + gamma * value[next_state])
                if sigma > max_v:
                    max_v = sigma

            value[state] = max_v

            delta = max(delta, np.abs(v - value[state]))

        if delta < theta:
            break

    for state in range(env.n_states-1):
        max_a = -1
        max_v = -1
        for action in range(env.n_actions):
            sigma = 0
            p = [env.p(next_state, state, action) for next_state in range(env.n_states)]
            for ns in range(len(p)):
                sigma += p[ns] * (env.r(next_state, state, action) + gamma * value[next_state])
            if sigma > max_v:
                max_v = sigma
                max_a = action

        policy[state][max_a] = 1
    return policy, value

