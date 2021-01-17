from environment_ljh import *
import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    iterationNum = 0
    while True:
        iterationNum += 1
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            # Calculate new value
            newValue = 0
            for a in range(env.n_actions):
                # Get the probability of next state as all states
                sigma = 0
                p = [env.p(ns, s, a) for ns in range(env.n_states)]
                for ns in range(len(p)):
                    sigma += p[ns] * (env.r(ns, s, a) + gamma * value[ns])
                newValue += policy[s][a] * sigma
            value[s] = newValue

            # Get the max delta after iteration
            delta = max(delta, np.abs(v - value[s]))


        # Check delta to determine if break or not
        if delta < theta:
            print("[Policy evaluation] iteration times: {}".format(iterationNum))
            break

        if iterationNum >= max_iterations:
            break

    return value


def policy_improvement(env, policy, value, gamma):
    improved_policy = np.zeros((env.n_states, env.n_actions), dtype=int)

    for s in range(env.n_states - 1):
        q = -1
        a_maxq = -1
        for a in range(env.n_actions):
            sigma = 0
            p = [env.p(ns, s, a) for ns in range(env.n_states)]
            for ns in range(len(p)):
                sigma += p[ns] * (env.r(ns, s, a) + gamma * value[ns])

            if sigma > q:
                q = sigma
                a_maxq = a

        # Set action into pai prime
        improved_policy[s][a_maxq] = 1

    return improved_policy


def policy_iteration(env, gamma, theta, max_iterations):
    policy = np.zeros((env.n_states, env.n_actions))
    policy[:, 3] = np.ones(env.n_states)

    value = np.zeros(env.n_states, dtype=int)
    iterationNum = 0
    while True:
        iterationNum += 1
        # Calculate value for policy
        v_pai = policy_evaluation(env, policy, gamma, theta, max_iterations)
        # Improve policy
        policy = policy_improvement(env, policy, v_pai, gamma)
        # Compare value
        delta = 0
        for s in range(env.n_states):
            # Get the max delta after iteration
            delta = max(delta, np.abs(v_pai[s] - value[s]))

        # Check delta to determine if break or not
        if delta < theta:
            print("[Policy iteration] iteration times: {}".format(iterationNum))
            break

        if iterationNum >= max_iterations:
            break

        # Update value
        value = v_pai

    return policy, value


def value_iteration(env, gamma, theta, max_iterations):
    policy = np.zeros((env.n_states, env.n_actions))

    value = np.zeros(env.n_states, dtype=np.float)
    iterationNum = 0
    while True:
        iterationNum += 1
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            maxV = -1
            for a in range(env.n_actions):
                sigma = 0
                p = [env.p(ns, s, a) for ns in range(env.n_states)]
                for ns in range(len(p)):
                    sigma += p[ns] * (env.r(ns, s, a) + gamma * value[ns])
                if sigma > maxV:
                    maxV = sigma

            value[s] = maxV

            delta = max(delta, np.abs(v - value[s]))

        if delta < theta:
            print("[Value iteration] iteration times: {}".format(iterationNum))
            break

        if iterationNum >= max_iterations:
            break

    for s in range(env.n_states):
        a_maxV = -1
        maxV = -1
        for a in range(env.n_actions):
            sigma = 0
            p = [env.p(ns, s, a) for ns in range(env.n_states)]
            for ns in range(len(p)):
                sigma += p[ns] * (env.r(ns, s, a) + gamma * value[ns])
            if sigma > maxV:
                maxV = sigma
                a_maxV = a

        policy[s][a_maxV] = 1


    return policy, value


